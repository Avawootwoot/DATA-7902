import json
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.persona_loader import load_all_personas, build_ground_facts
from agent.dialogue_manager import DialogueManager


PERSONAS_PATH = "personas.json"
INTERVIEWER_PROMPT_PATH = "The Interviewer Prompt.txt"
DB_PATH = "dialogue.db"
MODEL_NAME = "gpt-5.2"

# Full interviews need more than 3 turns.
MAX_TURNS = 6

OUTPUT_RESULTS_PATH = "batch_run_results.json"
OUTPUT_SUMMARY_PATH = "batch_run_summary.json"

# Keep these because your account is rate-limited.
TURN_SLEEP_SECONDS = 22
PERSONA_SLEEP_SECONDS = 25


def safe_name(persona: Dict[str, Any]) -> str:
    return str(persona.get("full_name") or persona.get("name") or "Unknown")


def normalize_transcript(transcript: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "role": str(turn.get("role", "")),
            "content": str(turn.get("content", "")),
        }
        for turn in transcript
    ]


def run_batch() -> List[Dict[str, Any]]:
    personas = load_all_personas(PERSONAS_PATH)

    dm = DialogueManager(
        model=MODEL_NAME,
        interviewer_prompt_path=INTERVIEWER_PROMPT_PATH,
        db_path=DB_PATH,
        target_turns=MAX_TURNS,
    )

    results: List[Dict[str, Any]] = []

    print(f"Loaded {len(personas)} personas.", flush=True)
    print(f"Running auto interviews with max_turns={MAX_TURNS} ...", flush=True)

    for idx, persona in enumerate(personas, start=1):
        persona_id = str(persona.get("persona_id"))
        name = safe_name(persona)

        print(f"[{idx}/{len(personas)}] Running interview for {persona_id} - {name}", flush=True)

        try:
            ground_facts = build_ground_facts(persona)
            state = dm.start(persona_id)

            if not state.transcript:
                dm.get_opening_question(state)
                print("  Opening question created.", flush=True)

            turns_run = 0
            while not state.completed and turns_run < MAX_TURNS:
                print(f"  Turn {turns_run + 1}/{MAX_TURNS} ...", flush=True)

                dm.run_persona_turn(
                    persona=persona,
                    ground_facts=ground_facts,
                    state=state,
                )
                turns_run += 1

                print(
                    f"    Transcript rows: {len(state.transcript)} | "
                    f"Covered topics: {len(state.covered_topics)} | "
                    f"Completed: {state.completed}",
                    flush=True,
                )

                if not state.completed and turns_run < MAX_TURNS:
                    print(f"    Sleeping {TURN_SLEEP_SECONDS}s to avoid RPM limits...", flush=True)
                    time.sleep(TURN_SLEEP_SECONDS)

            if state.completed:
                final_status = "completed"
            elif turns_run >= MAX_TURNS:
                final_status = "max_turns_reached"
            else:
                final_status = str(state.status)

            results.append(
                {
                    "persona_id": persona_id,
                    "name": name,
                    "completed": bool(state.completed),
                    "status": final_status,
                    "turn_idx": int(state.turn_idx),
                    "turns_run": turns_run,
                    "turns_recorded": len(state.transcript),
                    "covered_topics": list(state.covered_topics),
                    "open_threads": list(state.open_threads),
                    "flags": list(state.flags),
                    "facts": dict(state.facts),
                    "timeline": list(state.timeline),
                    "transcript": normalize_transcript(state.transcript),
                }
            )

            print(f"  Finished {persona_id} with status={final_status}.", flush=True)

        except Exception as e:
            results.append(
                {
                    "persona_id": persona_id,
                    "name": name,
                    "completed": False,
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)}",
                    "turn_idx": 0,
                    "turns_run": 0,
                    "turns_recorded": 0,
                    "covered_topics": [],
                    "open_threads": [],
                    "flags": [],
                    "facts": {},
                    "timeline": [],
                    "transcript": [],
                }
            )
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)

        if idx < len(personas):
            print(f"  Sleeping {PERSONA_SLEEP_SECONDS}s before next persona...", flush=True)
            time.sleep(PERSONA_SLEEP_SECONDS)

    return results


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    completed = sum(1 for r in results if r.get("status") == "completed")
    errored = sum(1 for r in results if r.get("status") == "error")
    max_turns_reached = sum(1 for r in results if r.get("status") == "max_turns_reached")

    return {
        "total_personas": total,
        "completed_interviews": completed,
        "max_turns_reached": max_turns_reached,
        "error_count": errored,
        "model": MODEL_NAME,
        "max_turns": MAX_TURNS,
        "rows": [
            {
                "persona_id": r.get("persona_id"),
                "name": r.get("name"),
                "completed": r.get("completed"),
                "status": r.get("status"),
                "turns_run": r.get("turns_run", 0),
                "turns_recorded": r.get("turns_recorded", 0),
                "covered_topics_count": len(r.get("covered_topics", [])),
                "flags_count": len(r.get("flags", [])),
            }
            for r in results
        ],
    }


def save_json(path: str, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    results = run_batch()
    summary = build_summary(results)

    save_json(OUTPUT_RESULTS_PATH, results)
    save_json(OUTPUT_SUMMARY_PATH, summary)

    print("\nBatch run complete.", flush=True)
    print(f"Saved full results to: {OUTPUT_RESULTS_PATH}", flush=True)
    print(f"Saved summary to: {OUTPUT_SUMMARY_PATH}", flush=True)
    print(
        f"Completed: {summary['completed_interviews']}/{summary['total_personas']} | "
        f"Max-turns: {summary['max_turns_reached']} | "
        f"Errors: {summary['error_count']}",
        flush=True,
    )