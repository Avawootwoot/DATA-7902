import argparse
import json
from pathlib import Path

from agent.persona_loader import load_persona, build_ground_facts
from agent.dialogue_manager import DialogueManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona_id", required=True)
    ap.add_argument("--personas_path", default="personas.json")
    ap.add_argument("--prompt_path", default="The Interviewer Prompt.txt")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--turns", type=int, default=8)
    args = ap.parse_args()

    persona = load_persona(args.personas_path, args.persona_id)
    ground_facts = build_ground_facts(persona)

    dm = DialogueManager(model=args.model, interviewer_prompt_path=args.prompt_path)
    state = dm.start(persona_id=args.persona_id)

    q = dm.get_opening_question(state)
    print(f"\n[Interviewer] {q}\n")

    for _ in range(args.turns):
        if state.completed:
            break

        user_answer = input("[User] ").strip()
        if not user_answer:
            continue
        if user_answer.lower() in {"exit", "quit"}:
            break

        q = dm.on_user_turn(ground_facts, state, user_answer)
        print(f"\n[Interviewer] {q}\n")

        if q.strip() == "[INTERVIEW COMPLETE]":
            break

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"{args.persona_id}_run.json"
    out.write_text(json.dumps(state.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()