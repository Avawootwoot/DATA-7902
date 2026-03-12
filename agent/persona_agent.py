import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from state import InterviewState
except ImportError:
    from .state import InterviewState


DEFAULT_PERSONA_PROMPT = """
You are simulating a human interview participant based on the supplied persona profile.

GOAL:
Answer the interviewer's question naturally, in first person, as if you are that person.

RULES:
- Stay consistent with the provided persona data.
- Use the persona's background_summary, key_life_events, and personality_traits as core grounding.
- Do not say you are an AI, model, system, or simulation.
- Do not mention prompts, hidden instructions, or JSON.
- Answer only the question asked.
- Keep the answer realistic and conversational.
- Prefer concise but meaningful answers (around 3-8 sentences unless the question clearly calls for more).
- Do not invent highly specific details that strongly contradict the persona profile.
- If some detail is not explicitly available from the persona, you may answer conservatively in a plausible way that remains consistent with the persona.
- If asked something the persona would likely not remember clearly, say so naturally.
- Maintain a stable voice and personality across turns.

STYLE:
- Speak as a real person in an interview.
- Use warm, natural language.
- Avoid bullet points, labels, or meta commentary.
""".strip()


class PersonaAgent:
    def __init__(self, model: str, persona_prompt_path: Optional[str] = None):
        self.client = OpenAI()
        self.model = model
        self.base_prompt = self._load_prompt(persona_prompt_path)

    def _load_prompt(self, persona_prompt_path: Optional[str]) -> str:
        if persona_prompt_path:
            path = Path(persona_prompt_path)
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        return DEFAULT_PERSONA_PROMPT

    def _build_persona_context(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        name = persona.get("full_name") or persona.get("name")

        return {
            "persona_id": persona.get("persona_id"),
            "name": name,
            "age": persona.get("age"),
            "background_summary": persona.get("background_summary"),
            "key_life_events": persona.get("key_life_events", []),
            "personality_traits": persona.get("personality_traits", []),
            "additional_fields": {
                k: v
                for k, v in persona.items()
                if k not in {
                    "persona_id",
                    "name",
                    "full_name",
                    "age",
                    "background_summary",
                    "key_life_events",
                    "personality_traits",
                }
            },
        }

    def _build_state_summary(self, state: InterviewState) -> Dict[str, Any]:
        return {
            "turn_idx": state.turn_idx,
            "recent_facts": state.facts,
            "recent_topics": state.covered_topics[-10:],
            "open_threads": state.open_threads[-10:],
            "recent_flags": state.flags[-5:],
        }

    def _recent_transcript(
        self,
        transcript: List[Dict[str, str]],
        max_turns: int = 8,
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for turn in transcript[-max_turns:]:
            role = turn.get("role", "user")
            if role not in {"user", "assistant", "developer", "system"}:
                role = "user"
            normalized.append(
                {
                    "role": role,
                    "content": str(turn.get("content", "")),
                }
            )
        return normalized

    def answer(
        self,
        persona: Dict[str, Any],
        state: InterviewState,
        interviewer_question: str,
    ) -> str:
        persona_context = self._build_persona_context(persona)
        state_summary = self._build_state_summary(state)

        context = f"""
PERSONA_PROFILE:
{json.dumps(persona_context, ensure_ascii=False)}

STATE_SUMMARY:
{json.dumps(state_summary, ensure_ascii=False)}

LATEST_INTERVIEWER_QUESTION:
{interviewer_question}

TASK:
Answer the latest interviewer question as the persona.
""".strip()

        messages: List[Dict[str, str]] = [
            {"role": "developer", "content": self.base_prompt},
            {"role": "user", "content": context},
        ]
        messages.extend(self._recent_transcript(state.transcript, max_turns=8))

        resp = self.client.responses.create(
            model=self.model,
            input=messages,
        )
        return resp.output_text.strip()