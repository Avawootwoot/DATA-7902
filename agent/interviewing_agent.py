import json
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    from state import InterviewState
except ImportError:
    from .state import InterviewState


STARTING_QUESTION = (
    "Hello, thank you for agreeing to speak with me today. "
    "To get started, could you tell me a little bit about where you grew up "
    "and what your life was like as a child?"
)


class InterviewingAgent:
    def __init__(self, model: str, interviewer_prompt_path: str):
        self.client = OpenAI()
        self.model = model
        self.base_prompt = Path(interviewer_prompt_path).read_text(encoding="utf-8").strip()

    def first_question(self) -> str:
        return STARTING_QUESTION

    def ask(
        self,
        ground_facts: Dict[str, Any],
        state: InterviewState,
        intent: str,
        focus: Optional[str] = None,
    ) -> str:
        state_summary = {
            "turn_idx": state.turn_idx,
            "covered_topics": state.covered_topics[-10:],
            "open_threads": state.open_threads[-10:],
            "facts_keys": list(state.facts.keys())[-15:],
            "recent_flags": state.flags[-5:],
            "intent": intent,
            "focus": focus,
        }

        context = f"""
GROUND_FACTS (trusted context; do not invent beyond this):
{json.dumps(ground_facts, ensure_ascii=False)}

STATE_SUMMARY:
{json.dumps(state_summary, ensure_ascii=False)}

POLICY:
- Ask exactly one question.
- If intent=clarify_flag, ask a gentle clarification question about the flagged issue.
- If intent=follow_thread, ask a follow-up about the focus thread.
- If intent=cover_topic, ask an open-ended question about the focus topic.
- If intent=advance_story, ask the next sensible life-story question.
- If intent=wrap_up, thank the user warmly and say exactly: [INTERVIEW COMPLETE]
""".strip()

        messages = [{"role": "developer", "content": self.base_prompt}]
        messages.append({"role": "user", "content": context})
        messages.extend(state.transcript[-8:])

        resp = self.client.responses.create(
            model=self.model,
            input=messages,
        )
        return resp.output_text.strip()