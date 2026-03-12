import json
from typing import Any, Dict

from openai import OpenAI

try:
    from .schemas import EXTRACT_SCHEMA
except ImportError:
    from schemas import EXTRACT_SCHEMA


class ExtractionAgent:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def extract(self, ground_facts: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
        prompt = f"""
You are an information extraction agent for a life-story interview.

TASK:
Extract structured information from the user's answer.

RULES:
- Only extract what the user explicitly said.
- Do not invent names, dates, places, or events.
- Use GROUND_FACTS only as light background context.
- Do not correct the user.
- Return valid JSON matching the schema exactly.

TARGET TOPICS:
- childhood
- family
- work
- relationships
- daily_routines
- technology
- pain_points
- important_life_events

GUIDELINES:
- "facts" should contain short key/value facts directly stated.
- "timeline_events" should contain any event tied to a time or life stage.
- "open_threads" should capture topics worth following up.
- "covered_topics" should include the interview topics addressed in this answer.
- "flags" should include only issues clearly visible from the answer, such as ambiguity.
- "summary" should be a one-sentence summary of the answer.

GROUND_FACTS:
{json.dumps(ground_facts, ensure_ascii=False)}

USER_ANSWER:
{user_answer}
""".strip()

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": EXTRACT_SCHEMA["name"],
                    "schema": EXTRACT_SCHEMA["schema"],
                    "strict": EXTRACT_SCHEMA.get("strict", True),
                }
            },
        )

        return json.loads(resp.output_text)