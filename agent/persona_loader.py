import json
from pathlib import Path
from typing import Any, Dict, List


def _read_persona_payload(personas_path: str) -> List[Dict[str, Any]]:
    raw = json.loads(Path(personas_path).read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "personas" in raw:
        personas = raw["personas"]
    else:
        personas = raw

    if not isinstance(personas, list):
        raise ValueError("personas.json must contain a list or a dict with a 'personas' list")

    validated: List[Dict[str, Any]] = []
    for idx, persona in enumerate(personas):
        if not isinstance(persona, dict):
            raise ValueError(f"Persona at index {idx} is not a JSON object")
        validated.append(persona)

    return validated


def load_all_personas(personas_path: str) -> List[Dict[str, Any]]:
    return _read_persona_payload(personas_path)


def load_persona(personas_path: str, persona_id: str) -> Dict[str, Any]:
    personas = _read_persona_payload(personas_path)

    for persona in personas:
        if str(persona.get("persona_id")) == str(persona_id):
            return persona

    raise KeyError(f"persona_id={persona_id} not found")


def build_ground_facts(persona: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "persona_id": persona.get("persona_id"),
        "name": persona.get("full_name") or persona.get("name"),
        "age": persona.get("age"),
        "background_summary": persona.get("background_summary"),
        "key_life_events": persona.get("key_life_events", []),
        "personality_traits": persona.get("personality_traits", []),
    }


def list_persona_ids(personas_path: str) -> List[str]:
    personas = _read_persona_payload(personas_path)
    return [
        str(persona.get("persona_id"))
        for persona in personas
        if persona.get("persona_id") is not None
    ]