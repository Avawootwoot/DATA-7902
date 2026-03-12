EXTRACT_SCHEMA = {
    "name": "extract_turn",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["key", "value", "confidence"],
                },
            },
            "timeline_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "year": {"type": ["integer", "null"]},
                        "event": {"type": "string"},
                        "location": {"type": ["string", "null"]},
                    },
                    "required": ["year", "event", "location"],
                },
            },
            "open_threads": {"type": "array", "items": {"type": "string"}},
            "covered_topics": {"type": "array", "items": {"type": "string"}},
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["type", "detail"],
                },
            },
        },
        "required": [
            "summary",
            "facts",
            "timeline_events",
            "open_threads",
            "covered_topics",
            "flags",
        ],
    },
    "strict": True,
}