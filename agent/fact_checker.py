from typing import Any, Dict, List


class FactChecker:
    """
    Deterministic validation layer.
    Keep this conservative: only flag clear issues.
    """

    def validate(
        self,
        extracted: Dict[str, Any],
        ground_facts: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        flags: List[Dict[str, str]] = []

        for ev in extracted.get("timeline_events", []):
            event_text = ev.get("event", "") or "the event mentioned"

            if ev.get("year") is None:
                flags.append(
                    {
                        "type": "missing_time",
                        "detail": f"Please clarify when this happened: {event_text}",
                    }
                )

            if ev.get("location") is None:
                flags.append(
                    {
                        "type": "missing_place",
                        "detail": f"Please clarify where this happened: {event_text}",
                    }
                )

        user_facts = {
            str(f.get("key", "")).lower(): str(f.get("value", "")).strip()
            for f in extracted.get("facts", [])
            if f.get("key") and f.get("value")
        }

        background = (ground_facts.get("background_summary") or "").lower()

        # Conservative contradiction check only when the answer directly claims
        # a birthplace and it clearly conflicts with a broad rural/coastal/city signal.
        if "birthplace" in user_facts and background:
            claimed = user_facts["birthplace"].lower()
            broad_markers = ["rural", "coastal", "city", "township", "village", "regional capital"]

            background_has_marker = any(marker in background for marker in broad_markers)
            claimed_has_marker = any(marker in claimed for marker in broad_markers)

            if background_has_marker and claimed_has_marker and claimed not in background:
                flags.append(
                    {
                        "type": "contradiction",
                        "detail": "The birthplace may conflict with the known background summary.",
                    }
                )

        if (
            not extracted.get("facts")
            and not extracted.get("timeline_events")
            and len(extracted.get("open_threads", [])) == 0
        ):
            flags.append(
                {
                    "type": "low_information",
                    "detail": "The answer was vague. Ask for a more specific example, place, or time.",
                }
            )

        return flags