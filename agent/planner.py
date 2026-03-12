from typing import Optional

try:
    from state import InterviewState
except ImportError:
    from .state import InterviewState


class Planner:
    REQUIRED_TOPICS = [
        "childhood",
        "family",
        "daily_routines",
        "technology",
        "pain_points",
    ]

    CLARIFICATION_FLAG_TYPES = {"contradiction", "missing_time", "missing_place"}

    def __init__(self, target_turns: int = 6):
        self.target_turns = target_turns

    def _latest_actionable_flag(self, state: InterviewState):
        for flag in reversed(state.flags):
            if flag.get("type") in self.CLARIFICATION_FLAG_TYPES:
                return flag
        return None

    def decide_next_intent(self, state: InterviewState) -> str:
        if state.turn_idx >= self.target_turns - 1:
            return "wrap_up"

        actionable_flag = self._latest_actionable_flag(state)
        if actionable_flag:
            return "clarify_flag"

        if state.open_threads:
            return "follow_thread"

        missing_topics = [t for t in self.REQUIRED_TOPICS if t not in state.covered_topics]
        if missing_topics:
            return "cover_topic"

        return "advance_story"

    def choose_focus(self, state: InterviewState) -> Optional[str]:
        actionable_flag = self._latest_actionable_flag(state)
        if actionable_flag:
            return actionable_flag.get("detail")

        if state.open_threads:
            return state.open_threads[-1]

        missing_topics = [t for t in self.REQUIRED_TOPICS if t not in state.covered_topics]
        if missing_topics:
            return missing_topics[0]

        return None

    def should_end(self, state: InterviewState) -> bool:
        return state.turn_idx >= self.target_turns - 1