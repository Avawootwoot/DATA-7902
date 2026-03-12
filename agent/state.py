from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class InterviewState:
    persona_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_idx: int = 0
    status: str = "active"
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # memory
    facts: Dict[str, str] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    open_threads: List[str] = field(default_factory=list)
    covered_topics: List[str] = field(default_factory=list)
    flags: List[Dict[str, str]] = field(default_factory=list)

    # planning
    last_intent: Optional[str] = None
    last_focus: Optional[str] = None
    completed: bool = False

    # full transcript
    transcript: List[Dict[str, str]] = field(default_factory=list)

    def add_flag(self, flag_type: str, detail: str) -> None:
        flag = {"type": flag_type, "detail": detail}
        if flag not in self.flags:
            self.flags.append(flag)