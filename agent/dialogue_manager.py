from typing import Any, Dict, Optional, Tuple

try:
    from state import InterviewState
    from extraction_agent import ExtractionAgent
    from fact_checker import FactChecker
    from planner import Planner
    from interviewing_agent import InterviewingAgent
    from persona_agent import PersonaAgent
    from recorder import TranscriptRecorder
except ImportError:
    from .state import InterviewState
    from .extraction_agent import ExtractionAgent
    from .fact_checker import FactChecker
    from .planner import Planner
    from .interviewing_agent import InterviewingAgent
    from .persona_agent import PersonaAgent
    from .recorder import TranscriptRecorder


class DialogueManager:
    """
    Orchestrator / Control Loop:
    - routes data between agents
    - updates state
    - records transcript + planner memory
    """

    def __init__(
        self,
        model: str,
        interviewer_prompt_path: str,
        persona_prompt_path: Optional[str] = None,
        db_path: str = "dialogue.db",
        target_turns: int = 6,
    ):
        self.extractor = ExtractionAgent(model=model)
        self.checker = FactChecker()
        self.planner = Planner(target_turns=target_turns)
        self.interviewer = InterviewingAgent(
            model=model,
            interviewer_prompt_path=interviewer_prompt_path,
        )
        self.persona_agent = PersonaAgent(
            model=model,
            persona_prompt_path=persona_prompt_path,
        )
        self.recorder = TranscriptRecorder(db_path=db_path)

    def start(self, persona_id: str) -> InterviewState:
        state = InterviewState(persona_id=persona_id)
        self.recorder.create_session(
            session_id=state.session_id,
            persona_id=state.persona_id,
            status=state.status,
            started_at=state.started_at,
        )
        return state

    def get_opening_question(self, state: InterviewState) -> str:
        question = self.interviewer.first_question()
        state.transcript.append({"role": "assistant", "content": question})
        self.recorder.save_turn(
            session_id=state.session_id,
            turn_idx=state.turn_idx,
            role="assistant",
            content=question,
            intent="opening",
            focus="childhood",
        )
        return question

    @staticmethod
    def _flag_key(flag: Dict[str, str]) -> Tuple[str, str]:
        return (
            str(flag.get("type", "")).strip(),
            str(flag.get("detail", "")).strip(),
        )

    def _merge_flags(self, state: InterviewState, new_flags):
        seen = {self._flag_key(flag) for flag in state.flags}
        for flag in new_flags:
            key = self._flag_key(flag)
            if key not in seen:
                state.flags.append(flag)
                seen.add(key)

    def _process_answer(
        self,
        ground_facts: Dict[str, Any],
        state: InterviewState,
        answer_text: str,
    ) -> str:
        if state.completed:
            return "[INTERVIEW COMPLETE]"

        # Log respondent turn
        state.transcript.append({"role": "user", "content": answer_text})
        self.recorder.save_turn(
            session_id=state.session_id,
            turn_idx=state.turn_idx,
            role="user",
            content=answer_text,
        )

        # 1) Extract structured information from answer
        extracted = self.extractor.extract(ground_facts, answer_text)
        self.recorder.save_extraction(state.session_id, state.turn_idx, extracted)

        # 2) Validate extracted information
        rule_flags = self.checker.validate(extracted, ground_facts)
        extracted_flags = extracted.get("flags", [])
        self._merge_flags(state, extracted_flags + rule_flags)

        # 3) Update state memory
        for fact in extracted.get("facts", []):
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            if key and value:
                state.facts[key] = value

        existing_timeline = {
            (item.get("year"), item.get("event"), item.get("location"))
            for item in state.timeline
        }
        for event in extracted.get("timeline_events", []):
            event_key = (event.get("year"), event.get("event"), event.get("location"))
            if event_key not in existing_timeline:
                state.timeline.append(event)
                existing_timeline.add(event_key)

        for thread in extracted.get("open_threads", []):
            if thread and thread not in state.open_threads:
                state.open_threads.append(thread)

        for topic in extracted.get("covered_topics", []):
            if topic and topic not in state.covered_topics:
                state.covered_topics.append(topic)

        # 4) Plan next move
        intent = self.planner.decide_next_intent(state)
        focus = self.planner.choose_focus(state)
        state.last_intent = intent
        state.last_focus = focus

        # 5) Decide whether to end or ask next question
        if self.planner.should_end(state) or intent == "wrap_up":
            question = "[INTERVIEW COMPLETE]"
            state.completed = True
            state.status = "completed"
        else:
            question = self.interviewer.ask(
                ground_facts=ground_facts,
                state=state,
                intent=intent,
                focus=focus,
            )

            if question.strip() == "[INTERVIEW COMPLETE]":
                state.completed = True
                state.status = "completed"

        # Log interviewer turn
        state.transcript.append({"role": "assistant", "content": question})
        self.recorder.save_turn(
            session_id=state.session_id,
            turn_idx=state.turn_idx,
            role="assistant",
            content=question,
            intent=intent,
            focus=focus,
        )

        # Advance turn and persist state
        state.turn_idx += 1
        self.recorder.save_state(state)

        if state.completed:
            self.recorder.close_session(state.session_id)

        return question

    def on_user_turn(
        self,
        ground_facts: Dict[str, Any],
        state: InterviewState,
        user_answer: str,
    ) -> str:
        return self._process_answer(
            ground_facts=ground_facts,
            state=state,
            answer_text=user_answer,
        )

    def run_persona_turn(
        self,
        persona: Dict[str, Any],
        ground_facts: Dict[str, Any],
        state: InterviewState,
    ) -> str:
        if state.completed:
            return "[INTERVIEW COMPLETE]"

        if not state.transcript:
            raise ValueError(
                "Cannot run persona turn before an opening question exists. "
                "Call get_opening_question(state) first."
            )

        latest_turn = state.transcript[-1]
        if latest_turn.get("role") != "assistant":
            raise ValueError("Expected the latest transcript turn to be an interviewer question.")

        interviewer_question = latest_turn.get("content", "").strip()
        if not interviewer_question:
            raise ValueError("Latest interviewer question is empty.")

        persona_answer = self.persona_agent.answer(
            persona=persona,
            state=state,
            interviewer_question=interviewer_question,
        )

        return self._process_answer(
            ground_facts=ground_facts,
            state=state,
            answer_text=persona_answer,
        )

    def run_auto_interview(
        self,
        persona: Dict[str, Any],
        ground_facts: Dict[str, Any],
        state: InterviewState,
        max_turns: int = 20,
    ) -> InterviewState:
        if not state.transcript:
            self.get_opening_question(state)

        turns_run = 0
        while not state.completed and turns_run < max_turns:
            self.run_persona_turn(
                persona=persona,
                ground_facts=ground_facts,
                state=state,
            )
            turns_run += 1

        return state