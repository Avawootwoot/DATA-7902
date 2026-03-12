import streamlit as st
from agent.persona_loader import load_all_personas, build_ground_facts
from agent.dialogue_manager import DialogueManager

try:
    from agent.persona_loader import (
        build_ground_facts,
        list_persona_ids,
        load_all_personas,
        load_persona,
    )
    from agent.dialogue_manager import DialogueManager
except ImportError:
    from agent.persona_loader import (
        build_ground_facts,
        list_persona_ids,
        load_all_personas,
        load_persona,
    )
    from agent.dialogue_manager import DialogueManager


PERSONAS_PATH = "personas.json"
INTERVIEWER_PROMPT_PATH = "The Interviewer Prompt.txt"
MODEL_NAME = "gpt-5.2"


st.set_page_config(layout="wide")
st.title("Generative Adaptive AI Interviewing Agent")

mode = st.sidebar.radio(
    "Mode",
    ["Manual interview", "Auto-run all personas"],
)

if "dm" not in st.session_state:
    st.session_state.dm = DialogueManager(
        model=MODEL_NAME,
        interviewer_prompt_path=INTERVIEWER_PROMPT_PATH,
        db_path="dialogue.db",
    )

dm = st.session_state.dm


def reset_manual_state(persona_id: str):
    persona = load_persona(PERSONAS_PATH, persona_id)
    ground_facts = build_ground_facts(persona)
    state = dm.start(persona_id)
    dm.get_opening_question(state)

    st.session_state.manual_persona_id = persona_id
    st.session_state.manual_persona = persona
    st.session_state.manual_ground_facts = ground_facts
    st.session_state.manual_state = state


if mode == "Manual interview":
    persona_ids = list_persona_ids(PERSONAS_PATH)
    default_idx = 0 if persona_ids else None

    selected_persona_id = st.sidebar.selectbox(
        "Select persona",
        persona_ids,
        index=default_idx,
    )

    if (
        "manual_state" not in st.session_state
        or st.session_state.get("manual_persona_id") != selected_persona_id
    ):
        reset_manual_state(selected_persona_id)

    if st.sidebar.button("Restart interview"):
        reset_manual_state(selected_persona_id)

    state = st.session_state.manual_state
    ground_facts = st.session_state.manual_ground_facts

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interview")
        for turn in state.transcript:
            speaker = "Interviewer" if turn["role"] == "assistant" else "User"
            st.write(f"**{speaker}:** {turn['content']}")

        if state.completed:
            st.success("Interview completed.")
        else:
            with st.form("reply_form", clear_on_submit=True):
                user_input = st.text_input("Your answer")
                submitted = st.form_submit_button("Send")

            if submitted and user_input.strip():
                dm.on_user_turn(
                    ground_facts=ground_facts,
                    state=state,
                    user_answer=user_input.strip(),
                )
                st.rerun()

    with col2:
        st.subheader("Structured Memory")
        st.json(
            {
                "facts": state.facts,
                "timeline": state.timeline,
                "open_threads": state.open_threads,
                "flags": state.flags,
                "covered_topics": state.covered_topics,
                "completed": state.completed,
                "last_intent": state.last_intent,
                "last_focus": state.last_focus,
            }
        )

else:
    st.subheader("Automatic simulation across all personas")
    st.write(
        "This mode loads every persona from `personas.json`, starts an interview for each one, "
        "and lets the persona agent speak automatically with the dialogue manager."
    )

    max_turns = st.number_input("Max persona turns per interview", min_value=1, max_value=50, value=12)

    if st.button("Run all personas"):
        personas = load_all_personas(PERSONAS_PATH)
        results = []

        progress = st.progress(0)
        status_box = st.empty()

        for idx, persona in enumerate(personas):
            persona_id = str(persona.get("persona_id"))
            status_box.info(f"Running interview for {persona_id} ...")

            ground_facts = build_ground_facts(persona)
            state = dm.start(persona_id)
            final_state = dm.run_auto_interview(
                persona=persona,
                ground_facts=ground_facts,
                state=state,
                max_turns=int(max_turns),
            )

            results.append(
                {
                    "persona_id": persona_id,
                    "name": persona.get("full_name") or persona.get("name"),
                    "completed": final_state.completed,
                    "turns_recorded": len(final_state.transcript),
                    "covered_topics": final_state.covered_topics,
                    "flags": final_state.flags,
                    "facts": final_state.facts,
                    "timeline": final_state.timeline,
                    "transcript": final_state.transcript,
                }
            )

            progress.progress((idx + 1) / len(personas))

        st.session_state.auto_results = results
        status_box.success("Finished running all personas.")

    if "auto_results" in st.session_state:
        st.subheader("Run Results")
        for result in st.session_state.auto_results:
            with st.expander(f"{result['persona_id']} - {result['name']}"):
                st.write(f"**Completed:** {result['completed']}")
                st.write(f"**Transcript length:** {result['turns_recorded']}")
                st.write(f"**Covered topics:** {result['covered_topics']}")
                st.write("**Flags:**")
                st.json(result["flags"])
                st.write("**Facts:**")
                st.json(result["facts"])
                st.write("**Timeline:**")
                st.json(result["timeline"])
                st.write("**Transcript:**")
                for turn in result["transcript"]:
                    speaker = "Interviewer" if turn["role"] == "assistant" else "Persona"
                    st.write(f"**{speaker}:** {turn['content']}")