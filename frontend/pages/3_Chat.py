"""Chat — the RAG chatbot MVP feature. Sends the full visible
conversation history with every request since the backend is
deliberately stateless (see backend/app/chat/agent.py's module
docstring on why there's no server-side session store in this MVP).
"""
import os

import requests
import streamlit as st

from utils import friendly_error_message

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Chat", page_icon="💬")
st.title("💬 Ask the Legal Assistant")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if not st.session_state.user_id:
    st.warning("No profile yet.")
    st.page_link("pages/1_Onboarding.py", label="📝 Go to Onboarding first", icon="➡️")
    st.stop()

user_id = st.session_state.user_id

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

st.caption(
    "This assistant can search the ingested legal corpus and check/update your "
    "checklist. It's not a substitute for professional legal advice."
)

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

user_message = st.chat_input("Ask a question, e.g. 'What tax deadlines apply to me?'")

if user_message:
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "user_id": user_id,
                        "message": user_message,
                        "history": st.session_state.chat_history[:-1],  # everything before this turn
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.write(data["text"])
                if data["tool_calls_made"]:
                    st.caption("Tools used: " + ", ".join(data["tool_calls_made"]))
                st.session_state.chat_history.append({"role": "assistant", "content": data["text"]})
            except requests.RequestException as exc:
                st.error(f"Request failed: {friendly_error_message(exc)}")
