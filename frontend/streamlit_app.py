"""Home page: backend connectivity check + navigation into the three
MVP flows (onboarding, dashboard/checklist, chat).
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="VN Legal Assistant — MVP", page_icon="🇻🇳")
st.title("🇻🇳 Personal Legal Assistant — MVP")
st.caption(
    "A redesigned, focused MVP: personal info collection, an auto-generated "
    "obligation checklist you activate manually, a RAG legal chatbot, and a "
    "real legal-source crawler behind the scenes."
)

if "user_id" not in st.session_state:
    st.session_state.user_id = None

st.subheader("Backend status")
try:
    resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    st.success(f"Backend reachable: {data['status']}")
    if data["llm_providers_available"]:
        st.caption("LLM providers available: " + ", ".join(data["llm_providers_available"]))
    else:
        st.warning(
            "No LLM provider keys detected — the checklist generator and chat "
            "will not be able to make real model calls until at least one of "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY is set in backend/.env."
        )
except requests.RequestException as exc:
    st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")

st.divider()

if st.session_state.user_id:
    st.info(f"Signed in as user `{st.session_state.user_id}`.")
else:
    st.info("No profile yet — start with Onboarding.")

st.page_link("pages/1_Onboarding.py", label="📝 Onboarding — personal info collection", icon="➡️")
st.page_link("pages/2_Dashboard.py", label="✅ Dashboard — generate & view your obligation checklist", icon="➡️")
st.page_link("pages/3_Chat.py", label="💬 Chat — ask the RAG legal assistant", icon="➡️")
