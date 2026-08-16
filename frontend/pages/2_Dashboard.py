"""Dashboard — displays the obligation checklist, and hosts the MANUAL
ACTIVATION button that is this MVP's "auto-generated checklist with a
manual activation mechanism for showcase" feature.

Nothing generates a checklist automatically (no scheduler, nothing
triggered by onboarding) — clicking "Generate My Checklist" is the only
way one gets produced, and it runs the real retrieval -> extraction ->
deadline pipeline live, synchronously, so the person watching the demo
sees the AI-generated result appear in response to their own click.
See backend/app/rag/router.py's module docstring for the same point on
the API side.
"""
import os

import requests
import streamlit as st

from utils import friendly_error_message

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Dashboard", page_icon="✅")
st.title("✅ Your Obligation Checklist")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if not st.session_state.user_id:
    st.warning("No profile yet.")
    st.page_link("pages/1_Onboarding.py", label="📝 Go to Onboarding first", icon="➡️")
    st.stop()

user_id = st.session_state.user_id


def _fetch_checklist():
    resp = requests.get(f"{BACKEND_URL}/checklist/{user_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


st.markdown(
    "This checklist is **not generated automatically**. Click the button below "
    "to run it live — the app will search the ingested legal corpus for "
    "obligations that apply to your profile, extract structured deadlines with "
    "an LLM, and compute due dates, right now."
)

if st.button("🚀 Generate My Checklist", type="primary"):
    with st.spinner("Running retrieval + extraction pipeline..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/checklist/{user_id}/generate", timeout=120)
            resp.raise_for_status()
            st.success(f"Generated {len(resp.json())} checklist item(s).")
        except requests.RequestException as exc:
            st.error(f"Generation failed: {friendly_error_message(exc)}")

st.divider()

try:
    items = _fetch_checklist()
except requests.RequestException as exc:
    st.error(f"Could not load checklist: {friendly_error_message(exc)}")
    st.stop()

if not items:
    st.info("No checklist items yet — click **Generate My Checklist** above.")
    st.stop()

status_icon = {"pending": "🔲", "done": "✅", "dismissed": "🚫"}

for item in items:
    with st.container(border=True):
        col_main, col_status = st.columns([4, 1])
        with col_main:
            st.markdown(f"**{status_icon.get(item['status'], '🔲')} {item['title']}**")
            st.caption(f"Category: {item['category']}" + (f" · Due: {item['due_date']}" if item['due_date'] else ""))
            st.write(item["description"])
            st.caption(f"⚠️ {item['penalty_summary']}")
            st.caption(f"📖 Source: {item['source_citation']}")
        with col_status:
            new_status = st.selectbox(
                "Status",
                ["pending", "done", "dismissed"],
                index=["pending", "done", "dismissed"].index(item["status"]),
                key=f"status_{item['id']}",
                label_visibility="collapsed",
            )
            if new_status != item["status"]:
                try:
                    requests.patch(
                        f"{BACKEND_URL}/checklist/{user_id}/items/{item['id']}",
                        json={"status": new_status},
                        timeout=10,
                    ).raise_for_status()
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Update failed: {friendly_error_message(exc)}")
