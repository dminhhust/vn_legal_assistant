"""Onboarding — the "personal info collection" MVP feature.

A short multi-step form rather than one long page, but every field on
it is one that app/profile/traits.py actually turns into a trait tag
that app/rag/query_builder.py reads to decide which legal categories
apply to this user (see backend/app/profile/schemas.py's module
docstring for why unused fields were deliberately dropped in this
redesign).
"""
import os

import requests
import streamlit as st

from utils import friendly_error_message

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Onboarding", page_icon="📝")
st.title("📝 Onboarding")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id:
    st.success(f"You're already onboarded as user `{st.session_state.user_id}`.")
    st.caption("Editing an existing profile isn't wired into this page yet — use the API directly if needed.")
    st.page_link("pages/2_Dashboard.py", label="✅ Go generate your checklist", icon="➡️")
    st.stop()

st.caption("A few quick questions — used only to figure out which legal obligations apply to you.")

with st.form("onboarding_form"):
    username = st.text_input("Username", placeholder="e.g. minh_nguyen")

    st.subheader("Identity")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        marital_status = st.selectbox("Marital status", ["", "single", "married", "divorced", "widowed"])
    with col2:
        gender = st.selectbox("Gender", ["", "male", "female", "other", "prefer_not_to_say"])
        dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=0, step=1)
    province = st.text_input("Province / city of residence", placeholder="e.g. Hanoi")

    st.subheader("Work")
    occupation_type = st.selectbox(
        "Occupation type",
        ["", "employee", "freelancer", "business_owner", "student", "retired", "unemployed"],
    )
    income_sources = st.multiselect(
        "Income sources", ["salary", "freelance", "business", "investment", "rental", "other"]
    )
    has_business = st.checkbox("I own a business")
    business_sector = st.text_input("Business sector (if applicable)", placeholder="e.g. retail")

    st.subheader("Assets")
    col3, col4 = st.columns(2)
    with col3:
        owns_property = st.checkbox("I own property (land / housing)")
    with col4:
        owns_vehicle = st.checkbox("I own a vehicle")

    st.subheader("Preferences")
    reminder_lead_days = st.slider("Remind me this many days before a deadline", 1, 30, 3)

    submitted = st.form_submit_button("Create profile", type="primary")

if submitted:
    if not username.strip():
        st.error("Username is required.")
        st.stop()

    payload = {
        "username": username.strip(),
        "age": int(age) if age else None,
        "gender": gender or None,
        "marital_status": marital_status or None,
        "province": province.strip() or None,
        "dependents": int(dependents),
        "occupation_type": occupation_type or None,
        "income_sources": income_sources,
        "has_business": has_business,
        "business_sector": business_sector.strip() or None,
        "owns_property": owns_property,
        "owns_vehicle": owns_vehicle,
        "reminder_lead_days": int(reminder_lead_days),
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/profile", json=payload, timeout=10)
        if resp.status_code == 409:
            st.error(f"Username '{username}' is already taken — pick another.")
        else:
            resp.raise_for_status()
            data = resp.json()
            st.session_state.user_id = data["user_id"]
            st.success("Profile created!")
            if data["traits"]:
                st.caption("Derived traits: " + ", ".join(data["traits"]))
            st.page_link(
                "pages/2_Dashboard.py", label="✅ Go generate your obligation checklist", icon="➡️"
            )
    except requests.RequestException as exc:
        st.error(f"Request failed: {friendly_error_message(exc)}")
