
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Feedback", page_icon="📝")

st.title("Provide Feedback")

feedback_type = st.selectbox(
    "Feedback Type",
    ["Bug Report", "Feature Request", "Analysis Feedback", "General Comment"]
)

description = st.text_area("Description")
impact = st.slider("Impact Rating", 1, 5, 3)

if st.button("Submit Feedback"):
    feedback = {
        "type": feedback_type,
        "description": description,
        "impact": impact,
        "timestamp": datetime.now().isoformat(),
        "status": "new"
    }
    # Store feedback in session state for now
    if "feedback_list" not in st.session_state:
        st.session_state.feedback_list = []
    st.session_state.feedback_list.append(feedback)
    st.success("Thank you for your feedback!")
