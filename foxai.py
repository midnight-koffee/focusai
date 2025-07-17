import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Focus Mode App", layout="centered")

st.title("🧠 Brain Focus Assistant")

if "focus_started" not in st.session_state:
    st.session_state.focus_started = False
    st.session_state.start_time = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Focus Mode"):
        st.session_state.focus_started = True
        st.session_state.start_time = datetime.now()
        st.success("✅ Focus mode started!")

with col2:
    if st.button("End Focus Mode"):
        if st.session_state.focus_started:
            focus_duration = datetime.now() - st.session_state.start_time
            st.info(f"🕒 You focused for **{focus_duration.seconds // 60} minutes** and **{focus_duration.seconds % 60} seconds**.")
            st.session_state.focus_started = False
        else:
            st.warning("⚠️ You haven’t started focus mode yet.")

st.markdown("---")
st.write("💡 Use this as your personal focus tracker. Next up, we’ll add typing stats, AI feedback, and webcam fatigue detection.")
