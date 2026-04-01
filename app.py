# app.py - replace everything with this

import streamlit as st

st.set_page_config(
    page_title="File RAG Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 File RAG Agent")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 💬 Chat Widget
    See how the public chatbot looks
    and test it out
    """)
    if st.button(
        "Open Chat",
        use_container_width=True,
        type="primary"
    ):
        st.switch_page("pages/chat.py")

with col2:
    st.markdown("""
    ### ⚙️ Admin Panel
    Upload files, change settings
    and manage the chatbot
    """)
    if st.button(
        "Open Admin",
        use_container_width=True
    ):
        st.switch_page("pages/admin.py")