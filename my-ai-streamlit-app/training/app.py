import streamlit as st
from ai import RealAI  # import your AI class

st.set_page_config(page_title="RealAI Chatbot", layout="centered")

st.title("🤖 RealAI Chatbot")

# Initialize your AI ONCE (takes time, do this at startup)
@st.cache_resource(show_spinner=True)
def load_ai():
    return RealAI()

ai = load_ai()

# Chat input box
user_input = st.text_input("You:", placeholder="Type your message here...")

# When user sends something:
if user_input:
    with st.spinner("Generating response..."):
        response = ai.generate(user_input)
    st.markdown(f"**AI:** {response}")

