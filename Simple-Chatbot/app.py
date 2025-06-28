import streamlit as st
from chatbot import get_response

st.title("Simple Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.text_area("Chatbot:", response, height=100)
