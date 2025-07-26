from dotenv import load_dotenv
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
import os
load_dotenv()
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

import google.generativeai as genai

import streamlit as st

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model=genai.GenerativeModel("gemini-1.5-pro")
chat=model.start_chat(history=[])

def get_gemini_response(question):
    response=chat.send_message(question,stream=True)
    return response

st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("Gemini Chatbot 🤖")
if "chat_history" not in st.session_state:
    st.session_state['chat_history']=[]

input=st.text_input("Input your query:",key="input")
submit=st.button("Ask the Question")
if submit and input:
    response=(get_gemini_response(input))
    st.session_state['chat_history'].append(("You:", input))
    st.subheader("The Response is:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Gemini ChatBot:", chunk.text))
st.subheader("Chat History")
for role,text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")





