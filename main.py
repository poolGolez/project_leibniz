import streamlit as st
from streamlit_chat import message
from rag import ask

print("Refresh...")
QUERIES_HISTORY = "queries_history"
ANSWERS_HISTORY = "answers_history"
CHAT_HISTORY = "chat_history"

st.header("LangChain Docs Assistant")
if QUERIES_HISTORY not in st.session_state:
    st.session_state[QUERIES_HISTORY] = []
if ANSWERS_HISTORY not in st.session_state:
    st.session_state[ANSWERS_HISTORY] = []
if CHAT_HISTORY not in st.session_state:
    st.session_state[CHAT_HISTORY] = []

query = st.text_input("How can I help you?")
if query:
    with st.spinner("Thinking..."):
        response = ask(query, history=st.session_state[CHAT_HISTORY])
        st.session_state[QUERIES_HISTORY].append(query)
        st.session_state[ANSWERS_HISTORY].append(response)
        st.session_state[CHAT_HISTORY].append(("human", query))
        st.session_state[CHAT_HISTORY].append(("ai", response))

if st.session_state[ANSWERS_HISTORY]:
    for query, response in zip(st.session_state[QUERIES_HISTORY], st.session_state[ANSWERS_HISTORY]):
        message(query, is_user=True)
        message(response)
