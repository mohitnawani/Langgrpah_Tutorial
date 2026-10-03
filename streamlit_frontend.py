import streamlit as st
from Langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# **********************utility functions**********************#


def generate_thread_id():
    thread_id = str(uuid.uuid4())

    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


# add thread in session_state
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# load the conversaions in session_state
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ***********************session setup*************#

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# STAGE 1 — thread_id now lives in session_state instead of being hardcoded
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# st.session_state -> dict ->
# STAGE 1 — list that powers the sidebar
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])  # STAGE

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_turn",
}

# {'role': 'user', 'content': 'Hi'}
# {'role': 'assistant', 'content': 'Hi=ello'}


# ***************************** Sidebar *****************************#

st.sidebar.title("Langgraph Chatbot")

# STAGE 2 — new chat button
if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("My Conversations")

# STAGE 1 — render thread list + handle switching
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages


# main ui
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config=CONFIG
    )
    ai_message = response["messages"][-1].content

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    with st.chat_message("assistant"):
        st.text(ai_message)
