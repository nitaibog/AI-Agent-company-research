import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio
from agent.graph import build_agent


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

async def initialize_agent():
    if 'graph' not in st.session_state:
        st.session_state.graph = build_agent()
    

async def render_ui():
# Streamlit page setup
    st.set_page_config(page_title="Company Research Chatbot", page_icon="📊", layout="centered")

    st.markdown("""
        <h1 style='text-align: center;'>📊 Company Research Chatbot</h1>
        <p style='text-align: center; color: gray;'>Chat with the AI to get company insights.</p>
        """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about a company...")
    thread_id = st.session_state.thread_id
    
    # User input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": thread_id}}

        # Add AI response to chat history
        result = await st.session_state.graph.ainvoke({"raw_user_inputs" : [HumanMessage(content=user_input)]},config=config,stream_mode='values')
        output_str = result["final_report"][-1].replace("$", "\\$")
        st.session_state.messages.append({"role": "assistant", "content": output_str})
        
        # Display AI response 
        with st.chat_message("assistant"):
            st.markdown(output_str)


async def main():
    await initialize_agent()
    await render_ui()



if __name__ == '__main__':
    asyncio.run(main())