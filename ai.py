from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import streamlit as st

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from dotenv import load_dotenv

from agent import AgentDependencies, rag_agent
import psycopg2
from pgvector.psycopg2 import register_vector
load_dotenv()

DB_NAME = "vectordb"
DB_USER = "postgres"
DB_PASSWORD = "password"  
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "engineering_notes"

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**SyllabiQ**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str):
    async with rag_agent.run_stream(
        user_input,
        deps=AgentDependencies(db_connection=conn, table_name=TABLE_NAME),
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("SyllabiQ 📚🤖")
    st.write("Ask any question about KTU B.Tech CSE curriculum!")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)
if __name__ == "__main__":
    asyncio.run(main())