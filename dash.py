import streamlit as st
import sys
from pathlib import Path
from pydantic_ai import Agent
from dotenv import load_dotenv
from google import genai
from settings import AIConfig
from chat import rag_agent
import lancedb
from chat import AgentDependencies, rag_search_tool

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
        }
        .stMarkdown h1 {
            color: #0066cc;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_table" not in st.session_state:
    try:
        db = lancedb.connect("./lancedb_data")
        st.session_state.db_table = db.open_table("engineering_notes")
    except:
        st.session_state.db_table = None


# Main chat interface
st.title("🤖 AI Chat Assistant")
st.markdown("---")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:        
                result = rag_agent.run_sync(user_input, deps=AgentDependencies(db_table=st.session_state.db_table))
                response = result.output
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
