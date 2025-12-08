import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from typing import List
from dotenv import load_dotenv
from settings import AIConfig
import os
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
@dataclass
class AgentDependencies:
    db_connection: psycopg2.extensions.connection
    table_name: str

async def rag_search_tool(ctx: RunContext[AgentDependencies], query: str) -> str:
    """
    Retrieves the most relevant document chunks from the knowledge base 
    based on the user's query.
    
    Args:
        query: The user's question or statement to search for.
    
    Returns:
        A formatted string of the relevant document context.
    """
    conn = ctx.deps.db_connection
    table_name = ctx.deps.table_name
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = result['embedding']
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT text, filename, chunk_index, chunk_type,
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """, (query_embedding, query_embedding))
        results = cur.fetchall()

    print(f"RAG Tool - Retrieved {len(results)} results from PostgreSQL.")
    for r in results:
        print(f"Text Snippet: {r['text'][:100]!r}… (similarity: {r['similarity']:.4f})")
    context = "\n---\n".join([
        f"Content: {res['text']}" for res in results
    ])
    if not context:
        return "No relevant context found in the knowledge base."
    return f"Retrieved Context:\n{context}"

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

rag_agent = Agent(
    AIConfig.model_name, 
    deps_type=AgentDependencies,
    system_prompt=(
        """
You are SyllabiQ, an academic assistant trained specifically on the KTU B.Tech CSE curriculum (later expandable to other branches and semesters).
Your purpose is to help students understand concepts, answer syllabus questions, explain topics, provide point-wise exam answers, and offer evaluation-focused guidance.
Your goal is to help students learn, understand, revise, and write answers confidently.
Answer naturally, do not sound like a machine.
If context is not enough, decline politely.
If the question asked is out of syllabus, decline politely.
DO NOT USE EXTERNAL KNOWLEDGE!
"""
    ),
)

rag_agent.tool(rag_search_tool)

async def run_chatbot():
    print("--- RAG Chatbot Initialized ---")
    print(f"Model: {AIConfig.model_name}")
    print(f"Database: PostgreSQL with pgvector")
    print(f"Table: {TABLE_NAME}")
    
    while True:
        user_input = input("\nAsk a question (or 'quit'): ")
        if user_input.lower() == 'quit' or user_input.lower() == 'q':
            break
            
        print("\nAgent Thinking...")
        response = await rag_agent.run(
            user_input, 
            deps=AgentDependencies(db_connection=conn, table_name=TABLE_NAME)
        )
        
        print("\n🤖 AI Response:")
        print(response.output)
    conn.close()
    print("\nDatabase connection closed. Goodbye!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_chatbot())
