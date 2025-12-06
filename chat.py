import lancedb
from lancedb.rerankers import RRFReranker
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from typing import List
from dotenv import load_dotenv
from settings import AIConfig 

load_dotenv()

# 4. Define Agent Dependencies (to pass the DB connection)
@dataclass
class AgentDependencies:
    """Dependencies for the RAG Agent."""
    db_table: lancedb.table.LanceTable

async def rag_search_tool(ctx: RunContext[AgentDependencies], query: str) -> str:
    """
    Retrieves the most relevant document chunks from the knowledge base 
    based on the user's query.
    
    Args:
        query: The user's question or statement to search for.
    
    Returns:
        A formatted string of the relevant document context.
    """
    
    # Access the pre-configured LanceDB table from the agent's context
    table = ctx.deps.db_table 
    
    # Use the table's search function
    # results = table.search(query).limit(3).to_list()
    reranker = RRFReranker()
    results = (
        table.search(
            query=query,
            query_type="hybrid",
            vector_column_name="vector",
            fts_columns="text",
        )
        .rerank(reranker)
        .limit(10)
        .to_list()
    )

    print(f"RAG Tool - Retrieved {len(results)} results from LanceDB.")
    for r in results:
        print(f"Text Snippet: {r['text'][:100]!r}…")
    
    context = "\n---\n".join([
        f"Content: {res['text']}" for res in results
    ])
    
    if not context:
        return "No relevant context found in the knowledge base."
        
    return f"Retrieved Context:\n{context}"


db = lancedb.connect("./lancedb_data")
table = db.open_table("engineering_notes")
rag_agent = Agent(
    AIConfig.model_name, # Use the fast, free-tier-friendly model
    deps_type=AgentDependencies,
    system_prompt=(
        "You are an expert RAG chatbot. Answer the user's question based ONLY "
        "on the context provided by the 'rag_search_tool'. If the context "
        "is insufficient or irrelevant, state that you cannot answer."
    ),
)

rag_agent.tool(rag_search_tool)

async def run_chatbot():
    table = db.open_table("engineering_notes")
    print("--- RAG Chatbot Initialized ---")
    
    while True:
        user_input = input("\nAsk a question (or 'quit'): ")
        if user_input.lower() == 'quit' or user_input.lower() == 'q':
            break
            
        print("\nAgent Thinking...")
        
        # The agent decides whether to call the tool or not
        response = await rag_agent.run(user_input, deps=AgentDependencies(db_table=table))
        
        print("\n🤖 AI Response:")
        print(response.output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_chatbot())