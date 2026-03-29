import os
import asyncio
import random

import google.generativeai as genai
import psycopg2
from dataclasses import dataclass
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from settings import AIConfig
from config import (
    GEMINI_API_KEY,
    DB_PASSWORD,
    DB_USER,
    DB_PORT,
    GROQ_API_KEY,
    DB_HOST,
    DB_NAME,
    TABLE_NAME,
    GROQ_API_KEY,
)


load_dotenv()

genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[attr-defined]

# Existing environment variable usage is preserved.
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


@dataclass
class AgentDependencies:
    db_connection: psycopg2.extensions.connection
    table_name: str


async def rag_search_tool(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    alpha: float = 0.7,
) -> str:
    """Hybrid retrieval for the most relevant document chunks.

    - Computes an embedding for the query and uses pgvector for semantic
      similarity.
    - Uses pg_trgm `similarity(text, query)` for lexical matching.
    - Combines both into a single hybrid score:

        hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score

      where semantic_score = 1 - (embedding <=> query_embedding).
    """

    conn = ctx.deps.db_connection
    table_name = ctx.deps.table_name

    result = genai.embed_content(  # type: ignore[attr-defined]
        model="gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
        output_dimensionality=1536,
    )
    query_embedding = result["embedding"]

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT
                text,
                filename,
                chunk_index,
                chunk_type,
                1 - (embedding <=> %s::vector) AS semantic_score,
                similarity(text, %s) AS keyword_score,
                (
                    %s * (1 - (embedding <=> %s::vector))
                    + (1 - %s) * similarity(text, %s)
                ) AS hybrid_score
            FROM {table_name}
            ORDER BY hybrid_score DESC
            LIMIT %s
            """,
            (
                query_embedding,
                query,
                alpha,
                query_embedding,
                alpha,
                query,
                limit,
            ),
        )
        results = cur.fetchall()

    print(
        f"RAG Tool (hybrid) - Retrieved {len(results)} results "
        f"from PostgreSQL with alpha={alpha}, limit={limit}."
    )
    for r in results:
        print(
            f"Snippet: {r['text'][:100]!r}... "
            f"(semantic={r['semantic_score']:.4f}, "
            f"keyword={r['keyword_score']:.4f}, "
            f"hybrid={r['hybrid_score']:.4f})"
        )

    context = "\n---\n".join([f"Content: {res['text']}" for res in results])
    return f"Retrieved Context:\n{context}"


conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)


class RetrievalResult(BaseModel):
    """Output of the retrieval / planning agent.

    - enough_context: whether the retrieved chunks are sufficient.
    - draft_answer: initial answer based only on retrieved context.
    - context_chunks: text snippets actually used in the draft answer.
    """

    enough_context: bool
    draft_answer: str
    context_chunks: list[str]
    notes: str | None = None


# First agent: retrieve context and draft an answer.
rag_agent = Agent(
    AIConfig.model_name,
    deps_type=AgentDependencies,
    tools=[rag_search_tool],
    output_type=RetrievalResult,
    system_prompt=(
        """
You are SyllabiQ's retrieval and planning agent.

Your job is iterative hybrid retrieval and planning for a RAG system.

Capabilities:
- You can call `rag_search_tool(query: str, limit: int = 10, alpha: float = 0.7)`
  multiple times.
- Each call performs HYBRID retrieval over the knowledge base using both
  semantic (pgvector) and lexical (pg_trgm) similarity.

Workflow:

1. Start from the student's question.
2. Form an initial retrieval query (often the question itself) and call
   `rag_search_tool`.
3. Carefully read the returned chunks. Decide:
   - what parts of the question are still not covered;
   - what follow-up or more specific sub-queries are needed.
4. Refine the query (for example, add course/unit/topic keywords, important
   terms, or constraints) and call `rag_search_tool` again.
5. Repeat steps 3–4, up to about 3–5 tool calls in total, until:
   - you have enough coverage to answer the question reliably, or
   - it is clear the knowledge base does not contain the answer.

When you finish, fill the `RetrievalResult` fields as follows:
- `enough_context`: true only if the retrieved chunks clearly provide enough
  information to answer the question. Otherwise false.
- `draft_answer`: a careful, step-by-step answer that uses only the retrieved
  chunks. If context is not enough, clearly say that and explain what is
  missing or why the syllabus does not cover it.
- `context_chunks`: a list of the most relevant text snippets you actually
  used when forming the answer. Prefer concise but complete excerpts.
- `notes`: (optional) short planner notes about how many retrieval calls
  you made, which queries you used, and why you stopped.

Rules:
- Use ONLY the knowledge you retrieved with `rag_search_tool`.
- If the question is out of syllabus or the context is clearly insufficient,
  set `enough_context` to false and state this explicitly in `draft_answer`.
- Avoid repeating identical or obviously redundant chunks.
"""
    ),
)


# Second agent: polish and format the final answer.
answer_agent = Agent(
    AIConfig.model_name,
    system_prompt=(
        """
You are SyllabiQ, an academic assistant trained specifically on the
KTU B.Tech CSE curriculum (later expandable to other branches and semesters).

Your purpose is to help students understand concepts, answer syllabus
questions, explain topics, provide point-wise exam answers, and offer
evaluation-focused guidance.

You receive:
- the student's original question,
- a draft answer written by another agent,
- and the context snippets that were used.

Rewrite the draft answer so that it is:
- precise and technically correct,
- natural and conversational, like a helpful human tutor,
- well-structured for exam preparation (clear points, lists where useful).
- DO NOT include any AI or system-related commentary in the final answer. The student should only see the polished response.

If you are told that context was not enough, explicitly mention that
limitation and avoid inventing unsupported facts.
Do not mention that another agent wrote the draft; just answer as SyllabiQ.
"""
    ),
)


async def rag_pipeline_stream(query: str):
    """Two-agent pipeline used by the Streamlit UI.

    1. `rag_agent` retrieves context and drafts an answer.
    2. `answer_agent` rewrites the draft into a polished final response,
       which is streamed back to the caller.

    This is implemented as an async generator that yields a single
    streaming result object compatible with `result.stream_text(...)` and
    `result.new_messages()`, matching the expectations in `ai.py`.
    """

    # Retry retrieval a few times in case of transient failures.
    max_retries = 3
    retrieval = None
    for attempt in range(max_retries):
        try:
            retrieval = await rag_agent.run(
                query,
                deps=AgentDependencies(db_connection=conn, table_name=TABLE_NAME),
            )
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            if attempt == max_retries - 1:
                # Out of retries; re-raise so the UI can show an error.
                raise
            # Simple exponential backoff.
            wait = 1.0 * (2**attempt)
            print(
                f"rag_agent.run failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {wait:.1f}s..."
            )
            await asyncio.sleep(wait)

    if retrieval is None:  # Defensive, should be unreachable.
        raise RuntimeError("Retrieval agent did not return a result")

    retrieval_output = retrieval.output

    context_text = "\n\n".join(retrieval_output.context_chunks)
    sufficiency = "enough" if retrieval_output.enough_context else "not enough"

    prompt = f"""Question:
{query}

Draft answer (from retrieval agent):
{retrieval_output.draft_answer}

Context sufficiency: {sufficiency}

Context used:
{context_text}

Rewrite the draft answer into the best possible response for the student.
"""

    # Retry opening the streaming answer as well; if it fails after all
    # retries, propagate the error so the caller can handle it.
    for attempt in range(max_retries):
        try:
            async with answer_agent.run_stream(prompt) as result:
                # Yield the streaming result so the Streamlit layer can consume
                # `result.stream_text(delta=True)` and handle `new_messages()`.
                yield result
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            if attempt == max_retries - 1:
                raise
            wait = 1.0 * (2**attempt)
            print(
                f"answer_agent.run_stream failed (attempt {attempt + 1}/{
                    max_retries
                }): {exc}. "
                f"Retrying in {wait:.1f}s..."
            )
            await asyncio.sleep(wait)
