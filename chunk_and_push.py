import os
import time
import random
import sys

import google.generativeai as genai
import psycopg2
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from config import (
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    TABLE_NAME,
    GEMINI_API_KEY,
)

file = sys.argv[1] if len(sys.argv) > 1 else None


MD_FILE_PATH = f"./markdowns/{file}"


genai.configure(api_key=GEMINI_API_KEY)


conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
""")

register_vector(conn)


cur.execute(f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding vector(1536),  -- Gemini text-embedding-004 produces 768-dimensional vectors
        filename TEXT,
        chunk_index INTEGER,
        chunk_type TEXT
    );

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_ivfflat_idx
        ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_text_trgm_idx
        ON {TABLE_NAME} USING gin (text gin_trgm_ops);
""")

print(f"Table '{TABLE_NAME}' is ready.")


def retry(func, retries=5, base_delay=1):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise

            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1} after error: {e}")
            time.sleep(delay)


def get_embedding(text):
    def call():
        return genai.embed_content(
            model="gemini-embedding-001", content=text, output_dimensionality=1536
        )

    result = retry(call)
    return result["embedding"]


#
# def get_embedding(text: str) -> list:
#     """Generate embedding using Gemini API."""
#     result = genai.embed_content(
#         model="gemini-embedding-001",
#         content=text,
#         task_type="retrieval_document",
#         output_dimensionality=1536,
#     )
#     return result["embedding"]


def process_and_store_md(file_path: str):
    print(f"Processing: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    chunker = HybridChunker(max_tokens=512, overlap_tokens=100, merge_peers=True)

    chunk_iter = chunker.chunk(doc)
    data_to_ingest = []
    for i, chunk in enumerate(chunk_iter):
        headers = [h for h in chunk.meta.headings]
        hierarchy_path = " > ".join(headers) if headers else "Root"
        content_type = "text"
        if "```" in chunk.text:
            content_type = "code"
        elif "|" in chunk.text and "-|-" in chunk.text:
            content_type = "table"

        embedding = get_embedding(chunk.text)

        time.sleep(0.5)  # To avoid hitting rate limits
        entry = (
            chunk.text,
            embedding,
            os.path.basename(f"{file_path}.md"),
            i,
            content_type,
        )
        data_to_ingest.append(entry)

    batch_size = 48
    if data_to_ingest:
        for i in range(0, len(data_to_ingest), batch_size):
            batch = data_to_ingest[i : i + batch_size]
            execute_values(
                cur,
                f"INSERT INTO {
                    TABLE_NAME
                } (text, embedding, filename, chunk_index, chunk_type) VALUES %s",
                batch,
            )
            time.sleep(0.5)
        print(f"Successfully added {len(data_to_ingest)} chunks to PostgreSQL.")
    else:
        print("No chunks generated.")


if __name__ == "__main__":
    process_and_store_md(MD_FILE_PATH)


# Search using cosine similarity
# query = "Explain OSI networking"
# query_embedding = get_embedding(query)
#
# cur.execute(
#     f"""
#     SELECT text, filename, chunk_index, chunk_type,
#            1 - (embedding <=> %s::vector) as similarity
#     FROM {TABLE_NAME}
#     ORDER BY embedding <=> %s::vector
#     LIMIT 5
# """,
#     (query_embedding, query_embedding),
# )
#
# results = cur.fetchall()
# for i in results:
#     print(i)
#
# # PostgreSQL doesn't have built-in hybrid search like LanceDB
# # You can implement full-text search separately and combine results
# query = "Explain Logical Link Control"
# query_embedding = get_embedding(query)
#
# cur.execute(
#     f"""
#     SELECT text, filename, chunk_index, chunk_type,
#            1 - (embedding <=> %s::vector) as similarity
#     FROM {TABLE_NAME}
#     ORDER BY embedding <=> %s::vector
#     LIMIT 10
# """,
#     (query_embedding, query_embedding),
# )
#
# results = cur.fetchall()
# print("Vector search results:")
# for i, (text, filename, chunk_index, chunk_type, similarity) in enumerate(results, 1):
#     print(f"\n--- Result {i} (similarity: {similarity:.4f}) ---")
#     print(f"File: {filename}, Chunk: {chunk_index}, Type: {chunk_type}")
#     print(f"Text: {text[:200]}...")
