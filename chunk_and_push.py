import os
import time

import google.generativeai as genai
import psycopg2
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

os.environ["GEMINI_API_KEY"] = "AIzaSyDgDgn96TXSaMuBK0mjMXEyiUL8mie5l98"
print(os.environ.get("GEMINI_API_KEY"))
print(os.environ.get("GOOGLE_API_KEY"))


MD_FILE_PATH = "./markdowns/book.md"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "engineering_notes"


genai.configure(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)


conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"""
CREATE EXTENSION IF NOT EXISTS vector;
""")

register_vector(conn)


cur.execute(f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    DROP TABLE IF EXISTS {TABLE_NAME};

    CREATE TABLE {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding vector(768),  -- Gemini text-embedding-004 produces 768-dimensional vectors
        filename TEXT,
        chunk_index INTEGER,
        chunk_type TEXT
    );

    CREATE INDEX ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
""")

print(f"Table '{TABLE_NAME}' created successfully.")


def get_embedding(text: str) -> list:
    """Generate embedding using Gemini API."""
    result = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="retrieval_document"
    )
    return result["embedding"]


def process_and_store_md(file_path: str):
    print(f"Processing: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    chunker = HybridChunker(max_tokens=800, overlap_tokens=100, merge_peers=True)

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
        entry = (chunk.text, embedding, os.path.basename("M3.md"), i, content_type)
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
