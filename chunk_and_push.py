import os
import time
import random
import sys
import re

import google.generativeai as genai
import psycopg2
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values, Json

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


def _parse_cli_metadata(argv: list[str]) -> dict:
    """Parse key=value pairs from CLI into a simple metadata dict.

    Expected keys (all optional): source, subject, semester, department.
    """

    meta: dict[str, object] = {}
    for arg in argv:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key:
            continue
        if key == "semester":
            try:
                meta[key] = int(value)
            except ValueError:
                continue
        else:
            meta[key] = value
    return meta


DOC_METADATA = _parse_cli_metadata(sys.argv[2:]) if len(sys.argv) > 2 else {}


MD_FILE_PATH = f"./markdowns/{file}"


genai.configure(api_key=GEMINI_API_KEY)


conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
conn.autocommit = True
cur = conn.cursor()

# Core extensions for vector and trigram search.
cur.execute(
    """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"""
)

register_vector(conn)


# Ensure table and columns exist (idempotent schema setup).
cur.execute(
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding vector(1536),
        filename TEXT,
        chunk_index INTEGER,
        chunk_type TEXT,
        keywords TEXT[],
        metadata JSONB,
        source TEXT,
        subject TEXT,
        semester INT,
        department TEXT,
        chapter TEXT,
        section TEXT
    );
    """
)

cur.execute(
    f"""
    ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS keywords TEXT[],
        ADD COLUMN IF NOT EXISTS metadata JSONB,
        ADD COLUMN IF NOT EXISTS source TEXT,
        ADD COLUMN IF NOT EXISTS subject TEXT,
        ADD COLUMN IF NOT EXISTS semester INT,
        ADD COLUMN IF NOT EXISTS department TEXT,
        ADD COLUMN IF NOT EXISTS chapter TEXT,
        ADD COLUMN IF NOT EXISTS section TEXT;
    """
)

cur.execute(
    f"""
    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_ivfflat_idx
        ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_text_trgm_idx
        ON {TABLE_NAME} USING gin (text gin_trgm_ops);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_keywords_gin_idx
        ON {TABLE_NAME} USING gin (keywords);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_subject_idx
        ON {TABLE_NAME}(subject);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_department_idx
        ON {TABLE_NAME}(department);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_semester_idx
        ON {TABLE_NAME}(semester);

    CREATE INDEX IF NOT EXISTS {TABLE_NAME}_metadata_gin_idx
        ON {TABLE_NAME} USING gin (metadata);
    """
)

print(f"Table '{TABLE_NAME}' is ready.")


def retry(func, retries: int = 5, base_delay: float = 1.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise

            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1} after error: {e}")
            time.sleep(delay)


def get_embedding(text: str):
    def call():
        return genai.embed_content(
            model="gemini-embedding-001",
            content=text,
            output_dimensionality=1536,
        )

    result = retry(call)
    return result["embedding"]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "over",
    "under",
    "above",
    "below",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "shall",
    "may",
    "might",
    "not",
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "by",
    "to",
    "as",
    "an",
    "a",
}


def extract_keywords(text: str, max_keywords: int = 12) -> list[str]:
    """Very simple keyword extractor for a chunk of text.

    Uses token frequency over alphabetic tokens, removes common stopwords,
    and returns the top `max_keywords` terms.
    """

    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    counts: dict[str, int] = {}
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        counts[tok] = counts.get(tok, 0) + 1

    sorted_terms = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [term for term, _ in sorted_terms[:max_keywords]]


def process_and_store_md(file_path: str, stop: bool = False):
    print(f"Processing: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    # Docling's HybridChunker already uses structural signals; keep tokens
    # moderate and add ~15% overlap for better continuity.
    chunker = HybridChunker(max_tokens=512, overlap_tokens=80, merge_peers=True)
    chunk_iter = list(chunker.chunk(doc))
    print(f"Total chunks for {file_path}: {len(chunk_iter)}")
    if stop:
        return

    data_to_ingest = []
    for i, chunk in enumerate(chunk_iter):
        headers = [h for h in chunk.meta.headings]
        hierarchy_path = " > ".join(headers) if headers else "Root"

        chapter = headers[0] if headers else None
        section = " > ".join(headers[1:]) if len(headers) > 1 else None

        content_type = "text"
        if "```" in chunk.text:
            content_type = "code"
        elif "|" in chunk.text and "-|" in chunk.text:
            content_type = "table"

        keywords = extract_keywords(chunk.text)
        embedding = get_embedding(chunk.text)
        metadata = {
            "hierarchy_path": hierarchy_path,
            "content_type": content_type,
            "source": DOC_METADATA.get("source"),
            "subject": DOC_METADATA.get("subject"),
            "semester": DOC_METADATA.get("semester"),
            "department": DOC_METADATA.get("department"),
            "chapter": chapter,
            "section": section,
        }

        time.sleep(0.5)  # To avoid hitting rate limits
        entry = (
            chunk.text,
            embedding,
            os.path.basename(f"{file_path}.md"),
            i,
            content_type,
            keywords,
            Json(metadata),
            DOC_METADATA.get("source"),
            DOC_METADATA.get("subject"),
            DOC_METADATA.get("semester"),
            DOC_METADATA.get("department"),
            chapter,
            section,
        )
        print(
            f"Prepared chunk {i} with {len(chunk.text)} chars, {
                len(embedding)
            }-dim embedding, and keywords: {keywords}"
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
                } (text, embedding, filename, chunk_index, chunk_type, keywords, metadata, source, subject, semester, department, chapter, section) VALUES %s",
                batch,
            )
            time.sleep(0.5)
        print(f"Successfully added {len(data_to_ingest)} chunks to PostgreSQL.")
    else:
        print("No chunks generated.")


if __name__ == "__main__":
    process_and_store_md(MD_FILE_PATH, stop=True)
