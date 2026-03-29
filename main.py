import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import rag_pipeline_stream

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("syllabiq")

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_TOKEN", "6000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SyllabiQ FastAPI service starting up")
    yield
    logger.info("SyllabiQ FastAPI service shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SyllabiQ API",
    description="RAG-powered syllabus Q&A service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The student's question",
    )


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

async def stream_response(question: str):
    """Wrap rag_pipeline_stream as an SSE generator."""
    try:
        async for chunk in rag_pipeline_stream(
            question,
            max_context_tokens=MAX_CONTEXT_TOKENS,
        ):
            import json
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.exception("Streaming error")
        yield f"data: [ERROR] {exc}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "syllabiq"}


@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    logger.info("Chat request: %s", question[:80])

    try:
        return StreamingResponse(
            stream_response(question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming endpoint — returns the full answer as JSON."""
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    logger.info("Sync chat request: %s", question[:80])

    try:
        result = ""
        async for chunk in rag_pipeline_stream(
            question,
            max_context_tokens=MAX_CONTEXT_TOKENS,
        ):
            result += chunk
        return {"answer": result}
    except Exception as exc:
        logger.exception("Sync chat error")
        raise HTTPException(status_code=500, detail=str(exc))
