from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.orchestrator import bank_bot


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_SUFFIXES = {".txt", ".csv", ".pdf"}
STREAM_DELAY_SECONDS = 0.04


app = FastAPI(
    title="NUST Bank Agentic API",
    description="REST API gateway for the LangGraph-powered banking assistant.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_query: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    final_response: str
    is_safe: bool
    scrubbed_query: str
    context_used: bool


class UploadAcceptedResponse(BaseModel):
    task_id: str
    status: str
    filename: str


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: dict | None = None
    error: str | None = None


def _validate_query(query: str) -> str:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )
    return cleaned_query


def _validate_upload_file(filename: str, content: bytes) -> str:
    safe_filename = Path(filename or "upload.bin").name
    suffix = Path(safe_filename).suffix.lower()

    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed types: {sorted(ALLOWED_UPLOAD_SUFFIXES)}"
            ),
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty.",
        )

    return safe_filename


def _build_upload_metadata(filename: str) -> dict[str, str | int]:
    return {
        "source_file": filename,
        "topic": filename,
        "question": "Uploaded policy",
        "sheet": "User Upload",
        "source_row_index": -1,
        "source_type": "upload",
    }


def _stage_uploaded_file(filename: str, content: bytes) -> Path:
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    staged_path = settings.uploads_dir / f"{uuid4().hex}_{filename}"
    staged_path.write_bytes(content)
    return staged_path


def _cleanup_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to remove staged file %s: %s", path, exc)


def queue_document_ingestion(file_path: Path, metadata: dict[str, str | int]) -> str:
    from backend.tasks.document_ingestion import ingest_document_task

    task = ingest_document_task.delay(str(file_path), metadata)
    return task.id


def get_task_status_payload(task_id: str) -> TaskStatusResponse:
    from celery.result import AsyncResult

    from backend.celery_app import celery_app

    task = AsyncResult(task_id, app=celery_app)

    result_payload = task.info if isinstance(task.info, dict) else None
    error_message = None

    if task.state == "SUCCESS":
        result_payload = task.result if isinstance(task.result, dict) else result_payload
    elif task.state == "FAILURE":
        error_message = str(task.result or task.info or "Unknown task failure.")

    return TaskStatusResponse(
        task_id=task_id,
        state=task.state,
        result=result_payload,
        error=error_message,
    )


def _invoke_bank_bot(query: str) -> ChatResponse:
    result = bank_bot.invoke({"user_query": query})

    return ChatResponse(
        final_response=result.get("final_response", "Error generating response."),
        is_safe=result.get("is_safe", False),
        scrubbed_query=result.get("scrubbed_query", ""),
        context_used=bool(
            result.get("selected_context") or result.get("retrieved_context")
        ),
    )


@app.get("/health")
def health_check():
    return {
        "status": "operational",
        "system": "NUST Bank LangGraph Orchestrator",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        query = _validate_query(request.user_query)
        return _invoke_bank_bot(query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat endpoint failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        ) from exc


@app.post(
    "/api/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadAcceptedResponse,
)
async def upload_document(file: UploadFile = File(...)):
    staged_path: Path | None = None

    try:
        content = await file.read()
        filename = _validate_upload_file(file.filename or "upload.bin", content)

        staged_path = _stage_uploaded_file(filename, content)
        metadata = _build_upload_metadata(filename)
        task_id = queue_document_ingestion(staged_path, metadata)

        logger.info("Queued upload '%s' as task %s", filename, task_id)

        return UploadAcceptedResponse(
            task_id=task_id,
            status="queued",
            filename=filename,
        )

    except HTTPException:
        _cleanup_file(staged_path)
        raise
    except Exception as exc:
        _cleanup_file(staged_path)
        logger.exception("Upload queueing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue file for ingestion.",
        ) from exc


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str):
    try:
        return get_task_status_payload(task_id)
    except Exception as exc:
        logger.exception("Task status lookup failed for %s: %s", task_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch task status.",
        ) from exc


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    query = _validate_query(request.user_query)

    async def event_generator():
        try:
            result = await asyncio.to_thread(bank_bot.invoke, {"user_query": query})
            full_text = result.get("final_response", "")

            if not full_text:
                yield "No response generated."
                return

            for word in full_text.split():
                yield f"{word} "
                await asyncio.sleep(STREAM_DELAY_SECONDS)

        except Exception as exc:
            logger.exception("Streaming chat failed: %s", exc)
            yield "Error: Failed to generate streaming response."

    return StreamingResponse(event_generator(), media_type="text/plain")
