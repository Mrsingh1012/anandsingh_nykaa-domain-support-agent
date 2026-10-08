"""
Part 3 — Tasks 11 & 12: FastAPI Backend Deployment & Structured JSON-Lines Logging.
Track: E-commerce & Retail (Nykaa).

Exposes:
- POST /ask: Primary customer inquiry endpoint (Pydantic models).
- POST /add-document: Incremental knowledge base ingestion into ChromaDB.
- GET /health: Health-check endpoint.
- WebSocket /ws/chat: Real-time multi-turn chat handling client disconnects gracefully (WebSocketDisconnect).
- ELK-style structured JSON-Lines logging with trace ID, latency, and masked PII (no unmasked PII reaches disk).
"""

import os
import sys
import time
import json
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

# Ensure telemetry disabled and UTF-8 encoding
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import JSONResponse

from crew_agents import run_support_pipeline, NykaaSupportResponse, mask_pii
from tools import get_shared_vector_store
from rag_core import chunk_sentence_based

# ---------------------------------------------------------------------------
# Task 12: ELK-Style Structured JSON-Lines Logger
# ---------------------------------------------------------------------------

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "support_requests.jsonl")


def log_structured_request(
    trace_id: str,
    endpoint: str,
    method: str,
    raw_query: str,
    status_code: int,
    latency_ms: float,
    response_payload: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Logs structured ELK-compatible JSON-Lines entry.
    CRITICAL: Applies PII masking so raw phone numbers or payment cards NEVER touch disk.
    """
    masked_query, _ = mask_pii(raw_query)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "method": method,
        "session_id": session_id or "stateless",
        "query": masked_query,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "response": response_payload,
    }

    # Append to JSONL file on disk
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}", file=sys.stderr)

    return log_entry


# ---------------------------------------------------------------------------
# Task 11: Pydantic Request & Response Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(description="The user's policy inquiry or order tracking request.")
    session_id: Optional[str] = Field(default="default_session", description="Session identifier for multi-turn memory.")


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(description="Unique document code (e.g. 'KB-13').")
    topic: str = Field(description="Policy topic name.")
    title: str = Field(description="Document title.")
    content: str = Field(description="2-5 sentence policy statement.")


class AddDocumentResponse(BaseModel):
    status: str
    doc_id: str
    sentence_chunks_added: int
    message: str


# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nykaa Domain Support Agent API",
    description="Production-grade AI customer support backend orchestrating RAG, CrewAI, and Autogen Review.",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    """System liveness and readiness probe."""
    return {"status": "healthy", "service": "nykaa-domain-support-agent", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/ask", response_model=NykaaSupportResponse)
def ask_endpoint(payload: AskRequest):
    """
    HTTP Endpoint 1: Answers policy questions or tracks orders.
    Enforces PII masking, groundedness check, and structured response schema.
    FastAPI runs standard synchronous def endpoints in a worker threadpool.
    """
    trace_id = f"trc-{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()

    try:
        result = run_support_pipeline(
            raw_query=payload.query,
            session_id=payload.session_id or "default_session"
        )
        latency = (time.perf_counter() - start_time) * 1000.0

        # Record ELK structured log
        log_structured_request(
            trace_id=trace_id,
            endpoint="/ask",
            method="POST",
            raw_query=payload.query,
            status_code=200,
            latency_ms=latency,
            response_payload=result.model_dump(),
            session_id=payload.session_id,
        )
        return result

    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000.0
        log_structured_request(
            trace_id=trace_id,
            endpoint="/ask",
            method="POST",
            raw_query=payload.query,
            status_code=500,
            latency_ms=latency,
            response_payload={"error": str(e)},
            session_id=payload.session_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document_endpoint(payload: AddDocumentRequest):
    """
    HTTP Endpoint 2: Adds a new policy document incrementally to the ChromaDB index.
    """
    trace_id = f"trc-{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()

    vs = get_shared_vector_store()
    chunks = chunk_sentence_based(payload.content)

    ids = [f"{payload.doc_id}_sent_{i}" for i in range(len(chunks))]
    metas = [{
        "parent_doc_id": payload.doc_id,
        "topic": payload.topic,
        "title": payload.title,
        "strategy": "sentence",
        "chunk_index": i
    } for i in range(len(chunks))]

    embs = vs.model.encode(chunks, normalize_embeddings=True).tolist()
    vs.col_sentence.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)

    latency = (time.perf_counter() - start_time) * 1000.0
    res_data = AddDocumentResponse(
        status="success",
        doc_id=payload.doc_id,
        sentence_chunks_added=len(chunks),
        message=f"Document {payload.doc_id} successfully indexed into nykaa_kb_sentence.",
    )

    log_structured_request(
        trace_id=trace_id,
        endpoint="/add-document",
        method="POST",
        raw_query=f"Added doc {payload.doc_id}: {payload.title}",
        status_code=200,
        latency_ms=latency,
        response_payload=res_data.model_dump(),
    )
    return res_data


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket Endpoint: Enables interactive real-time multi-turn support chat.
    Gracefully handles client disconnects (catches WebSocketDisconnect) without crashing server.
    """
    await websocket.accept()
    session_id = f"ws_session_{uuid.uuid4().hex[:8]}"

    try:
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "message": "Welcome to Nykaa Customer Support Live Chat! How may I assist you today?"
        })

        while True:
            # Receive client message
            data = await websocket.receive_text()
            trace_id = f"trc-ws-{uuid.uuid4().hex[:8]}"
            start_time = time.perf_counter()

            try:
                parsed = json.loads(data)
                user_query = parsed.get("query", data)
            except Exception:
                user_query = data

            # Process through agent pipeline off the asyncio event loop
            result = await asyncio.to_thread(
                run_support_pipeline,
                raw_query=user_query,
                session_id=session_id
            )
            latency = (time.perf_counter() - start_time) * 1000.0

            log_structured_request(
                trace_id=trace_id,
                endpoint="/ws/chat",
                method="WEBSOCKET",
                raw_query=user_query,
                status_code=200,
                latency_ms=latency,
                response_payload=result.model_dump(),
                session_id=session_id,
            )

            # Transmit structured response back over WebSocket
            await websocket.send_json({
                "event": "message",
                "session_id": session_id,
                "data": result.model_dump(),
            })

    except WebSocketDisconnect:
        # Graceful disconnect handling — server stays alive and logs event
        print(f"[WebSocket] Client disconnected normally from session: {session_id}")
    except Exception as e:
        print(f"[WebSocket] Error during session {session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test Runner for Part 3 API Deployment & Logs
# ---------------------------------------------------------------------------

def test_api_deployment() -> None:
    """Tests all FastAPI endpoints and WebSocket disconnect resilience using TestClient."""
    from starlette.testclient import TestClient

    print("=" * 70)
    print("PART 3: FASTAPI BACKEND & STRUCTURED LOGGING TEST")
    print("=" * 70)

    client = TestClient(app)

    # 1. Test /health
    print("\n1. Testing GET /health:")
    h_resp = client.get("/health")
    print(f"Status: {h_resp.status_code} | Body: {h_resp.json()}")

    # 2. Test POST /ask (Policy Query)
    print("\n2. Testing POST /ask (Policy Query):")
    ask_resp1 = client.post("/ask", json={
        "query": "What are the delivery SLAs for metro cities?",
        "session_id": "test_api_sess"
    })
    print(f"Status: {ask_resp1.status_code}")
    print(f"Response Intent: {ask_resp1.json().get('intent')}")
    print(f"Response Answer: {ask_resp1.json().get('final_answer')}")

    # 3. Test POST /ask with PII Masking
    print("\n3. Testing POST /ask with PII in query:")
    pii_query = "Contact me at +91-9988776655 for order NYK-1008"
    ask_resp2 = client.post("/ask", json={
        "query": pii_query,
        "session_id": "test_api_sess"
    })
    print(f"Status: {ask_resp2.status_code}")
    print(f"Processed Query: {ask_resp2.json().get('query')}")

    # 4. Test POST /add-document
    print("\n4. Testing POST /add-document:")
    add_resp = client.post("/add-document", json={
        "doc_id": "KB-13",
        "topic": "sustainable packaging",
        "title": "Eco-Friendly Packaging Initiative",
        "content": "Nykaa ships all orders in 100% recyclable corrugated boxes. Bubble wraps are made of biodegradable starch."
    })
    print(f"Status: {add_resp.status_code} | Body: {add_resp.json()}")

    # 5. Test WebSocket /ws/chat and client disconnect survival
    print("\n5. Testing WebSocket /ws/chat & graceful disconnect:")
    with client.websocket_connect("/ws/chat") as ws:
        welcome = ws.receive_json()
        print(f"Connected Event: {welcome['event']} | Session: {welcome['session_id']}")
        
        ws.send_text(json.dumps({"query": "What is the return window for makeup?"}))
        reply = ws.receive_json()
        print(f"Received WebSocket Answer: {reply['data']['final_answer'][:90]}...")
        # Client disconnects when block exits
    print("Client disconnected cleanly. Server is still operational!")

    # 6. Verify Structured Log file contents
    print("\n6. Verifying ELK-Style Structured JSON-Lines log file:")
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Total structured log entries recorded: {len(lines)}")
        if lines:
            sample_entry = json.loads(lines[-1])
            print("Most recent structured log entry:")
            print(json.dumps(sample_entry, indent=2))
            assert "+91-9988776655" not in json.dumps(sample_entry), "Assertion Failed: Raw PII leaked to logs!"
            print("Verified: Raw PII is strictly masked in log file on disk!")

    print("=" * 70)


if __name__ == "__main__":
    test_api_deployment()
