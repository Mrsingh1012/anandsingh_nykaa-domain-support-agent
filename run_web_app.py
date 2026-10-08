"""
Interactive Web Application Runner — Nykaa Domain Support Agent.
Track: E-commerce & Retail (Nykaa).

Launches the FastAPI backend with WebSocket support and opens the
interactive Web Chat UI directly in your default web browser.

Usage:
    python run_web_app.py
"""

import os
import sys
import threading
import time
import webbrowser
import uvicorn

# Ensure telemetry disabled and UTF-8 encoding
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"


def open_browser():
    """Waits for uvicorn server startup and opens the web application in browser."""
    time.sleep(1.2)
    print("\n[Browser] Launching interactive Web UI at http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")


def main():
    print("=" * 72)
    print("  NYKAA DOMAIN SUPPORT AGENT — INTERACTIVE MULTI-AGENT WEB APP")
    print("=" * 72)
    print("  • Architecture: Local SentenceTransformers RAG + CrewAI + Autogen")
    print("  • Mode: Deterministic Local MOCK_LLM (Zero API keys, zero network)")
    print("  • Web Server: http://127.0.0.1:8000")
    print("  • API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 72)

    # Spawn browser opener thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
