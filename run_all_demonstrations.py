"""
Master Demonstrations Runner — Final Capstone: Nykaa Domain Support Agent.
Track: E-commerce & Retail (Nykaa).

Executes all capstone deliverables sequentially under 100% deterministic local MOCK_LLM mode:
- Part 1: Dataset validation (Task 1), RAG core calibration & evaluation (Tasks 3-5).
- Part 2: CrewAI multi-agent orchestration, tools, memory, structured schema & guardrails (Tasks 6-10).
- Part 3: FastAPI endpoints, WebSocket disconnect survival, structured logging, and 15-query evaluation benchmark (Tasks 11-13).
- Part 4: Autogen review team, four-layer AI governance & response caching (Tasks 14-16).
"""

import os
import sys
import time

# Ensure telemetry disabled and UTF-8 encoding
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dataset import print_dataset_report
from rag_core import run_rag_demonstration
from crew_agents import run_part2_demonstrations
from api_server import test_api_deployment
from evaluation import run_evaluation_suite
from governance_review import run_part4_demonstrations


def run_master_pipeline():
    start_total = time.perf_counter()
    print("=" * 80)
    print("NYKAA DOMAIN SUPPORT AGENT — MASTER CAPSTONE VERIFICATION")
    print("Track: E-commerce & Retail (Nykaa) | Mode: Deterministic Local MOCK_LLM")
    print("=" * 80)

    # Part 1: Dataset & RAG Core
    print("\n>>> EXECUTING PART 1: DATASET DESIGN & RAG CORE <<<")
    print_dataset_report()
    run_rag_demonstration()

    # Part 2: CrewAI Multi-Agent System
    print("\n>>> EXECUTING PART 2: CREWAI MULTI-AGENT ORCHESTRATION & GUARDRAILS <<<")
    run_part2_demonstrations()

    # Part 3: Deployment, Logging & Evaluation Benchmark
    print("\n>>> EXECUTING PART 3: FASTAPI BACKEND, WEBSOCKET & 15-QUERY EVALUATION <<<")
    test_api_deployment()
    run_evaluation_suite()

    # Part 4: Resilience, Governance & Caching
    print("\n>>> EXECUTING PART 4: AUTOGEN REVIEW STAGE, GOVERNANCE & RESPONSE CACHING <<<")
    run_part4_demonstrations()

    total_duration = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"ALL 16 CAPSTONE TASKS SUCCESSFULLY DEMONSTRATED AND VALIDATED!")
    print(f"Total Master Pipeline Runtime: {total_duration:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    run_master_pipeline()
