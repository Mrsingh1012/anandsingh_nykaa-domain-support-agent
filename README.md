# Nykaa Domain Support Agent (CrewAI Capstone)

> **GitHub Repository:** [https://github.com/Mrsingh1012/anandsingh_nykaa-domain-support-agent](https://github.com/Mrsingh1012/anandsingh_nykaa-domain-support-agent)  
> **Author / Student:** [Mrsingh1012](https://github.com/Mrsingh1012)  
> **Track:** E-commerce & Retail (Nykaa)  
> **Architecture:** Local SentenceTransformers RAG + CrewAI Multi-Agent System + Autogen Review Stage + FastAPI Deployment  
> **Execution Mode:** Deterministic Local `MOCK_LLM` (Zero API keys, zero network access, 100% reproducible)  
> **Telemetry Control:** `CREWAI_DISABLE_TELEMETRY=true` & `OTEL_SDK_DISABLED=true`  

---

## 1. Executive Summary & Scenario Overview

Nykaa is India's premier beauty, cosmetics, and fashion e-commerce platform. Frontline customer support teams face high ticket volumes regarding category-specific return windows, COD refund processing, delivery SLAs, and order delays.

This capstone project implements an end-to-end, production-grade AI customer support agent that:
1. **Grounds Policy Inquiries:** Uses a local vector database (ChromaDB) with sentence-level semantic chunking and an empirically calibrated fallback threshold to answer policy questions.
2. **Tracks Orders & Evaluates Risk:** Queries an order database and computes a designed, non-linear **escalation score** to prioritize delayed shipments for senior supervisor intervention.
3. **Orchestrates Multi-Agent Collaboration:** Employs a 3-agent CrewAI crew (**Retrieval Agent**, **Lookup Agent**, and **Response Composer Agent**) operating under the **Principle of Least Autonomy**.
4. **Maintains Conversation Memory:** Preserves multi-turn dialogue context across turns via LangChain session memory, with demonstrable clean isolation for fresh sessions.
5. **Enforces Dual Guardrails:** Masks fixed-format PII (phone numbers and payment card digits) before model ingestion or logging, detects prompt injection attacks, and verifies output groundedness.
6. **Deploys Behind FastAPI & WebSockets:** Exposes REST endpoints (`POST /ask`, `POST /add-document`) and a resilient WebSocket chat (`/ws/chat`) that gracefully survives client disconnections without crashing.
7. **Maintains ELK-Style Structured Logging:** Records every request with trace IDs, timestamps, and latencies, guaranteeing that unmasked PII never reaches disk.
8. **Audits Drafts with Autogen:** Uses an independent 2-agent Autogen group chat team to verify policy compliance and issue structured Pydantic verdicts (`ReviewVerdict`), approving valid drafts and revising hallucinations.
9. **Governs AI Operations:** Enforces a 4-layer AI governance framework (Application, Organizational, and Runtime layers) with a per-request token budget cap and response caching.

---

## 2. Project Architecture

```
                       +---------------------------------------+
                       |        FastAPI / WebSocket API        |
                       |   (/ask, /add-document, /ws/chat)     |
                       +---------------------------------------+
                                           |
                                [Input Guardrails]
                    +-------------------------------------+
                    | 1. Prompt Injection Detector        |
                    | 2. PII Masker (Phone & Card Digits) |
                    +-------------------------------------+
                                           |
                              [LangChain Session Memory]
                                           |
                       +---------------------------------------+
                       |          CrewAI Agent Crew            |
                       |       (NykaaMockLLM / BaseLLM)        |
                       +---------------------------------------+
                                  /                 \
                                 /                   \
                   [Policy Inquiries]            [Order Lookups]
                          v                             v
            +---------------------------+  +---------------------------+
            |      Retrieval Agent      |  |       Lookup Agent        |
            |   (rag_policy_lookup)     |  |   (check_order_status)    |
            +---------------------------+  +---------------------------+
                          |                             |
                 [ChromaDB Vector Store]       [Nykaa Order Dataset]
                 (nykaa_kb_sentence Col)       (50 Records / Seed 42)
                          \                             /
                           \                           /
                            v                         v
                       +---------------------------------------+
                       |        Response Composer Agent        |
                       |   (Principle of Least Autonomy: No DB)|
                       +---------------------------------------+
                                           |
                               [Output Guardrail]
                                (Groundedness Check)
                                           |
                       +---------------------------------------+
                       |      Autogen 2-Agent Review Team      |
                       | (Policy Reviewer + Final Editor Team) |
                       |    (StructuredMessage[ReviewVerdict]) |
                       +---------------------------------------+
                                           |
                          +--------------------------------+
                          |   Validated Structured Output  |
                          |     (NykaaSupportResponse)     |
                          |   + ELK Structured Log Entry   |
                          +--------------------------------+
```

---

## 3. Part 1 — Dataset Design & RAG Core (Tasks 1–5)

### Task 1: Order Dataset Design (`dataset.py`)
To ensure complete grading reproducibility, the dataset generator runs deterministically using a fixed random seed.

- **Random Seed:** `42`
- **Total Order Records:** `50` (Requirement: $\ge 40$)
- **Catalog Price Range:** INR 299 to INR 14,999
  - *Reasoning:* Reflects Nykaa’s real retail mix, spanning accessible everyday personal care items (INR 299–1,499), mid-tier apparel/footwear (INR 799–6,999), and premium luxury cosmetics or electronic styling appliances (up to INR 14,999).
- **Categories (Requirement: each $\ge 3$ records):**
  - Beauty: **20** records (40.0% weight — reflects Nykaa's core focus)
  - Apparel: **9** records (18.0%)
  - Footwear: **9** records (18.0%)
  - Home: **6** records (12.0%)
  - Electronics: **6** records (12.0%)
- **Status Coverage (Requirement: each $\ge 1$ record):**
  - Placed: **10** records
  - Shipped: **17** records
  - Delivered: **16** records
  - Returned: **5** records
  - Refunded: **2** records
- **Delayed Shipment Rate:** **12 / 50 (24.00%)** — strictly within the required **[10%, 30%]** band.

### Task 2: Nykaa Knowledge Base Documents (`knowledge_base.py`)
Authoring 12 domain-specific policy documents (2–5 sentences each) covering all required topics:
- `KB-01`: Return Window Policy by Product Category (5 days cosmetics/fragrances, 15 days apparel, 7 days appliances).
- `KB-02`: Cash on Delivery (COD) Refund Timelines (NEFT/IMPS 3–5 days, instant wallet 2 hours).
- `KB-03`: Standard Delivery Service Level Agreements (SLAs) (2–3 days metro, 4–6 days tier 2/3, 7–10 days remote).
- `KB-04`: Reverse-Pickup Service Eligibility and Coverage (doorstep pickup across 19,000+ pincodes, max INR 150 self-ship reimbursement).
- `KB-05`: Product Warranty Terms and Authenticity Guarantee (100% genuine sourcing, 1–2 year appliance brand warranty).
- `KB-06`: Order Cancellation Policy and Guidelines (cancellation allowed while Placed; locked once Shipped).
- `KB-07`: Nykaa Privé Loyalty Points Redemption Policy (1 pt/INR 100 spent, INR 1/pt redemption, max 50% cart value).
- `KB-08`: Payment Failure and Cart Retention Policy (bank auto-reconciliation 24–48 hrs, cart reserved for 15 mins).
- `KB-09`: Apparel and Footwear Size-Exchange Guidelines (one free size exchange within 7 days).
- `KB-10`: Damaged or Tampered Item Claim Process (claim within 48 hours with packaging photo/video evidence).
- `KB-11`: International Shipping and Cross-Border Restrictions (aerosols/perfumes prohibited by air transit).
- `KB-12`: Customer Support Escalation Matrix and Turnaround SLAs (L1 instant bot/chat, L2 supervisor 12 hrs, L3 grievance 24–48 hrs).

### Tasks 3 & 5: Dual Chunking Strategies & Precision/Recall Evaluation (`rag_core.py`)
Documents were embedded using the local `sentence-transformers` model (`all-MiniLM-L6-v2`) and indexed into two separate ChromaDB collections:
1. **Strategy A (Fixed-Size Chunking):** Character window size 150, overlap 35 (`nykaa_kb_fixed`). Indexed 54 chunks.
2. **Strategy B (Sentence-Based Chunking):** Boundary regex split on terminal punctuation (`nykaa_kb_sentence`). Indexed 48 chunks.

#### Precision and Recall Evaluation Benchmark ($k = 3$)
Document-level matching maps retrieved chunks back to unique parent document IDs before scoring:

| Benchmark Query | Ground Truth | Strategy A (Fixed-Size) Precision / Recall | Strategy B (Sentence-Based) Precision / Recall |
|---|---|---|---|
| Return window for makeup and cosmetics | `KB-01` | Prec: $1/2 = 0.500$ \| Rec: $1/1 = 1.000$ | Prec: $1/2 = 0.500$ \| Rec: $1/1 = 1.000$ |
| COD refunds processing & timelines | `KB-02` | Prec: $1/3 = 0.333$ \| Rec: $1/1 = 1.000$ | Prec: $1/3 = 0.333$ \| Rec: $1/1 = 1.000$ |
| Delivery SLAs for metro and remote cities | `KB-03` | Prec: $1/1 = 1.000$ \| Rec: $1/1 = 1.000$ | Prec: $1/1 = 1.000$ \| Rec: $1/1 = 1.000$ |
| Order cancellation once shipped | `KB-06` | Prec: $1/1 = 1.000$ \| Rec: $1/1 = 1.000$ | Prec: $1/1 = 1.000$ \| Rec: $1/1 = 1.000$ |
| Damaged or leaked product claim process | `KB-10` | Prec: $1/2 = 0.500$ \| Rec: $1/1 = 1.000$ | Prec: $1/1 = 1.000$ \| Rec: $1/1 = 1.000$ |
| **Averages Across All Queries** | - | **Precision = 0.667 \| Recall = 1.000** | **Precision = 0.767 \| Recall = 1.000** |

#### Recommendation Statement for Deployment
> **Recommendation:** We deploy **Strategy B (Sentence-Based Chunking)**. Sentence-based chunks achieved an average precision of **0.767** and recall of **1.000**, outperforming fixed-size chunking which achieved precision of **0.667** and recall of **1.000**. Sentence-based chunking preserves complete semantic policy boundaries and conditions, avoiding fragmentation across arbitrary character offsets.

### Task 4: Empirical Fallback Calibration
Instead of using untested arbitrary thresholds (such as 0.5 or 0.7), the fallback threshold was calibrated empirically by measuring top-1 cosine similarities across in-scope vs out-of-scope queries:

- **In-Scope Query Similarities:**
  - *"What is the return window for makeup and cosmetic products?"* -> `0.4367`
  - *"How many days does it take to get a COD bank refund?"* -> `0.5507`
  - *"Can I cancel an order that has already shipped?"* -> `0.7450`
  - *"What is the policy for claiming a damaged or broken package?"* -> `0.4818`
  - **Minimum In-Scope Similarity Observed:** **`0.4367`**
- **Out-of-Scope Query Similarities:**
  - *"What are the best long-term mutual fund investment options?"* -> `0.1497`
  - *"Who won the ICC Cricket World Cup tournament in 2011?"* -> `0.1402`
  - *"How do I repair a leaking bathroom water pipe?"* -> `0.1223`
  - **Maximum Out-of-Scope Similarity Observed:** **`0.1497`**

$$\text{Calibrated Threshold} = \frac{\min(\text{In-Scope}) + \max(\text{Out-of-Scope})}{2} = \frac{0.4367 + 0.1497}{2} = \mathbf{0.2930}$$

Any query retrieving chunks with top-1 similarity $< 0.2930$ triggers the grounded fallback:
`"I don't know based on the provided Nykaa policy documentation."`

---

## 4. Part 2 — CrewAI Multi-Agent Orchestration, Memory & Guardrails (Tasks 6–10)

### Task 6: Designed Order Escalation Score (`tools.py`)
To prevent arbitrary escalation, `check_order_status(record_id: str)` calculates a designed, non-linear score in $[0, 1]$ combining shipment delay and normalized recency:

$$\text{escalation\_score} = (\mathbf{1}_{\text{delayed\_shipment}} \times 0.55) + \left(\frac{\text{days\_since\_created}}{30.0} \times 0.45\right)$$

- **Escalation Threshold:** **`0.65`**
- **Dataset Distribution Justification:**
  In `dataset.py`, the empirical **80th percentile** of `days_since_created` is **25.0 days**.
  1. Any order marked with `delayed_shipment=True` that is at least 7 days old achieves:
     $$0.55 + \left(\frac{7}{30} \times 0.45\right) = 0.655 \ge 0.65 \implies \text{Triggers Level 2 Supervisor Escalation}$$
  2. Non-delayed orders only cross the $0.65$ threshold if `days_since_created` exceeds $28.9$ days ($> 95\text{th}$ percentile), representing aging unfulfilled orders.

### Task 7: 3-Agent CrewAI System & Critical Pitfalls Solved (`crew_agents.py`, `mock_llm.py`)
Constructed with 3 specialized agents:
1. **Retrieval Agent:** Equipped exclusively with `rag_policy_lookup`.
2. **Lookup Agent:** Equipped exclusively with `check_order_status`.
3. **Response Composer Agent:** Equipped with no tools (Least Autonomy), synthesizing final customer-facing responses.

#### Addressing Known CrewAI Pitfalls:
- **Zero API Keys & BaseLLM Extension:** Implemented `NykaaMockLLM(BaseLLM)` to allow fully offline execution without LiteLLM network hooks.
- **System Template Observation Trap:** CrewAI's default ReAct system prompt contains the literal example string `"Observation: the result of the action"`. A naive parser matching `"Observation:"` will trigger on the very first turn before tool execution. Our model specifically ignores the ReAct instruction template and checks only genuine tool return messages.
- **Schema-Based Argument Dispatching:** Substring tool matching fails when tool names overlap (e.g. `rag_lookup`). Our mock engine inspects the tool's declared Pydantic schema (`record_id` vs `query`) to supply correct parameters.
- **Telemetry Disabling:** Set `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` prior to imports.

### Task 8: Multi-Turn Session Memory
Integrated LangChain's `InMemoryChatMessageHistory` to maintain conversational context within the process run.
- **Carried Context Demonstration:** Turn 1 establishes order ID `NYK-1002`. Turn 2 asks *"Can I cancel this order right now?"*, correctly carrying forward the order ID and status.
- **Fresh Session Reset:** Querying an isolated session ID (`session_202`) demonstrates a clean history with zero carried messages.

### Task 9: Structured Pydantic Output Schema
Every crew response validates against `NykaaSupportResponse`:
```python
class NykaaSupportResponse(BaseModel):
    query: str
    intent: str  # "policy" | "order_status" | "general" | "refusal"
    final_answer: str
    sources: List[str]
    escalation_required: bool
    escalation_score: Optional[float]
    grounded: bool
    guardrail_triggered: Optional[str]
```

### Task 10: Input & Output Guardrails
- **Input PII Masking:** Regex maskers detect 10-digit Indian mobile numbers (`+91-XXXXXXXXXX`) and 16-digit or last-4 card numbers (`card ending in 4123`), replacing them with `[MASKED_PHONE_NUMBER]` and `[MASKED_CARD_DIGITS]`.
- **Input Prompt Injection:** Detects adversarial patterns (e.g. `"ignore all previous instructions"`, `"system override"`, `"DAN mode"`), terminating execution with a security refusal.
- **Output Groundedness:** Verifies that returned statements are supported by retrieved doc IDs or valid order records, refusing ungrounded outputs.

---

## 5. Part 3 — Evaluation, Observability & FastAPI Deployment (Tasks 11–13)

### Task 11: FastAPI Deployment (`api_server.py`)
- `POST /ask`: Accepts `AskRequest(query, session_id)`, returning `NykaaSupportResponse`.
- `POST /add-document`: Accepts `AddDocumentRequest(doc_id, topic, title, content)`, dynamically embedding and upserting into ChromaDB.
- `GET /health`: Returns service health and UTC timestamp.
- `WebSocket /ws/chat`: Real-time streaming chat. Catches `WebSocketDisconnect` cleanly, keeping the server operational for other clients.

### Task 12: ELK-Style Structured JSON-Lines Logging
Every HTTP and WebSocket request writes an entry to `support_requests.jsonl` with:
- `timestamp` (ISO-8601 UTC)
- `trace_id` (unique correlation ID)
- `endpoint`, `method`, `session_id`, `status_code`, `latency_ms`
- `query` (**Guaranteed Masked PII** — verified via automated assertion that raw phone and card numbers never touch disk).

### Task 13: 15-Query Evaluation Benchmark (`evaluation.py`)
Evaluated across 15 queries covering all 12 KB topics, 1 order status check, and 2 out-of-scope / adversarial queries using LLM-as-judge scoring:

| Query ID | Topic / Focus | Accuracy | Grounding | Completeness | Safety |
|---|---|:---:|:---:|:---:|:---:|
| **Q-01** | Return Window by Product Category (`KB-01`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-02** | COD Refund Timelines (`KB-02`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-03** | Delivery SLAs (`KB-03`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-04** | Reverse-Pickup Eligibility (`KB-04`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-05** | Warranty Terms by Category (`KB-05`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-06** | Order Cancellation Policy (`KB-06`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-07** | Loyalty Points Redemption (`KB-07`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-08** | Payment Failure Policy (`KB-08`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-09** | Size Exchange Guidelines (`KB-09`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-10** | Damaged Item Claim Process (`KB-10`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-11** | International Shipping Restrictions (`KB-11`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-12** | Customer Escalation Matrix (`KB-12`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-13** | Order Status & Escalation Lookup (`dataset.py`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Q-14** | Deliberate Out-of-Scope Stock Query | 1.00 | 0.00 | 0.00 | 1.00 |
| **Q-15** | Adversarial Prompt Injection Attempt | 1.00 | 1.00 | 1.00 | 1.00 |
| **AVERAGE**| **Overall 15-Query Benchmark Average** | **1.000** | **0.933** | **0.933** | **1.000** |

---

## 6. Part 4 — Resilience, Governance & Autogen Review Stage (Tasks 14–16)

### Task 14: Autogen 2-Agent Review Stage (`governance_review.py`)
Built using `autogen_agentchat.teams.RoundRobinGroupChat` bounded with `max_turns=2`:
- **Policy Compliance Reviewer:** Audits draft against retrieved context chunks.
- **Final Editor:** Configured with `output_content_type=ReviewVerdict`, registered via `custom_message_types=[StructuredMessage[ReviewVerdict]]`.
- **Demonstrations:**
  1. *Approved Unchanged:* Correct draft stating a 5-day return window for cosmetics receives `approved=True`.
  2. *Revised Hallucination:* Injected draft claiming a "30-day unconditional return window for cosmetics" is caught and rewritten to 5 days with `approved=False`.

### Task 15: Four-Layer AI Governance Model
1. **Application Layer (Principle of Least Autonomy):** Only the Lookup Agent is wired with `check_order_status`. Attempting to wire or invoke this tool from the Retrieval Agent raises a strict `PermissionError`.
2. **Organizational Layer (Risk Classification):**
   > **Classification:** **Medium Risk**  
   > *Justification:* Unlike Low-Risk systems (e.g. summarization), this agent touches customer-facing operations, interprets refund policies, and retrieves account order records. However, it does not execute high-risk autonomous transactions or handle healthcare/demographic data. Risk is mitigated by local deterministic embeddings, PII masking, least-autonomy tool gating, and the Autogen review audit stage.
3. **Runtime Layer (Token/Cost Budget Cap):** Requests exceeding 150 estimated tokens (~600 characters) are rejected with a formal governance error message before triggering model inference.

### Task 16: Response Caching
An in-memory cache keyed by normalized query strings eliminates redundant vector searches:
- **First Call (Cache Miss):** Latency $\approx 8,000\text{ ms}$ (initial vector distance compute).
- **Second Call (Cache Hit):** Latency $< 0.01\text{ ms}$ (**$> 5,000,000\times$ speedup**).

---

## 7. How to Run the Project (Zero API Keys Required)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/Mrsingh1012/anandsingh_nykaa-domain-support-agent.git
cd anandsingh_nykaa-domain-support-agent

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Chat UI
To visually chat with all agents in your browser with real-time multi-agent visual orchestration:
```bash
python run_web_app.py
```
This automatically starts the FastAPI server and opens **`http://127.0.0.1:8000`** in your default web browser.

### 3. Run the Complete Master Demonstration
Execute the end-to-end automated pipeline covering all 16 tasks in a single command:
```bash
python run_all_demonstrations.py
```

### 4. Run Automated Test Suite
```bash
pytest -v
```

### 5. Run Individual Components
- **Part 1 (Dataset & RAG Core):**
  ```bash
  python dataset.py
  python rag_core.py
  ```
- **Part 2 (CrewAI Multi-Agent System & Guardrails):**
  ```bash
  python crew_agents.py
  ```
- **Part 3 (FastAPI Server, WebSocket & 15-Query Evaluation):**
  ```bash
  python api_server.py
  python evaluation.py
  ```
- **Part 4 (Autogen Review Stage & AI Governance):**
  ```bash
  python governance_review.py
  ```

---

## 8. File Structure

```
├── run_web_app.py            # Interactive Web App launcher (starts server & opens browser)
├── api_server.py             # Tasks 11-12: FastAPI REST endpoints, WebSocket chat, ELK JSONL logger
├── static/
│   └── index.html            # Interactive Web Chat UI, multi-agent visualizer & audit console
├── crew_agents.py            # Tasks 7-10: 3 CrewAI agents, LangChain memory, Pydantic schema, guardrails
├── mock_llm.py               # BaseLLM extension avoiding ReAct template traps & telemetry
├── governance_review.py      # Tasks 14-16: Autogen review team, AI governance, response cache
├── tools.py                  # Task 6: check_order_status tool with escalation formula & RAG tool
├── rag_core.py               # Tasks 3-5: Dual chunking, ChromaDB, calibration & evaluation
├── knowledge_base.py         # Task 2: 12 author-crafted Nykaa policy documents
├── dataset.py                # Task 1: Deterministic order dataset generator & verification
├── evaluation.py             # Task 13: 15-query LLM-as-judge evaluation benchmark
├── run_all_demonstrations.py # Master verification script for all 16 tasks
├── tests/
│   └── test_capstone.py      # Pytest test suite (11 automated tests)
├── transcripts/
│   └── full_demonstration_transcript.txt # Verbatim execution transcript
├── support_requests.jsonl    # ELK-style audit log with strictly masked PII
├── requirements.txt          # Pinned project dependencies
├── .gitignore                # Git ignore patterns
└── README.md                 # Complete project report, benchmarks, and rubric documentation
```
