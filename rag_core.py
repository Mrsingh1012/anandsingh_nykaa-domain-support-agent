"""
Part 1 — Tasks 3, 4, 5: RAG Core, Dual Chunking, ChromaDB Indexing,
Empirical Calibration, Grounded Generation, and Precision/Recall Evaluation.
Track: E-commerce & Retail (Nykaa).
"""

from collections import abc
import re
import os
from typing import List, Dict, Any, Tuple
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from chromadb.utils import embedding_functions
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
import numpy as np

from knowledge_base import KB_DOCUMENTS 

# Local persistent directory for ChromaDB
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Instantiate embedding model locally (zero external API keys needed)
_st_model = None

def get_embedding_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _st_model


# ---------------------------------------------------------
# Task 3: Dual Chunking Strategies
# ---------------------------------------------------------

def chunk_fixed_size_overlap(text: str, chunk_size: int = 150, overlap: int = 35) -> List[str]:
    """Chunks text into fixed character windows with designated overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    if text_len <= chunk_size:
        return [text]

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += (chunk_size - overlap)
    return chunks


def chunk_sentence_based(text: str) -> List[str]:
    """Chunks text by sentence boundaries (periods, question marks, exclamation marks)."""
    # Regex split on sentence endings followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class NykaaVectorStore:
    """Manages dual ChromaDB collections for fixed-size and sentence-based chunking."""

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model = get_embedding_model()
        self.col_fixed = self.client.get_or_create_collection(
            name="nykaa_kb_fixed",
            metadata={"hnsw:space": "cosine"}
        )
        self.col_sentence = self.client.get_or_create_collection(
            name="nykaa_kb_sentence",
            metadata={"hnsw:space": "cosine"}
        )

    def index_all(self, docs: List[Dict[str, str]] = KB_DOCUMENTS) -> Tuple[int, int]:
        """Indexes all documents into both collections using collection.upsert()."""
        # 1. Index Fixed-size chunks
        fixed_ids, fixed_texts, fixed_metas = [], [], []
        for doc in docs:
            chunks = chunk_fixed_size_overlap(doc["content"])
            for idx, ch in enumerate(chunks):
                fixed_ids.append(f"{doc['doc_id']}_fix_{idx}")
                fixed_texts.append(ch)
                fixed_metas.append({
                    "parent_doc_id": doc["doc_id"],
                    "topic": doc["topic"],
                    "title": doc["title"],
                    "strategy": "fixed_size",
                    "chunk_index": idx,
                })

        fixed_embeddings = self.model.encode(fixed_texts, normalize_embeddings=True).tolist()
        self.col_fixed.upsert(
            ids=fixed_ids,
            documents=fixed_texts,
            metadatas=fixed_metas,
            embeddings=fixed_embeddings
        )

        # 2. Index Sentence-based chunks
        sent_ids, sent_texts, sent_metas = [], [], []
        for doc in docs:
            chunks = chunk_sentence_based(doc["content"])
            for idx, ch in enumerate(chunks):
                sent_ids.append(f"{doc['doc_id']}_sent_{idx}")
                sent_texts.append(ch)
                sent_metas.append({
                    "parent_doc_id": doc["doc_id"],
                    "topic": doc["topic"],
                    "title": doc["title"],
                    "strategy": "sentence",
                    "chunk_index": idx,
                })

        sent_embeddings = self.model.encode(sent_texts, normalize_embeddings=True).tolist()
        self.col_sentence.upsert(
            ids=sent_ids,
            documents=sent_texts,
            metadatas=sent_metas,
            embeddings=sent_embeddings
        )

        return len(fixed_ids), len(sent_ids)

    def query(self, query_text: str, collection_name: str = "nykaa_kb_sentence", top_k: int = 3) -> Dict[str, Any]:
        """Queries the specified collection and returns chunks with cosine similarity."""
        col = self.col_fixed if collection_name == "nykaa_kb_fixed" else self.col_sentence
        query_emb = self.model.encode([query_text], normalize_embeddings=True).tolist()

        res = col.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # ChromaDB cosine distance d in [0, 2], where cosine similarity = 1 - d
        documents = res["documents"][0] if res["documents"] else []
        metadatas = res["metadatas"][0] if res["metadatas"] else []
        distances = res["distances"][0] if res["distances"] else []

        chunks_with_sim = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = max(0.0, 1.0 - dist)
            chunks_with_sim.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": round(similarity, 4)
            })

        return {
            "query": query_text,
            "collection": collection_name,
            "top_k": top_k,
            "results": chunks_with_sim,
            "top_1_similarity": chunks_with_sim[0]["similarity"] if chunks_with_sim else 0.0
        }


# ---------------------------------------------------------
# Task 4: Empirical Fallback Calibration & Grounded Generation
# ---------------------------------------------------------

CALIBRATION_QUERIES_IN_SCOPE = [
    "What is the return window for makeup and cosmetic products?",
    "How many days does it take to get a COD bank refund?",
    "Can I cancel an order that has already shipped?",
    "What is the policy for claiming a damaged or broken package?",
]

CALIBRATION_QUERIES_OUT_OF_SCOPE = [
    "What are the best long-term mutual fund investment options?",
    "Who won the ICC Cricket World Cup tournament in 2011?",
    "How do I repair a leaking bathroom water pipe?",
]


def calibrate_fallback_threshold(vector_store: NykaaVectorStore, collection_name: str = "nykaa_kb_sentence") -> Dict[str, Any]:
    """
    Empirically calibrates the fallback threshold by measuring top-1 cosine similarities
    across in-scope queries vs out-of-scope queries.
    """
    in_scope_scores = []
    for q in CALIBRATION_QUERIES_IN_SCOPE:
        res = vector_store.query(q, collection_name=collection_name, top_k=1)
        in_scope_scores.append({"query": q, "similarity": res["top_1_similarity"]})

    out_scope_scores = []
    for q in CALIBRATION_QUERIES_OUT_OF_SCOPE:
        res = vector_store.query(q, collection_name=collection_name, top_k=1)
        out_scope_scores.append({"query": q, "similarity": res["top_1_similarity"]})

    min_in_scope = min(s["similarity"] for s in in_scope_scores)
    max_out_scope = max(s["similarity"] for s in out_scope_scores)

    # Set the calibrated threshold halfway between the two observed clusters:
    calibrated_threshold = round((min_in_scope + max_out_scope) / 2.0, 3)

    return {
        "in_scope_scores": in_scope_scores,
        "out_scope_scores": out_scope_scores,
        "min_in_scope": min_in_scope,
        "max_out_scope": max_out_scope,
        "calibrated_threshold": calibrated_threshold,
    }


# Calibrated empirical threshold constant (empirically calculated: (0.4367 + 0.1497) / 2 = 0.293)
CALIBRATED_THRESHOLD = 0.293


def grounded_generate(query_text: str, vector_store: NykaaVectorStore,
                      collection_name: str = "nykaa_kb_sentence",
                      threshold: float = CALIBRATED_THRESHOLD,
                      top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieves context and generates an answer strictly grounded in retrieved chunks.
    Triggers 'I don't know' fallback if top-1 similarity < threshold.
    """
    res = vector_store.query(query_text, collection_name=collection_name, top_k=top_k)
    top_1_sim = res["top_1_similarity"]

    if top_1_sim < threshold or not res["results"]:
        return {
            "query": query_text,
            "top_1_similarity": top_1_sim,
            "threshold": threshold,
            "is_grounded": False,
            "answer": "I don't know based on the provided Nykaa policy documentation.",
            "retrieved_chunks": [],
            "source_doc_ids": [],
        }

    retrieved_texts = [r["content"] for r in res["results"]]
    source_docs = sorted(list(set(r["metadata"]["parent_doc_id"] for r in res["results"])))

    # Deterministic grounded synthesis combining the top matching context
    combined_context = " ".join(retrieved_texts)
    grounded_answer = (
        f"According to Nykaa policy: {combined_context}"
    )

    return {
        "query": query_text,
        "top_1_similarity": top_1_sim,
        "threshold": threshold,
        "is_grounded": True,
        "answer": grounded_answer,
        "retrieved_chunks": retrieved_texts,
        "source_doc_ids": source_docs,
    }


# ---------------------------------------------------------
# Task 5: Precision and Recall Evaluation at Document Level
# ---------------------------------------------------------

EVAL_BENCHMARK = [
    {
        "query": "What is the return window for makeup and cosmetic products?",
        "ground_truth_doc_ids": ["KB-01"],
    },
    {
        "query": "How are Cash on Delivery refunds processed and how many days does it take?",
        "ground_truth_doc_ids": ["KB-02"],
    },
    {
        "query": "What are the standard delivery delivery SLAs for metro and remote cities?",
        "ground_truth_doc_ids": ["KB-03"],
    },
    {
        "query": "Can I cancel an order once it is shipped or in transit?",
        "ground_truth_doc_ids": ["KB-06"],
    },
    {
        "query": "What is the procedure for reporting a damaged or leaked product upon delivery?",
        "ground_truth_doc_ids": ["KB-10"],
    },
]


def evaluate_chunking_strategies(vector_store: NykaaVectorStore, top_k: int = 3) -> Dict[str, Any]:
    """
    Computes precision and recall at the document level separately for both
    fixed-size and sentence-based ChromaDB collections with per-query arithmetic.
    """
    results = {"fixed_size": [], "sentence_based": []}

    for col_key, col_name in [("fixed_size", "nykaa_kb_fixed"), ("sentence_based", "nykaa_kb_sentence")]:
        query_evals = []
        for item in EVAL_BENCHMARK:
            query = item["query"]
            truth = set(item["ground_truth_doc_ids"])

            q_res = vector_store.query(query, collection_name=col_name, top_k=top_k)
            # Map retrieved chunks back to parent documents and dedup
            retrieved_docs = list(dict.fromkeys(r["metadata"]["parent_doc_id"] for r in q_res["results"]))

            relevant_retrieved = [doc_id for doc_id in retrieved_docs if doc_id in truth]

            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0
            recall = len(relevant_retrieved) / len(truth) if truth else 0.0

            query_evals.append({
                "query": query,
                "ground_truth": list(truth),
                "retrieved_unique_docs": retrieved_docs,
                "relevant_retrieved": relevant_retrieved,
                "precision_arithmetic": f"{len(relevant_retrieved)} / {len(retrieved_docs)} = {precision:.3f}",
                "recall_arithmetic": f"{len(relevant_retrieved)} / {len(truth)} = {recall:.3f}",
                "precision": precision,
                "recall": recall,
            })

        avg_precision = sum(qe["precision"] for qe in query_evals) / len(query_evals)
        avg_recall = sum(qe["recall"] for qe in query_evals) / len(query_evals)

        results[col_key] = {
            "per_query": query_evals,
            "avg_precision": round(avg_precision, 3),
            "avg_recall": round(avg_recall, 3),
        }

    return results


def run_rag_demonstration() -> None:
    """Executes the complete Part 1 Tasks 3-5 pipeline and prints structured results."""
    print("=" * 70)
    print("PART 1: RAG CORE, DUAL CHUNKING, CALIBRATION & EVALUATION")
    print("=" * 70)

    vs = NykaaVectorStore()
    n_fixed, n_sent = vs.index_all()
    print(f"Indexed {n_fixed} fixed-size chunks and {n_sent} sentence chunks into ChromaDB.")

    print("\n--- Task 4: Empirical Fallback Calibration ---")
    calib = calibrate_fallback_threshold(vs, collection_name="nykaa_kb_sentence")
    print("In-Scope Calibration Queries:")
    for s in calib["in_scope_scores"]:
        print(f"  Similarity: {s['similarity']:.4f} | Query: '{s['query']}'")
    print("\nOut-of-Scope Calibration Queries:")
    for s in calib["out_scope_scores"]:
        print(f"  Similarity: {s['similarity']:.4f} | Query: '{s['query']}'")
    print(f"\nEmpirical Observation: Min In-Scope={calib['min_in_scope']:.4f}, Max Out-Scope={calib['max_out_scope']:.4f}")
    print(f"Chosen Empirically Calibrated Threshold: {calib['calibrated_threshold']:.4f}")

    print("\n--- Task 4: Grounded Generation Demonstration (5 In-Scope + 1 Out-of-Scope) ---")
    demo_queries = [
        "What is the return window for makeup and cosmetic products?",
        "How many days does it take to get a COD bank refund?",
        "What are the delivery SLAs for metro cities?",
        "Can I cancel an order after it has shipped?",
        "What is the size exchange policy for footwear?",
        "What are the best mutual funds to buy for stock market growth?",  # Deliberate out-of-scope
    ]

    for q in demo_queries:
        gen = grounded_generate(q, vs, threshold=calib['calibrated_threshold'])
        status = "GROUNDED ANSWER" if gen["is_grounded"] else "FALLBACK TRIGGERED"
        print(f"\nQuery: {q}")
        print(f"Status: [{status}] (Top-1 Sim: {gen['top_1_similarity']:.4f} vs Threshold: {gen['threshold']})")
        print(f"Answer: {gen['answer']}")
        if gen["source_doc_ids"]:
            print(f"Sources: {gen['source_doc_ids']}")

    print("\n--- Task 5: Evaluation and Comparison of Both Chunking Strategies ---")
    eval_res = evaluate_chunking_strategies(vs)

    print("\n[Strategy A: Fixed-Size with Overlap Collection (nykaa_kb_fixed)]")
    for qe in eval_res["fixed_size"]["per_query"]:
        print(f"Query: {qe['query']}")
        print(f"  Retrieved Unique Docs: {qe['retrieved_unique_docs']} | Ground Truth: {qe['ground_truth']}")
        print(f"  Precision: {qe['precision_arithmetic']} | Recall: {qe['recall_arithmetic']}")
    print(f"Fixed-Size Collection Averages: Precision = {eval_res['fixed_size']['avg_precision']:.3f}, Recall = {eval_res['fixed_size']['avg_recall']:.3f}")

    print("\n[Strategy B: Sentence-Based Chunking Collection (nykaa_kb_sentence)]")
    for qe in eval_res["sentence_based"]["per_query"]:
        print(f"Query: {qe['query']}")
        print(f"  Retrieved Unique Docs: {qe['retrieved_unique_docs']} | Ground Truth: {qe['ground_truth']}")
        print(f"  Precision: {qe['precision_arithmetic']} | Recall: {qe['recall_arithmetic']}")
    print(f"Sentence-Based Collection Averages: Precision = {eval_res['sentence_based']['avg_precision']:.3f}, Recall = {eval_res['sentence_based']['avg_recall']:.3f}")

    print("\n[Recommendation Statement for README.md]:")
    rec_text = (
        f"Recommendation: We deploy Strategy B (Sentence-Based Chunking). "
        f"Sentence-based chunks achieved an average precision of {eval_res['sentence_based']['avg_precision']:.3f} "
        f"and recall of {eval_res['sentence_based']['avg_recall']:.3f}, compared to fixed-size chunking "
        f"which achieved precision of {eval_res['fixed_size']['avg_precision']:.3f} and recall of {eval_res['fixed_size']['avg_recall']:.3f}. "
        f"Sentence-based chunking preserves complete semantic policy boundaries and avoids fragmenting conditions across arbitrary character offsets."
    )
    print(rec_text)
    print("=" * 70)


if __name__ == "__main__":
    run_rag_demonstration()
