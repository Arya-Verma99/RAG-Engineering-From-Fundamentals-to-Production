# RAG Mastery: A Complete Guide to Retrieval-Augmented Generation

> A step-by-step guide to RAG, from the basic ideas to production and debugging. Every chapter explains the concept in simple words, shows working code, lists the trade-offs, and ends with questions and exercises. It was written to show real understanding of RAG, not just to copy a tutorial.

---

## About This Project

I built a chatbot that answers questions from PDFs ([Chat-with-Your-PDFs](https://github.com/Arya-Verma99/Chat-with-Your-PDFs)). Building it raised many questions. Why did the right chunk not show up? Why did the bot make things up? How do I know a change actually helped?

This guide is what I learned while answering them. It follows the real RAG pipeline in order, and every chapter focuses on the decisions that matter, the mistakes I made, and how to measure the result.

**Who is this for?**

- Beginners who want to understand RAG from the ground up
- Developers who built a basic RAG demo and want better quality
- Anyone who wants a clear, honest picture of what RAG can and cannot do

**What you need:** basic Python, and a general idea of what an LLM is. Chapter 00 covers the rest.

---

## Table of Contents

### Foundation

| # | Chapter | What you'll learn |
|---|---|---|
| 00 | [Prerequisites](docs/00-prerequisites.md) | NLP, vectors, LLMs, tokens, and the context window |
| 01 | [Introduction to RAG](docs/01-introduction.md) | What RAG is, why it exists, RAG vs fine-tuning vs long context |
| 02 | [Architecture](docs/02-architecture.md) | The ingestion pipeline and the query pipeline, step by step |

### Building the Core Pipeline

| # | Chapter | What you'll learn |
|---|---|---|
| 03 | [Data Ingestion](docs/03-data-ingestion.md) | Loading PDFs, HTML, and tables, OCR, cleaning, and metadata |
| 04 | [Chunking](docs/04-chunking.md) | Fixed, recursive, structure-aware, semantic, and parent-child splitting |
| 05 | [Embeddings](docs/05-embeddings.md) | How embeddings work, choosing a model, similarity metrics, dimensions |
| 06 | [Vector Databases](docs/06-vector-databases.md) | ANN search, HNSW, IVF, FAISS, Chroma, Qdrant, and pgvector |
| 07 | [Retrieval](docs/07-retrieval.md) | Dense, BM25, hybrid search, MMR, metadata filters, choosing k |
| 10 | [Generation](docs/10-generation.md) | Prompt design, context assembly, citations, and guardrails |

### Improving Quality

| # | Chapter | What you'll learn |
|---|---|---|
| 08 | [Query Transformation](docs/08-query-transformation.md) | Rewriting, multi-query, HyDE, step-back, decomposition, routing |
| 09 | [Reranking](docs/09-reranking.md) | Bi-encoders vs cross-encoders, retrieve-then-rerank |
| 11 | [Conversational RAG](docs/11-conversational-rag.md) | Chat memory and follow-up questions |
| 13 | [Evaluation](docs/13-evaluation.md) | Test sets, retrieval and generation metrics, LLM judges, RAGAS |

### Advanced, Production, and Debugging

| # | Chapter | What you'll learn |
|---|---|---|
| 12 | [Advanced RAG](docs/12-advanced-rag.md) | Self-RAG, Corrective RAG, Graph RAG, agentic RAG, multimodal RAG |
| 14 | [Production](docs/14-production.md) | Latency, cost, caching, scaling, monitoring, and security |
| 15 | [Failure Modes](docs/15-failure-modes.md) | A debugging playbook: why RAG fails and how to fix it |

---

## What Every Chapter Contains

Each chapter follows the same simple structure:

1. **The problem** it solves, in plain words
2. **How it works**, with diagrams and analogies
3. **Code** you can run and adapt
4. **Options and trade-offs** in comparison tables
5. **Common mistakes**
6. **Key takeaways**
7. **Quick self-check** questions
8. **Hands-on practice** exercises
9. **Further reading** (papers and documentation)

---

## The Big Ideas in One Table

| Idea | One-line takeaway | Chapter |
|---|---|---|
| RAG | Look up the right information first, then let the LLM answer from it | 01 |
| Two pipelines | Ingestion prepares the data once. Querying answers each question | 02 |
| Garbage in, garbage out | Bad parsing ruins everything that comes after it | 03 |
| Chunking | Small chunks are precise but lose context. Large chunks have context but blur the match | 04 |
| Embeddings | Similar meaning gives nearby vectors. Use one model for documents and queries | 05 |
| ANN search | Trade a little accuracy for a lot of speed. HNSW is the common choice | 06 |
| Hybrid search | Combine meaning (dense) and exact words (BM25), merge with RRF | 07 |
| Query transformation | Fix vague and follow-up questions before searching | 08 |
| Reranking | Retrieve many cheaply, rerank to a few carefully | 09 |
| Grounded generation | "Use only the context", cite sources, and say "I don't know" | 10 |
| Chat memory | Rewrite follow-ups into standalone questions, keep a short memory | 11 |
| Advanced RAG | Add checks and loops only when a measured failure justifies the cost | 12 |
| Evaluation | Measure retrieval and generation separately on a real test set | 13 |
| Production | Watch latency, cost, caching, permissions, and prompt injection | 14 |
| Debugging | Find the failing stage first. Don't edit the prompt first | 15 |

---

## Repository Structure

```
rag-mastery/
├── README.md                     
├── docs/                         # The guide, one chapter per file
│   ├── 00-prerequisites.md
│   ├── 01-introduction.md
│   ├── 02-architecture.md
│   ├── 03-data-ingestion.md
│   ├── 04-chunking.md
│   ├── 05-embeddings.md
│   ├── 06-vector-databases.md
│   ├── 07-retrieval.md
│   ├── 08-query-transformation.md
│   ├── 09-reranking.md
│   ├── 10-generation.md
│   ├── 11-conversational-rag.md
│   ├── 12-advanced-rag.md
│   ├── 13-evaluation.md
│   ├── 14-production.md
│   └── 15-failure-modes.md

---

## Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/Arya-Verma99/rag-mastery.git
cd rag-mastery
```

**2. Create a virtual environment and install the main libraries**

```bash
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

pip install numpy sentence-transformers faiss-cpu pypdf pymupdf pdfplumber \
            rank-bm25 tiktoken chromadb
```

Optional, for evaluation and OCR (see Chapters 03 and 13): `ragas`, `datasets`, `pytesseract`, `pdf2image`. OCR also needs the Tesseract and Poppler programs installed on your machine.

**3. Set your LLM API key** (if you use a hosted model)

```bash
export YOUR_PROVIDER_API_KEY="..."      # never commit keys to the repository
```

**4. Start reading** at [Chapter 00](docs/00-prerequisites.md), or jump to the chapter you need.

> The code in the chapters shows each idea in its simplest form. Library versions change, so check the official documentation for the versions you install.

---




