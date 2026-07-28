# ⚖️ Tribunal

Adversarial multi-agent fact-checker with **just-in-time RAG** grounding.

Give it a factual claim → get a verdict (**True / False / Misleading / Unverifiable**) with a
confidence score and citations to real sources. Built to showcase the 2026-in-demand stack:
LangGraph multi-agent orchestration, RAG, an MCP server, Langfuse observability, and evals.

## Pipeline (v1)

```
claim → Decomposer → Retriever (RAG) → Analyst → Judge → verdict
```

- **Decomposer** — splits the claim into atomic sub-claims + search queries
- **Retriever (RAG)** — Wikipedia-first hybrid search → chunk → MiniLM embed → LanceDB → top-k
- **Analyst** — weighs supporting vs refuting evidence, using retrieved sources only
- **Judge** — final verdict + confidence + verbatim citations (structured via Instructor)

## Stack

LangGraph · OpenRouter (free models) · Instructor · fastembed MiniLM/ONNX (local, ~50MB) ·
LanceDB · Langfuse · FastMCP · FastAPI. **No local LLM, no torch — runs on an M1/8GB.**

## Setup

```bash
cd tribunal
uv venv && source .venv/bin/activate
uv pip install -e .          # add -e '.[evals,dev]' for RAGAS + pytest
cp .env.example .env         # then paste your OPENROUTER_API_KEY
```

## Run — three interfaces

```bash
# 1. CLI
tribunal "the Great Wall of China is visible from space"

# 2. MCP server (wire into Claude Desktop / Cursor as an MCP tool `verify_claim`)
tribunal-mcp

# 3. Web UI
uvicorn tribunal.web:app --reload      # http://127.0.0.1:8000
```

## Evals

```bash
python evals/run_evals.py     # verdict accuracy vs evals/golden.jsonl
```

## Observability

Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env` and every run streams a full
hierarchical trace (per-agent spans, tokens, cost, latency, retrieved chunks) to Langfuse.
Leave the keys blank to disable — the pipeline runs identically without it.
