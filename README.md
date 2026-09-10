# OneNoteRAG — Local RAG over your "Computer Science" notebook

100% local RAG pipeline (OneNote extraction → embeddings → reranking → generation),
built around Qwen (embeddings, reranker, LLM).

This repo started as a **learning skeleton**: every file used to contain TODOs and
explanations rather than a finished implementation, so the author could write each
step by hand. It has since been implemented and tested end-to-end — see
`ARCHITECTURE.md` for the detailed design choices and alternatives.

## Pipeline execution order

```
src/extract_onenote.py   → data/raw/*.html + *.json + manifest.json + last_sync.json
src/parse_clean.py       → data/processed/*.md
src/chunk.py             → data/processed/chunks/<page_id>.jsonl
src/embed_index.py       → data/index/ (Qdrant, hybrid: dense + sparse vectors)
src/retrieve_rerank.py   → hybrid search (RRF) + reranking, reused by generate.py and evaluate_ragas.py
src/generate.py          → question/answer CLI
src/evaluate_ragas.py    → eval/results/ (RAGAS scores: faithfulness, context_precision, ...)
src/app.py               → local chat UI (Gradio), same pipeline as generate.py
```

The evaluation test set (`eval/qa_dataset.jsonl`, written by hand) follows the
format of `eval/qa_dataset.example.jsonl`.

`src/pipeline.py` is an optional orchestrator that chains everything together.

**Incremental sync**: since the notebook changes daily, `extract_onenote.py`
only re-downloads pages that were added/modified since the last run (via
`data/raw/manifest.json` + `lastModifiedDateTime`), and every following step
only reprocesses those pages (via `data/raw/last_sync.json`). See
`ARCHITECTURE.md` section "1bis" for details. In practice: just rerun the 4
scripts in order whenever you want to resync — each one only does work on
what actually changed.

## Prerequisites

1. **Azure App Registration** (for the Graph API / OneNote):
   - https://portal.azure.com → Azure Active Directory (Entra ID) → App registrations → New registration
   - Account type: "Accounts in any organizational directory and personal Microsoft accounts"
   - No secret needed if you use the "public client" flow (device code) — check
     "Allow public client flows" = Yes
   - API permissions → Microsoft Graph → Delegated → `Notes.Read`
   - Note the `Application (client) ID` → put it in `.env`

2. **Qwen, running locally**:
   - Dense embedding: `Qwen/Qwen3-Embedding-0.6B` (via `sentence-transformers`)
   - Reranker: `BAAI/bge-reranker-v2-m3` (`sentence_transformers.CrossEncoder`)
   - Generation: **Ollama** + `qwen2.5:1.5b` (GGUF, quantized — far faster on
     CPU than raw transformers weights). Install
     [Ollama](https://ollama.com/) (`winget install Ollama.Ollama` on
     Windows), then `ollama pull qwen2.5:1.5b`. The Ollama server runs in
     the background after install/on login.

3. **Python environment**:
   - `python -m venv .venv` then activate it
   - `pip install -r requirements.txt`

4. Copy `.env.example` to `.env` and fill in the values.

## Design notes

- MSAL auth (device code flow) with a local token cache
- Graph API pagination (`@odata.nextLink`)
- Cleaning OneNote HTML (inline styles, absolutely-positioned `<div>`s, base64 images...)
- Chunking without breaking a code block or a definition mid-way
- Measuring retrieval quality (RAGAS metrics)
- Prompting for a final answer with source-page citations

See `ARCHITECTURE.md` for detailed recommendations on each step.

## Possible future improvements

- **On-device mobile app (Android/iOS)**: turning this into a consumer app
  ("sign in with your Microsoft account, ask your notes") would normally
  need a hosted multi-user backend (web OAuth, multi-tenant Qdrant, server-side
  inference) — a real project of its own, not an extension of this one. A more
  interesting angle: [ZETIC.ai / Melange](https://docs.zetic.ai/) converts
  ONNX/TorchScript models into NPU-optimized mobile runtimes — the dense
  embedding model and the reranker (both "single-pass" models) are good
  candidates for running on-device with NPU acceleration. If generation
  follows too (harder, autoregressive), the whole pipeline could stay
  **on-device**: no multi-tenant backend needed at all, the app would index
  the user's OneNote directly on their phone — better privacy story as a
  bonus. Still open: a lightweight on-device vector store (Qdrant isn't built
  for mobile) and a native mobile Microsoft OAuth flow.
