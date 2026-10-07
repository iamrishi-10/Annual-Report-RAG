# Annual_Report_RAG

A Retrieval-Augmented Generation (RAG) system for querying corporate annual reports, built and evaluated against Infosys' 2024-25 Integrated Annual Report (369 pages). Every answer cites the report pages it came from.

Annual reports are a hard RAG target: dense financial tables, image-heavy pages, and answers where a wrong number is a real problem. This project is built around that difficulty. Each pipeline stage is its own single-responsibility module, and each major design decision (which reranker, whether MultiQuery is worth its cost, which prompt changes to make) is backed by an evaluation number rather than intuition.

**Contents:** [Highlights](#highlights) · [Architecture](#architecture) · [Results](#results) · [Setup](#setup) · [Usage](#usage) · [Design decisions](#design-decisions) · [Project structure](#project-structure) · [Limitations](#limitations)

**More documentation**
- [`.claude/CLAUDE.md`](.claude/CLAUDE.md): full architecture reference (every file, function and gotcha).
- [`Interview_Explanation/`](Interview_Explanation/00_START_HERE.md): plain-language, function-by-function walkthrough for talking through the project out loud.

## Highlights

- **Page-level traceability:** every chunk carries its PDF page number, and answers cite sources inline. The UI shows them in a table.
- **Measured retrieval choices:** a 6-strategy comparison shows Cohere reranking is the biggest quality lever. MultiQuery adds a smaller recall gain on top.
- **Three-layer evaluation:** answer/source-page checks, retrieval-ranking metrics, and RAGAS semantic scoring, all over a hand-written golden dataset.
- **Streaming demo:** a Gradio app streams the answer token by token, with sources visible before the first token arrives.
- **Honest scorecard:** known failures and gaps are documented, not hidden (see [Limitations](#limitations)).

## Architecture

```
PDF (infosys-ar-25.pdf, 369 pages)
 │
 ▼
LangChain PDF Loader  +  Direct PyMuPDF Inspection  →  Hybrid cross-check
 ▼
Metadata Enrichment  (page_id, page_number, image/drawing counts)
 ▼
Chunking  (RecursiveCharacterTextSplitter, 1000 chars / 150 overlap)
 ▼
Metadata Cleaning  (allowlist before Pinecone)
 ▼
OpenAI Embeddings  (text-embedding-3-small, 1536-d)
 ▼
Pinecone Vector Store  (serverless, cosine similarity)
 ▼
═══════════════════ ingested once, queried many times ═══════════════════
 │
 ▼  question
Similarity Retrieval  (TOP_K=20)
 ▼
Multi-Query Expansion  (LLM generates alternate phrasings, unions results)
 ▼
Cohere Reranking  (rerank-v3.5, narrows to FINAL_TOP_K=5)
 ▼
Context Formatting  →  Prompt Construction  (grounding + numeric-precision + citation rules)
 ▼
LLM Answer  (gpt-4o-mini)  ──┬── normal call    → used by evaluation
                             └── streaming call → used by the Gradio demo
 ▼
Answer + page-number citations
```

Retrieval and reranking are never streamed. Only the final LLM answer is. The streaming and non-streaming paths share the same prompt, LLM and retrieval pipeline, so concatenating every streamed chunk reproduces the non-streaming answer exactly.

## Results

All evaluation runs use the first 30 questions of a hand-written 100-question golden dataset (9 categories). Result files are committed under [`data/evaluation/`](data/evaluation/).

### Basic evaluation (`scripts/15`)

| Metric | Score |
|---|---|
| Answered (not a fallback) | 29 / 30 |
| Source-page hit rate | 100% |

### Retrieval strategy comparison (`scripts/17`)

This comparison is why the pipeline uses MultiQuery + Cohere rerank.

| Strategy | Hit rate | Precision@k | Recall@k | MRR |
|---|---|---|---|---|
| similarity only | 86.67% | 0.293 | 0.689 | 0.528 |
| similarity + rerank | 100% | 0.340 | 0.839 | 0.824 |
| multiquery only | 83.33% | 0.253 | 0.611 | 0.480 |
| **multiquery + rerank** (used in production) | **100%** | **0.347** | **0.850** | **0.824** |
| MMR only | 46.67% | 0.107 | 0.294 | 0.361 |
| MMR + rerank | 80% | 0.247 | 0.606 | 0.700 |

Reranking is the dominant lever: every reranked strategy beats its non-reranked counterpart by a wide margin. MultiQuery adds a smaller but real recall gain on top, at the cost of one extra LLM call per query. MMR was tested and rejected, because annual-report questions want the *most* relevant chunks, not a diverse spread.

### RAGAS semantic evaluation (`scripts/18`)

Baseline, measured before the prompt tuning described below:

| Metric | Score |
|---|---|
| context_recall | 1.000 |
| answer_relevancy | 0.933 |
| context_precision | 0.840 |
| answer_correctness | 0.755 |
| faithfulness | 0.703 |

Faithfulness and answer_correctness were the weakest scores. Prompt changes in [`src/config/prompts.py`](src/config/prompts.py) target them directly (see [`Interview_Explanation/05_context_and_prompts.md`](Interview_Explanation/05_context_and_prompts.md)). **These changes have not been re-measured yet.** Re-running `scripts/15` then `scripts/18` is the next step.

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). You also need accounts and API keys for OpenAI, Pinecone and Cohere. The Cohere trial key is capped at 10 calls/minute, which the comparison script throttles around.

```bash
uv python install 3.12
uv venv
uv sync
cp .env.example .env
```

Fill in `.env`:

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=annual-report-rag
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
EVALUATION_MODEL=gpt-4o-mini
COHERE_API_KEY=...

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=annual-report-rag
```

## Usage

**1. Ingest the report** (loads the PDF, chunks, embeds and indexes into Pinecone; idempotent, safe to re-run):
```bash
uv run python scripts/09_run_ingestion_pipeline.py
```

**2. Launch the demo:**
```bash
uv run python scripts/20_run_gradio_app.py
```
Ask a question, watch the answer stream in, and see the cited source pages in a table alongside it.

**3. Run the evaluation suite.** Steps 15, 17 and 18 make real API calls, and 17 is the most expensive:
```bash
uv run python scripts/15_run_basic_evaluation.py          # answer/source-page scoring
uv run python scripts/16_evaluate_retrieval.py            # ranking metrics (free, reuses 15's output)
uv run python scripts/17_compare_retrieval_strategies.py  # 6-strategy comparison
uv run python scripts/18_run_ragas_evaluation.py          # RAGAS semantic scoring
```

Scripts 16 and 18 read the saved `basic_evaluation_results.json`, not the live golden dataset. If you edit `golden_questions.json`, re-run `scripts/15` first.

### Script reference

Scripts 01–08 each re-run the pipeline from the top as a stage-by-stage preview. Scripts 10–20 require script 09 to have populated the index.

| # | Script | What it does |
|---|---|---|
| 00 / 00b | `test_langsmith_trace.py`, `test_langsmith_rag_trace.py` | LangSmith tracing smoke tests (a toy chain, then the real RAG chain) |
| 00c / 00d | `debug_manufacturing_revenue.py`, `debug_test_cases.py` | Stage-by-stage debug sweeps that check where an expected page drops out of the pipeline |
| 01 | `run_pdf_inspection.py` | Per-page signals via PyMuPDF (image count, low-text flag) |
| 02 | `run_hybrid_inspection.py` | Cross-checks LangChain vs. PyMuPDF page counts |
| 03 | `run_metadata_enrichment.py` | Merges inspection data onto each page's metadata |
| 04 | `run_chunking.py` | Splits pages into ~1000-char chunks |
| 05 | `run_metadata_cleaning.py` | Trims chunk metadata to an allowlist |
| 06 | `run_embedding_preview.py` | Embeds a small sample (first script to call OpenAI) |
| 07 | `test_pinecone_connection.py` | Confirms the Pinecone index is reachable |
| 08 | `run_vectorstore_indexing.py` | Indexes a 5-chunk preview |
| 09 | `run_ingestion_pipeline.py` | **Full-corpus ingestion**: run this before anything below |
| 10 | `test_retrieval.py` | Plain similarity retrieval smoke test |
| 11 | `test_context_builder.py` | Formats retrieved chunks into a prompt-ready string |
| 12 | `test_answer_generation.py` | Historical smoke test predating the main chain (no MultiQuery/rerank) |
| 13 | `test_rag_chain.py` | The real main chain: MultiQuery + rerank + generate |
| 14 | `inspect_reranking.py` | Cohere candidates vs. final reranked chunks, side by side |
| 15 | `run_basic_evaluation.py` | 30 golden questions → answer/source-page scoring. **Calls the LLM and reranker.** |
| 16 | `evaluate_retrieval.py` | Retrieval-ranking metrics from 15's saved output; no extra cost |
| 17 | `compare_retrieval_strategies.py` | 6-way retrieval strategy comparison; the most expensive eval script |
| 18 | `run_ragas_evaluation.py` | RAGAS semantic scoring; real LLM and embedding cost |
| 20 | `run_gradio_app.py` | Launches the demo UI |

There is no `19`. It is deliberately left open for a future latency/cost-tracking script.

## Design decisions

- **Cohere reranking, not Pinecone's hosted reranker.** The original plan favored keeping the vendor surface to OpenAI + Pinecone. I switched to Cohere's `rerank-v3.5` to validate with a proven implementation whether reranking mattered. It did: reranking is the biggest quality lever measured here.
- **MultiQuery retrieval, justified by measurement.** Query expansion costs one extra LLM call per question. It is kept because the comparison shows a recall improvement over similarity + rerank alone, not because it is a well-known technique.
- **Prompt blocks added in response to measured weaknesses.** The numeric-precision and tightened-grounding instructions exist because RAGAS measured faithfulness at 0.703 and answer_correctness at 0.755, the two weakest of five metrics.
- **Fact-recall scoring was built, then deleted.** An early evaluator matched answers against hand-written `expected_facts` by substring. It produced near-zero, meaningless scores because the facts were sentence-length rather than atomic. Semantic correctness is now left to RAGAS, which judges meaning rather than string overlap.
- **One golden-dataset failure left uncorrected on purpose.** `financial_014` and `financial_019` had their expected pages extended after I confirmed the system was citing an equally valid second page. `business_segments_009` was left as a real retrieval failure, so the evaluation includes at least one honest failure case.
- **Streaming alongside the normal path, not instead of it.** `stream_rag_query()` powers the demo. `run_rag_query()` is what every evaluation script uses, so UI work never affects evaluation.

## Project structure

```
scripts/             numbered, single-purpose entry points (see script reference above)
src/
  document_loading/  PDF → LangChain Documents + PyMuPDF page inspection
  preprocessing/     metadata enrichment + cleaning
  chunking/          RecursiveCharacterTextSplitter wrapper
  embeddings/        OpenAI embedding model + embedding calls
  vectorstore/       Pinecone client + PineconeVectorStore wiring
  ingestion/         full ingestion pipeline orchestration
  retrieval/         similarity retriever + MultiQuery wrapper
  reranking/         Cohere reranker
  context/           prompt-ready context formatting
  config/            settings, API key checks, prompt templates
  generation/        LLM client, answer generation (normal + streaming), the main RAG chain
  app/               Gradio demo UI
  evaluation/        golden-dataset loading, basic scoring, retrieval metrics, RAGAS integration
  utils/             logging, JSON saving, serialization
data/
  raw/               the source PDF
  processed/         regenerable pipeline-stage previews (gitignored)
  evaluation/        golden_questions.json (hand-written) + results and summaries (committed)
Interview_Explanation/  plain-language walkthrough of every file and function
```

## Limitations

- Metadata pre-filtering, context deduplication/token-budget assembly, and citation-claim mapping were planned but never built. Citation is prompt-level only, with no programmatic check that a claim matches its source.
- Chunk size is a single global setting. Content-type-aware chunking (tables vs. prose) is not implemented, so tables can be split mid-row.
- No automated tests exist. Verification so far is via the numbered scripts and manual output inspection.
- No latency or cost tracking.
- Evaluation covers 30 of the 100 golden questions, to bound API cost. Scaling to all 100 is a straightforward next step.
- The post-prompt-tuning RAGAS scores have not been measured.

See [`Interview_Explanation/11_common_interview_questions.md`](Interview_Explanation/11_common_interview_questions.md) for a longer breakdown of gaps and what would come next.
