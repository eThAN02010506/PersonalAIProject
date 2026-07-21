# Qwopus-Agent

Qwopus-Agent is a local-first, modular AI Agent framework designed for Apple Silicon Mac
workflows and MLX-hosted OpenAI-compatible local models.

The project is interface-first and keeps orchestration, skills, memory, and UI independently
testable:

- unified LLM abstraction through `BaseLLM`
- provider-neutral model creation through `LLMConfig` and `LLMRegistry`
- local MLX model adapter through `LocalMLXLLM`
- reusable capability abstraction through `BaseSkill` and automatic discovery
- separate Planner, Executor, and Agent Router
- supervised Multi-Agent delegation, shared state, parallel waves, debate, and arbitration
- validated, versioned, persistent Workflow Skill growth
- real Markdown, Excel, PNG/SVG chart, and PDF report artifacts
- working module boundaries for memory, reflection, skills, reports, prompts, storage, and logs
- runnable tests for the core contracts

## Project Layout

```text
src/qwopus_agent/
  api/          FastAPI boundary, SQLite conversations, background runs, and SPA hosting
  agents/       Planner, Executor, Router, research, and supervised Multi-Agent orchestration
  llm/          BaseLLM interface and LocalMLXLLM adapter
  memory/       MiniRAG vectors, persistent knowledge graph, and multi-hop queries
  skills/       Reusable, auto-discovered, and learned Workflow Skills
  services/     UI-independent analysis and knowledge lifecycle workflows
  reflection/   Lightweight task reflection evaluator
  reports/      Unified report generation module
  prompts/      Prompt templates and system prompt assets
frontend/       React, assistant-ui, Vite, and the production chat/document workspace
tests/          Unit tests for first-stage architecture
storage/        Runtime data, ignored by Git except .gitkeep
logs/           Runtime logs, ignored by Git except .gitkeep
```

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e vendor/MiniRAG-main
pip install -e ".[dev,ui,api,documents]"
pytest
```

## Local MLX Server

The `LocalMLXLLM` adapter expects an OpenAI-compatible chat completions endpoint, such as:

```bash
python -m mlx_lm.server --model ~/Desktop/model/gemma-4-12B-it-qat-OptiQ-4bit
```

Model files are intentionally kept outside this repository. Qwopus-Agent should connect to the local
MLX API server and must not commit model weights, quantized checkpoints, or local benchmark artifacts.

Then configure the adapter with the local base URL:

```python
from qwopus_agent.llm import LLMConfig, create_default_llm_registry

registry = create_default_llm_registry()
llm = registry.create(LLMConfig(
    provider="local_mlx",
    model="gemma-4-12B-it-qat-OptiQ-4bit",
    base_url="http://127.0.0.1:8080/v1",
))
```

The Agent only depends on `BaseLLM`. To switch models, change `LLMConfig`; to support a new backend,
register a new provider factory in `LLMRegistry`. OpenAI-compatible runtimes can use the generic
`openai_compatible` provider without writing a new adapter.

## smolagents Smoke Test

The first application milestone is only to verify:

```text
smolagents -> local OpenAI-compatible MLX server -> model response
```

Start your local model server first:

```bash
python -m mlx_lm.server --model ~/Desktop/model/gemma-4-12B-it-qat-OptiQ-4bit
```

Then run:

```bash
qwopus-smolagents-smoke "用一句中文回答：你是否已经连接到本地大模型？"
```

Or without installing the console script:

```bash
PYTHONPATH=src python3.11 -m qwopus_agent.integrations.smolagents_smoke \
  "用一句中文回答：你是否已经连接到本地大模型？"
```

This smoke test intentionally creates a smolagents `CodeAgent` with no tools. The Streamlit workflow
then injects document, pandas sandbox, MiniRAG, graph, and Tavily capabilities as bounded tools.

## Primary React Application

The primary application uses FastAPI with a React/assistant-ui frontend while reusing the same
`AgentOrchestrator`, MinerU, MiniRAG, knowledge graph, Skills, and reports:

```bash
pip install -e ".[api,documents]"
cd frontend
pnpm install
pnpm run build
cd ..
PYTHONPATH=src python -m uvicorn qwopus_agent.api.app:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010`. The API serves `frontend/dist` in this production-style mode. For
frontend hot reload, run the API on port 8000 and `pnpm run dev` in `frontend`; Vite proxies `/api`
to port 8000.

The application boundaries are intentional:

```text
React + assistant-ui
        |
        v
FastAPI API + SQLite conversations
        |
        v
AgentOrchestrator + smolagents
        |
        +--> MinerU -> Markdown -> document/excel Skills
        +--> persistent MiniRAG + knowledge graph
        +--> Tavily, Multi-Agent, reports, reflection, and learned Skills
```

- React renders chat, document analysis, citations, run status, and report downloads. It never calls
  the model server or parses documents directly.
- FastAPI owns HTTP validation, conversation persistence, background run polling, uploads, and SPA
  hosting. Business decisions remain in `AgentOrchestrator` and the existing services.
- PDF, DOCX, PNG, and JPEG inputs continue through MinerU when available, with the established local
  fallback behavior. All unstructured content is normalized to Markdown before downstream use.
- Document analysis searches existing MiniRAG knowledge, inserts each newly normalized document into
  `storage/minirag/`, and gives smolagents bounded document, pandas, and MiniRAG tools. The browser only
  receives the final answer, citations, safe process events, and generated report links.
- The OpenAI-compatible endpoint is read from the existing environment configuration. The current
  model identifier is resolved from the server for each run, so changing Gemma, Qwopus, Qwen, or
  another compatible model does not require a frontend code change.

Conversations are persisted in `storage/qwopus.db`, and existing Streamlit JSONL history is imported
once when that database is first created. Streamlit remains available at its own port as a debugging
console; the production React application links to it but does not depend on Streamlit Session State.

Verify the new boundary with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest tests.test_api
cd frontend && pnpm run lint && pnpm run build
```

## Streamlit Chat And Upload Analysis

After the smoke test passes, verify multi-turn conversation and local upload analysis through
Streamlit:

```bash
pip install -e ".[dev,ui]"
python -m mlx_lm.server --model ~/Desktop/model/gemma-4-12B-it-qat-OptiQ-4bit --port 8080
PYTHONPATH=src streamlit run src/qwopus_agent/ui/streamlit_chat.py
```

The Streamlit page provides:

- sidebar model configuration and connection check
- multi-turn chat via `st.chat_message` and `st.chat_input`
- conversation history passed into smolagents through `run_smolagents_chat_turn`
- upload analysis for PDF, DOCX, Markdown, TXT, CSV, XLSX, and XLS files
- local pandas spreadsheet inspection with schema, sample rows, and numeric summaries
- MinerU-backed PDF/image parsing with Markdown normalization
- persistent vector and knowledge-graph retrieval with source/page evidence
- entity-type filtering, directed graph visualization, and DOT download
- per-source update, deletion, and derived-index rebuild controls
- a local-only Debug Console with safe orchestration events, complete smolagents prompts, raw model
  outputs, Tool arguments, Tool Observations, step errors, JSON trace downloads, and runtime logs

The Debug Console intentionally exposes more information than the primary React application and may
contain full local document excerpts. Use it only on a trusted local machine. It displays the raw
fields returned by smolagents and the current model server; it cannot display hidden reasoning that
the model/provider did not return. FastAPI `RunView` and `AnalysisView` do not serialize raw debug
runs, so the primary React frontend continues to receive only final answers, citations, and safe
process events.

Manual checks:

1. Click "检测模型连接" and confirm the MLX server is online.
2. Send "你好，请用中文自我介绍" and confirm a response appears.
3. Ask "上一句你说了什么？" and confirm the reply uses prior context.
4. Upload a CSV or Excel file and confirm schema/sample/numeric summary tables render.
5. Upload a TXT, Markdown, PDF, or DOCX file and confirm the Markdown preview renders.
6. Stop the MLX server and confirm the UI shows a clear offline error.

Runtime MiniRAG data lives under `storage/minirag`. Uploaded Markdown-normalized documents and safe
spreadsheet summaries are inserted through the `MiniRAG.insert(document)` facade, and later analysis
can retrieve existing context through `MiniRAG.search(query)`. The facade uses MiniRAG's persistent
NanoVectorDB backend and a local multilingual sentence-transformer, so retrieval does not depend on
the Gemma, Qwen, or other chat model currently served by the OpenAI-compatible endpoint. The installed
`minirag-hku` package is linked to `vendor/MiniRAG-main`; Qwopus adds a persistent directed graph layer
for evidence-bound entities, relations, cross-document aggregation, and bounded multi-hop paths.

Download the default embedding model once before the first document upload:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

After that initial download, indexing and search run locally. Set `QWOPUS_EMBEDDING_MODEL` to another
locally cached sentence-transformer when needed; changing it automatically rebuilds only the derived
vector index while preserving the original JSONL documents.

Reports are generated through `qwopus_agent.reports.ReportGenerator`, which creates Markdown, Excel
workbooks, real PNG/SVG charts, and a PDF artifact under `storage/reports`.

Reflection and research orchestration remain independently testable:

- `TaskReflectionEvaluator` checks basic execution quality and trace completeness.
- `ResearchAgent` reuses `AgentRouter` and reflection.
- `MultiAgentSupervisor` owns delegation, dependency-aware parallel execution, shared state, debate,
  and conflict arbitration.
- `SkillGrowthService` extracts repeated successful traces into validated, versioned Workflow Skills.

## Current Scope

Tavily provides live web search through smolagents. A browser provider still needs to be injected for
full browser automation; the core Browser Skill intentionally does not bind the framework to one
desktop or browser backend.
