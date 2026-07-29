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
  api/          FastAPI app composition, focused route modules, SQLite conversations, and runs
  agents/       Planner, Executor, Router, research, and supervised Multi-Agent orchestration
  documents/    Markdown normalization, section structure, chunking, summaries, and local storage
  integrations/ smolagents runtime, prompt policy, capability assembly, and focused Tool adapters
  llm/          BaseLLM interface and LocalMLXLLM adapter
  memory/       MiniRAG facade, conversation-scoped lifecycle, retrieval, graph, and multi-hop queries
  skills/       Reusable, auto-discovered, and learned Workflow Skills
  services/     UI-independent analysis and knowledge lifecycle workflows
  reflection/   Lightweight task reflection evaluator
  reports/      Artifact generation plus grounded facts, rendering, and report-contract validation
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
pip install -e ".[dev,api,documents]"
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

This smoke test intentionally creates a smolagents `CodeAgent` with no tools. The web application
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
  hosting. Model, conversation, document-analysis, and report routes are registered independently;
  business decisions remain in `AgentOrchestrator` and the existing services.
- PDF, DOCX, PNG, and JPEG inputs continue through MinerU when available, with the established local
  fallback behavior. All unstructured content is normalized to Markdown before downstream use.
- Document analysis inserts every normalized upload into the active conversation's private store at
  `storage/minirag/conversations/<conversation_id>/` and writes a conversation-namespaced mirror to
  the global aggregate. Chat exposes only the active conversation's semantic and graph tools by
  default. The browser receives only final answers, citations, safe process events, and generated
  report links.
- The Knowledge control enables the active conversation's private MiniRAG. Global is disabled by
  default and becomes selectable only after Knowledge is enabled; when explicitly selected for a
  turn, it additionally exposes the aggregate of uploads from every conversation under
  `storage/minirag/`. Switching or creating a conversation clears the Global permission, and deleting
  a conversation removes both its private index and its namespaced global mirror.
- Long documents are structured by heading and page, chunked with section provenance, and summarized
  hierarchically. The React document workspace can target a question, selected sections, or the whole
  document without sending the complete source to the model.
- Local folder mode accepts an absolute folder path, recursively displays supported files as a
  selectable tree, and analyzes only the selected originals. It skips hidden paths and symbolic
  links, allows up to 100 selected files per run, and uses in-memory document/Excel tools without
  copying those files into uploads, saved documents, or MiniRAG.
- Saved documents can be selected for direct re-analysis or explicitly attached to the active chat.
  Attachment indexes their normalized Markdown into that chat's private MiniRAG; merely seeing a
  document in the saved-document list does not grant chat access to it.
- The initial OpenAI-compatible endpoint is read from the existing environment configuration. The
  model settings dialog can switch the runtime address without modifying `.env`, and the current
  model identifier is resolved from the selected server for each run.
- Local MLX mode accepts an existing model directory, discovers `mlx_lm.server` from the model
  parent `.venv` or current environment, starts it on a free loopback port, and stops only that child
  process when the API exits. Arbitrary compatible models can still be used through Remote API mode.

Conversations are persisted in `storage/qwopus.db`, and existing Streamlit JSONL history is imported
once when that database is first created. FastAPI serves both the production React application and a
separate local-only React Debug Console; neither interface depends on Streamlit Session State.

Verify the new boundary with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest tests.test_model_runtime tests.test_api
MYPY_CACHE_DIR=/tmp/qwopus-mypy-cache mypy src/qwopus_agent
ruff check src tests
cd frontend && pnpm run build
```

## Local Debug Console

The read-only Debug Console is part of the FastAPI/React application. Chat, document analysis,
MiniRAG, MinerU, reports, and knowledge-graph workflows continue to use the same backend:

```bash
pip install -e ".[dev,api]"
python -m mlx_lm.server --model ~/Desktop/model/gemma-4-12B-it-qat-OptiQ-4bit --port 8080
PYTHONPATH=src python -m uvicorn qwopus_agent.api.app:app --host 127.0.0.1 --port 8010
```

Open the formal application at `http://127.0.0.1:8010/` and the Debug Console at
`http://127.0.0.1:8010/debug`.

To expose the application and read-only Debug Console to trusted devices on the same private LAN,
start it with an explicit diagnostics permission:

```bash
QWOPUS_DEBUG_ALLOW_LAN=1 \
PYTHONPATH=src python -m uvicorn qwopus_agent.api.app:app --host 0.0.0.0 --port 8010
```

Then open `http://<mac-lan-ip>:8010/debug`. LAN diagnostics are disabled by default, and the
backend still rejects non-private client addresses because traces may contain prompts, document
evidence, and Tool observations. Use this mode only on a trusted network.

The Debug Console provides:

- current model identity, endpoint, process, Python, platform, uptime, task counts, and trace storage
- persisted backend orchestration events, final status, final result, source/status filters, and search
- complete smolagents prompts, raw model outputs, Tool arguments, Tool Observations, and step errors
- downloadable per-run JSON traces and a bounded rotating runtime-log tail
- automatic five-second refresh and a link back to the formal React frontend
- no chat, upload, graph-maintenance, report, model-mutation, or record-deletion actions

The Debug Console intentionally exposes more information than the primary React application and may
contain full local document excerpts. Use it only on a trusted local machine. It displays the raw
fields returned by smolagents and the current model server; it cannot display hidden reasoning that
the model/provider did not return. FastAPI `RunView` and `AnalysisView` do not serialize raw debug
runs, so the primary React frontend continues to receive only final answers, citations, and safe
process events. FastAPI writes internal records atomically under `logs/debug_runs/`; the Debug API
only reads those files and does not initialize MiniRAG, MinerU, Torch, or an Agent worker. Raw
diagnostics reject non-loopback clients even if the main API is later exposed to the local network.
Debug retention keeps at most the newest 200 complete records, 64 MiB in aggregate, and 14 days of
history. Cleanup never touches an active `.tmp` write, so sensitive diagnostics stay bounded without
exposing partially written JSON.

Manual checks:

1. Send a message or analyze a document in the React application at `http://127.0.0.1:8010/`.
2. Open `http://127.0.0.1:8010/debug` and click "Refresh".
3. Confirm the matching backend run shows its safe trace, raw steps, Tool calls, and Observations.
4. Confirm the runtime summary shows the currently selected model server as online.

Runtime MiniRAG data lives under `storage/minirag`. New uploads are stored under
`storage/minirag/conversations/<conversation_id>/documents.jsonl`; a namespaced mirror is maintained
in the global aggregate at `storage/minirag/documents.jsonl`, alongside compatible legacy records.
That aggregate is never opened by a chat turn unless Global is explicitly authorized.
Markdown-normalized documents and safe spreadsheet summaries are inserted through the
`MiniRAG.insert(document)` facade, and later analysis can retrieve existing context through
`MiniRAG.search(query)`. Application Skills depend on the small `KnowledgeStore` protocol rather
than on vector or graph internals. The current `MiniRAG` class is a Qwopus adapter using the
upstream package's persistent NanoVectorDB component; it is not a wrapper around the upstream
project's complete `MiniRAG.query` pipeline. Qwopus owns chunking, conversation scoping, graph
extraction, and evidence rendering around that component. A local multilingual sentence-transformer
keeps retrieval independent from
the Gemma, Qwen, or other chat model currently served by the OpenAI-compatible endpoint. The installed
`minirag-hku` package is linked to `vendor/MiniRAG-main`; Qwopus adds a persistent directed graph layer
for evidence-bound entities, relations, cross-document aggregation, and bounded multi-hop paths.

When a known file is not returned in chat, check the scope before changing the similarity slider:

1. `Knowledge` searches only `storage/minirag/conversations/<conversation_id>/` by default.
2. Use **Saved documents → Attach to chat** to make an existing parsed document available to the
   active conversation.
3. Enable `Global` only when the current turn may use documents from other conversations or legacy
   global records.
4. Lower `Sources` only when the correct scope is selected but semantic evidence is still too weak.

MiniRAG treats `documents.jsonl` as the fact source and the NanoVectorDB file as rebuildable derived
state. On startup it compares chunk ids and automatically fills or rebuilds a stale vector index, so
missing derived vectors do not require re-uploading the original documents. Before each knowledge
operation, a cached instance also checks the fact-store version under the cross-process file lock;
API workers and Agent subprocesses therefore see documents written by another process without an
application restart.

MinerU parses each PDF, DOCX, PNG, or JPEG in its own directory under `storage/cache/mineru`. This
keeps concurrent uploads from selecting another task's generated Markdown while preserving every
output path in the debug metadata.

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
