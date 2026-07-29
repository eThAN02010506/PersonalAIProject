# Qwopus-Agent Architecture

## Design Principles

- Model-agnostic: all model providers implement `BaseLLM`.
- Interface-first: agents, memory, reflection, and skills communicate through stable contracts.
- Local-first: the first concrete model adapter targets `mlx_lm.server` through its OpenAI-compatible API.
- Incremental: each capability is independently testable and composed at CLI/UI boundaries.

## Runtime Modules

### LLM

`BaseLLM` defines the model contract. `LLMConfig` and `LLMRegistry` create concrete adapters from
configuration, so Planner, Executor, Skills, and CLI/UI never depend on Gemma, Qwopus, Qwen, or any
other model family.

`OpenAICompatibleLLM` supports any local or remote runtime that exposes `/v1/chat/completions`.
`LocalMLXLLM` is only a preset for `mlx_lm.server`, not a model-specific dependency.

### Agent

The agent layer is split into:

- `Planner`: creates a `Plan` from a user objective.
- `Executor`: executes a `Plan` through `SkillRegistry`.
- `AgentRouter`: coordinates planning and execution.
- `ResearchAgent`: reuses `AgentRouter` and reflection for research-style tasks.
- `MultiAgentSupervisor`: delegates dependency-aware tasks, executes independent waves in parallel,
  shares state, runs result debate, and returns one arbitrated final answer.

### Memory

`MiniRAG` exposes only `insert(document)` and `search(query)`. Original Markdown documents are
persisted as JSONL under `storage/minirag`; MiniRAG's `NanoVectorDBStorage` persists multilingual
semantic embeddings in a separate derived index. Search results include file and page citations
when the normalized Markdown contains that metadata. The embedding model is local and independent
from the currently selected chat model.

The same insert operation also feeds an evidence-constrained graph pipeline. A deterministic
extractor handles explicit relations, while `LLMGraphExtractor` resolves the currently selected
`BaseLLM` at insertion time for ordinary document prose. Invalid chunk ids, unsupported quotes, and
relations whose evidence omits an endpoint are rejected before persistence in a directed
`MultiDiGraph`. Search combines bounded graph paths with complementary vector chunks while
deduplicating source evidence. Document updates, deletion, and rebuilds are owned by
`KnowledgeMaintenanceService`, so the Agent-facing MiniRAG contract remains limited to `insert` and
`search`.

### Analysis Safety

LLM-generated pandas is parsed through an allowlisted AST in both the parent and worker process. The
worker receives no application secrets, has CPU/file/descriptor limits, and returns inert JSON. On
macOS it additionally runs under Seatbelt with network, file-write, process-fork, and sensitive-path
reads denied. This boundary is intended for generated dataframe analysis expressions, not arbitrary
user Python.

### Web Boundary

Loopback requests remain frictionless for local development. Every direct non-loopback HTTP request
fails closed unless `QWOPUS_LAN_PASSWORD` is configured, then requires one shared HTTP Basic
credential across React, API, OpenAPI, and Debug surfaces. Debug routes also require the independent
`QWOPUS_DEBUG_ALLOW_LAN` gate. This is private-LAN protection, not per-user authorization or tenant
isolation; HTTPS is still required when transport confidentiality matters.

### Skills

`SkillRegistry.discover()` scans `qwopus_agent.skills` and imports modules with `create_skill()`.
Planner selects skills; Executor runs them through the registry. `graph_search` exposes persistent
multi-hop paths and cross-document evidence without manual registration.

`SkillGrowthService` observes complete successful Agent runs. Repeated traces are converted into
declarative `WorkflowSpec` files, stripped of paths and credentials, integrity checked, assigned a
semantic version in `SkillCatalog`, and loaded into `SkillRegistry`.

`SkillAuthoringService` lets an approved Debug Console user ask the current `BaseLLM` to compose a
candidate from explicitly allowed existing Skills. Pydantic rejects unknown fields, unapproved
capabilities, persistent arguments, and malformed output. The candidate remains outside the runtime
Registry until manual promotion; review exposes its exact spec, checksum checks, version diff, and a
side-effect-free dry run. Model-generated arbitrary Python is not accepted or deployed.

### Reports

`ReportGenerator` is the unified report module. It writes Markdown, Excel, real PNG/SVG charts, and a
PDF artifact from one request.

### Prompts

`qwopus_agent.prompts` owns model-facing task construction, response-depth rules, language policy,
and evidence requirements. smolagents integration code consumes these policies but does not own
them.

### Reflection

`TaskReflectionEvaluator` provides structured quality observations and retry suggestions without
requiring another LLM call.

## Current Boundaries

- Browser automation has a tested Skill and provider contract, but no production browser provider
  is wired into the application.
- The FastAPI and React application supports one shared LAN credential. Per-user authorization and
  multi-user tenancy are outside the current runtime.
- The MiniRAG adapter uses upstream persistent vector storage inside Qwopus-owned chunking,
  conversation scoping, graph extraction, and evidence rendering. It does not expose the complete
  upstream `MiniRAG.query` pipeline.
- Model-authored Skills remain declarative workflows and require manual promotion. Arbitrary
  generated Python is not accepted as deployable Skill code.
