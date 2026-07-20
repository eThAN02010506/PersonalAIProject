# Qwopus-Agent Architecture

## Design Principles

- Model-agnostic: all model providers implement `BaseLLM`.
- Interface-first: agents, memory, reflection, and skills communicate through stable contracts.
- Local-first: the first concrete model adapter targets `mlx_lm.server` through its OpenAI-compatible API.
- Incremental: each capability is independently testable and composed at CLI/UI boundaries.

## First-Stage Modules

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

The same insert operation also feeds an evidence-constrained graph pipeline. `BaseLLM` extraction
accepts any OpenAI-compatible model, rejects facts whose quote or chunk id cannot be verified, and
passes valid mentions through entity normalization before persistence in a directed `MultiDiGraph`.
Search combines bounded graph paths with complementary vector chunks while deduplicating source
evidence. Document updates, deletion, and rebuilds are owned by `KnowledgeMaintenanceService`, so the
Agent-facing MiniRAG contract remains limited to `insert` and `search`.

### Skills

`SkillRegistry.discover()` scans `qwopus_agent.skills` and imports modules with `create_skill()`.
Planner selects skills; Executor runs them through the registry. `graph_search` exposes persistent
multi-hop paths and cross-document evidence without manual registration.

`SkillGrowthService` observes complete successful Agent runs. Repeated traces are converted into
declarative `WorkflowSpec` files, stripped of paths and credentials, integrity checked, assigned a
semantic version in `SkillCatalog`, and loaded into `SkillRegistry`. It never deploys model-generated
arbitrary Python.

### Reports

`ReportGenerator` is the unified report module. It writes Markdown, Excel, real PNG/SVG charts, and a
PDF artifact from one request.

### Reflection

`TaskReflectionEvaluator` provides structured quality observations and retry suggestions without
requiring another LLM call.

### Remaining Provider Work

- inject a production browser-automation provider into the existing Browser Skill contract

## Suggested Milestone Order

1. Stabilize `BaseLLM` and local MLX adapter.
2. Add agent loop observability and structured plan outputs.
3. Add a first real tool, such as a Python execution tool or file read tool.
4. Add memory interfaces and a local persistence backend.
5. Add reflection hooks.
6. Add skill loading and reuse.
7. Build the research agent on top of the stable primitives.
8. Add production web-search provider wiring.
9. Add report download integration in the UI.
