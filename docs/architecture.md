# Qwopus-Agent Architecture

## Design Principles

- Model-agnostic: all model providers implement `BaseLLM`.
- Interface-first: agent, tools, memory, reflection, and skills communicate through stable contracts.
- Local-first: the first concrete model adapter targets `mlx_lm.server` through its OpenAI-compatible API.
- Incremental: memory, reports, reflection, and research start as small testable modules; browser
  automation and multi-agent behavior remain deferred.

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

### Tools

`BaseTool` defines structured tool execution. `ToolRegistry` keeps tool discovery out of the agent loop,
which makes Python tools, file tools, project analyzers, and research tools easy to add later.

### Memory

`MiniRAG` exposes only `insert(document)` and `search(query)`. The current backend persists JSONL
documents under `storage/minirag` and uses a simple local search fallback.

### Skills

`SkillRegistry.discover()` scans `qwopus_agent.skills` and imports modules with `create_skill()`.
Planner selects skills; Executor runs them through the registry.

### Reports

`ReportGenerator` is the unified report module. It writes Markdown, Excel, chart manifests, and a
minimal PDF artifact from one request.

### Reflection

`TaskReflectionEvaluator` provides structured quality observations and retry suggestions without
requiring another LLM call.

### Deferred Modules

- production web-search provider
- browser automation
- multi-agent collaboration
- semantic/vector MiniRAG backend
- advanced skill versioning and reuse

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
