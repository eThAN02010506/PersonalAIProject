# Qwopus-Agent Architecture

## Design Principles

- Model-agnostic: all model providers implement `BaseLLM`.
- Interface-first: agent, tools, memory, reflection, and skills communicate through stable contracts.
- Local-first: the first concrete model adapter targets `mlx_lm.server` through its OpenAI-compatible API.
- Incremental: advanced memory, RAG, research, and multi-agent behavior are intentionally deferred.

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
- `Executor`: executes a `Plan`, optionally resolving named tools from `ToolRegistry`.
- `AgentLoop`: coordinates planning and execution.

### Tools

`BaseTool` defines structured tool execution. `ToolRegistry` keeps tool discovery out of the agent loop,
which makes Python tools, file tools, project analyzers, and research tools easy to add later.

### Future Modules

- `memory`: long-term memory and retrieval interfaces
- `reflection`: task critique and improvement loops
- `skills`: reusable skill definitions and loaders
- `reports`: structured research and task reports
- `prompts`: prompt assets and builders

## Suggested Milestone Order

1. Stabilize `BaseLLM` and local MLX adapter.
2. Add agent loop observability and structured plan outputs.
3. Add a first real tool, such as a Python execution tool or file read tool.
4. Add memory interfaces and a local persistence backend.
5. Add reflection hooks.
6. Add skill loading and reuse.
7. Build the research agent on top of the stable primitives.
