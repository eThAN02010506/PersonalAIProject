# Qwopus-Agent

Qwopus-Agent is a local-first, modular AI Agent framework designed for Apple Silicon Mac
workflows and MLX-hosted OpenAI-compatible local models.

The project is intentionally interface-first. The first milestone focuses on clean software
architecture rather than advanced agent behavior:

- unified LLM abstraction through `BaseLLM`
- provider-neutral model creation through `LLMConfig` and `LLMRegistry`
- local MLX model adapter through `LocalMLXLLM`
- tool abstraction through `BaseTool`
- minimal Planner, Executor, and Agent Loop skeleton
- reserved module boundaries for memory, reflection, skills, reports, prompts, storage, and logs
- runnable tests for the core contracts

## Project Layout

```text
src/qwopus_agent/
  agent/        Planner, Executor, and AgentLoop skeleton
  llm/          BaseLLM interface and LocalMLXLLM adapter
  tools/        BaseTool interface and tool registry
  memory/       Future long-term memory interfaces
  skills/       Future reusable skill system
  reflection/   Future task reflection interfaces
  reports/      Future report generation module
  prompts/      Prompt templates and system prompt assets
tests/          Unit tests for first-stage architecture
storage/        Runtime data, ignored by Git except .gitkeep
logs/           Runtime logs, ignored by Git except .gitkeep
```

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

This smoke test intentionally creates a smolagents `CodeAgent` with no tools. Excel analysis,
document upload, MiniRAG ingestion, and report generation remain separate placeholders
until the model connection is verified.

## Streamlit Chat Test

After the smoke test passes, verify multi-turn conversation through Streamlit:

```bash
cp .env.example .env
pip install -e ".[dev,ui]"
python -m mlx_lm.server --model ~/Desktop/model/gemma-4-12B-it-qat-OptiQ-4bit --port 8080
streamlit run src/qwopus_agent/ui/streamlit_chat.py
```

The chat page provides:

- sidebar model configuration and connection check
- multi-turn chat via `st.chat_message` and `st.chat_input`
- conversation history passed into smolagents through `run_smolagents_chat_turn`

Manual checks:

1. Click "检测模型连接" and confirm the MLX server is online.
2. Send "你好，请用中文自我介绍" and confirm a response appears.
3. Ask "上一句你说了什么？" and confirm the reply uses prior context.
4. Stop the MLX server and confirm the UI shows a clear offline error.

## Milestone 1 Scope

This stage deliberately avoids complex long-term memory, RAG, browser automation, multi-agent
systems, and production research workflows. Those will be added after the core contracts are
stable.
