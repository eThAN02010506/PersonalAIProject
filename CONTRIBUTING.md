# Contributing to Qwopus-Agent

Thank you for improving Qwopus-Agent. Keep changes small, model-independent, and aligned with
[docs/requirements.md](docs/requirements.md).

## Development setup

Use Python 3.11 and install the project from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e vendor/minirag
python -m pip install -e ".[dev,api,documents,browser]"
cd frontend && pnpm install --frozen-lockfile
```

Do not commit `.env`, credentials, model weights, uploaded documents, generated reports, runtime
logs, or local knowledge indexes.

## Before changing code

1. Identify the corresponding `FR-*` or use case in the requirements.
2. Confirm the responsible module in [docs/traceability.md](docs/traceability.md).
3. Add or update a failing test that expresses the requested behavior.
4. Keep business logic out of React components and FastAPI route handlers.
5. Add comments only when they explain why a non-obvious constraint exists and what it protects.

## Verification

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check src tests
MYPY_CACHE_DIR=/tmp/qwopus-mypy .venv/bin/mypy src/qwopus_agent
cd frontend && pnpm run lint && pnpm run build
```

Run the relevant P0 or external checks from
[docs/evaluation.md](docs/evaluation.md) when changing documents, retrieval, models, search, or
browser behavior.

## Pull requests

- Explain the user-visible reason for the change.
- List the requirements and modules affected.
- Report exact tests and real cases executed.
- Describe data, security, migration, and model compatibility effects.
- Keep unrelated formatting and refactors out of the same pull request.

The repository is currently source-visible but not open-source licensed. Contact
the project owner before submitting an external contribution or reusing source.
