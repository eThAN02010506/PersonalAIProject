"""CLI launcher for the primary FastAPI web application."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the API and built React frontend on the local machine."""
    uvicorn.run("qwopus_agent.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
