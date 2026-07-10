"""CLI smoke test for smolagents + local LLM connectivity."""

from __future__ import annotations

import argparse

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsDependencyError,
    SmolagentsModelSettings,
    run_smolagents_smoke_test,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the smoke-test CLI parser."""
    parser = argparse.ArgumentParser(description="Test smolagents with the configured local LLM.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="用一句中文回答：Qwopus-Agent 已经成功连接本地大模型了吗？",
    )
    return parser


def main() -> None:
    """Run the smolagents connectivity smoke test."""
    parser = build_parser()
    args = parser.parse_args()
    settings = SmolagentsModelSettings.from_env()

    try:
        print(run_smolagents_smoke_test(args.prompt, settings=settings))
    except SmolagentsDependencyError as exc:
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
