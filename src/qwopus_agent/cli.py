"""Command-line entry point for the first Qwopus-Agent milestone."""

from __future__ import annotations

import argparse

from qwopus_agent.agent import AgentLoop, Executor, Planner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Qwopus-Agent skeleton.")
    parser.add_argument("objective", help="Objective for the agent loop to plan and execute.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = AgentLoop(planner=Planner(), executor=Executor())
    result = agent.run(args.objective)
    print(result.execution.output)


if __name__ == "__main__":
    main()
