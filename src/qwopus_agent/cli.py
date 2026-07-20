"""Command-line composition root for the production Agent architecture."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from qwopus_agent.agents import AgentRouter, AgentRun, Executor, Planner
from qwopus_agent.services.skill_growth_service import SkillGrowthService
from qwopus_agent.skills import SkillRegistry


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments without containing Agent business logic."""
    parser = argparse.ArgumentParser(description="Run Qwopus-Agent.")
    parser.add_argument("objective", help="Objective to plan and execute.")
    parser.add_argument("--skill", help="Optional exact Skill name to execute.")
    parser.add_argument("--file", type=Path, help="Optional input file path.")
    return parser


def build_router() -> AgentRouter:
    """Compose production Planner, Executor, Registry, and growth observer."""
    registry = SkillRegistry.discover()
    growth = SkillGrowthService(registry=registry)
    # 原因：CLI 是依赖装配入口，不应再维护第二套 AgentLoop/ToolRegistry 业务实现。
    # 作用：CLI、测试和未来 UI 可以共享同一 Planner → Executor → Skill 主链。
    return AgentRouter(
        planner=Planner(skill_registry=registry),
        executor=Executor(skill_registry=registry),
        observers=(growth,),
    )


async def run_objective(
    objective: str,
    *,
    skill_name: str | None = None,
    file_path: Path | None = None,
    router: AgentRouter | None = None,
) -> AgentRun:
    """Run one CLI objective through an injected or production Router."""
    context: dict[str, object] = {}
    if skill_name:
        context["skill_name"] = skill_name
    if file_path is not None:
        context["arguments"] = {"file_path": str(file_path)}
    return await (router or build_router()).run(objective, context=context)


def main() -> None:
    """Parse command-line input and print only the final execution content."""
    args = build_parser().parse_args()
    result = asyncio.run(
        run_objective(
            args.objective,
            skill_name=args.skill,
            file_path=args.file,
        )
    )
    print(result.execution.content)
    if not result.execution.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
