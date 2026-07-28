"""Adapt reusable Qwopus Skills to smolagents Tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse
from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens

SkillRequestFactory = Callable[[Mapping[str, Any]], SkillRequest]


def build_skill_tool(
    skill: BaseSkill,
    *,
    inputs: Mapping[str, dict[str, Any]],
    request_factory: SkillRequestFactory | None = None,
    tool_name: str | None = None,
    description: str | None = None,
    max_output_tokens: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    start_phase: str = "executing",
    end_phase: str = "generating",
) -> Any:
    """Expose one BaseSkill through smolagents without duplicating its business logic."""
    try:
        from smolagents import Tool
    except ModuleNotFoundError as exc:
        raise RuntimeError("smolagents is required to build Agent tools.") from exc

    resolved_inputs = dict(inputs)
    input_names = tuple(resolved_inputs)
    resolved_name = tool_name or skill.name
    resolved_description = description or skill.description

    class SkillTool(Tool):  # type: ignore[misc]
        name = resolved_name
        description = resolved_description
        inputs = resolved_inputs
        output_type = "string"
        # 原因：不同 Skill 有不同的运行时 schema，静态 forward 签名无法逐项声明。
        # 作用：仍由 Tool.inputs 校验参数，再在这个唯一适配器中映射成 SkillRequest。
        skip_forward_signature_validation = True

        def forward(self, *args: Any, **kwargs: Any) -> str:
            if len(args) > len(input_names):
                raise TypeError(f"{resolved_name} received too many positional arguments.")
            values = dict(zip(input_names, args, strict=False))
            duplicate_names = set(values).intersection(kwargs)
            if duplicate_names:
                duplicate = ", ".join(sorted(duplicate_names))
                raise TypeError(f"{resolved_name} received duplicate arguments: {duplicate}.")
            values.update(kwargs)

            if progress_callback is not None:
                progress_callback(start_phase)
            request = (
                request_factory(values)
                if request_factory is not None
                else _default_request(values)
            )
            response = _run_skill(skill, request)
            if not response.success:
                raise RuntimeError(response.content)
            content = response.content
            if (
                max_output_tokens is not None
                and estimate_tokens(content) > max_output_tokens
            ):
                content = (
                    f"{truncate_to_tokens(content, max_output_tokens)}\n\n"
                    "[Skill output truncated by the Agent adapter.]"
                )
            if progress_callback is not None:
                progress_callback(end_phase)
            return content

    return SkillTool()


def _default_request(values: Mapping[str, Any]) -> SkillRequest:
    query = str(values.get("query", ""))
    arguments = {key: value for key, value in values.items() if key != "query"}
    return SkillRequest(query=query, arguments=arguments)


def _run_skill(skill: BaseSkill, request: SkillRequest) -> SkillResponse:
    """Run an async Skill from the synchronous smolagents Tool boundary."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(skill.run(request))

    # 原因：Tool 也可能被异步宿主直接调用，在已有事件循环中 asyncio.run 会失败。
    # 作用：只在该少见边界使用一个短线程，保持 BaseSkill 的异步公共契约不变。
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(skill.run(request))).result()
