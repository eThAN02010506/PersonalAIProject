"""Adapt reusable Qwopus Skills to smolagents Tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from qwopus_agent.skills import SkillRegistry, WorkflowSkill, WorkflowSpec
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse
from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens

SkillRequestFactory = Callable[[Mapping[str, Any]], SkillRequest]
_RUNTIME_SKILL_ALIASES = {
    "browser_open": "browser",
    "tavily_search": "web_search",
    "rag_search": "rag_search",
    "graph_search": "graph_search",
}


def build_registered_skill_tools(
    registry: SkillRegistry,
    *,
    enabled_permissions: set[str] | None = None,
    existing_tool_names: set[str] | None = None,
    max_output_tokens: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Any]:
    """Adapt every Registry Skill authorized for the current Agent run."""
    permissions = enabled_permissions or {"always"}
    occupied_names = set(existing_tool_names or ())
    tools: list[Any] = []
    for skill_name in registry.list_names():
        skill = registry.get(skill_name)
        permission = skill.agent_tool_permission
        tool_name = skill.agent_tool_name or skill.name
        if permission not in permissions or tool_name in occupied_names:
            continue
        # 原因：Registry 自动发现若停在目录清单，新 Skill 仍需手工写 smolagents 装配代码。
        # 作用：统一把获准 Skill 转为 Tool；新增 query-only Skill 无需修改任何中央列表。
        tools.append(
            build_skill_tool(
                skill,
                inputs=skill.agent_tool_inputs,
                tool_name=tool_name,
                max_output_tokens=max_output_tokens,
                progress_callback=progress_callback,
            )
        )
        occupied_names.add(tool_name)
    return tools


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


@dataclass
class _RuntimeToolSkill(BaseSkill):
    """Adapt one already-authorized query Tool to the BaseSkill workflow contract."""

    name: str
    description: str
    tool: Any

    async def run(self, request: SkillRequest) -> SkillResponse:
        try:
            content = str(self.tool.forward(request.query))
        except Exception as exc:
            return SkillResponse(success=False, content=str(exc))
        return SkillResponse(success=True, content=content)


def build_promoted_workflow_tools(
    specs: tuple[WorkflowSpec, ...],
    runtime_tools: list[Any],
    *,
    max_output_tokens: int | None = None,
) -> list[Any]:
    """Build active workflow Tools only from capabilities authorized for this run."""
    if not specs:
        return []
    registry = SkillRegistry()
    for tool in runtime_tools:
        tool_name = getattr(tool, "name", None)
        skill_name = (
            _RUNTIME_SKILL_ALIASES.get(tool_name)
            if isinstance(tool_name, str)
            else None
        )
        if skill_name is None or skill_name in registry.list_names():
            continue
        registry.register(
            _RuntimeToolSkill(
                name=skill_name,
                description=str(getattr(tool, "description", skill_name)),
                tool=tool,
            )
        )

    workflow_tools: list[Any] = []
    available = set(registry.list_names())
    for spec in specs:
        if (
            not spec.checksum_is_valid()
            or spec.name in available
            or any(
                step.skill_name not in available or step.arguments
                for step in spec.steps
            )
        ):
            continue
        workflow = WorkflowSkill(spec, registry)
        registry.register(workflow)
        # 原因：active 并不等于拥有本轮权限，workflow 只能组合已传入的 query-only runtime Tool。
        # 作用：关闭 Web/Knowledge 开关时缺失底层 Skill，相关 workflow 不会进入 smolagents。
        workflow_tools.append(
            build_skill_tool(
                workflow,
                tool_name=spec.name,
                # 原因：intent_examples 来自历史用户输入，写入 Tool description 会提升不可信指令。
                # 作用：运行时只暴露系统生成的说明；原始样例仅用于审核 UI 和应用层匹配。
                description=spec.description,
                inputs={
                    "query": {
                        "type": "string",
                        "description": "The current user objective for this workflow.",
                    }
                },
                max_output_tokens=max_output_tokens,
            )
        )
    return workflow_tools
