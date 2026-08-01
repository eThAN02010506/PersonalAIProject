import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    _build_answer_quality_checks,
    build_chat_messages,
    build_local_knowledge_tools,
    build_smolagents_code_agent,
    build_smolagents_model,
    build_smolagents_tool_calling_agent,
    check_model_connection,
    format_agent_chat_prompt,
    format_chat_prompt,
    format_file_analysis_agent_prompt,
    resolve_model_settings,
    run_agent_chat_turn,
    run_agent_chat_turn_with_debug,
    run_smolagents_chat_turn,
    run_smolagents_file_analysis_with_debug,
)
from qwopus_agent.integrations.smolagents_spreadsheets import remove_markdown_tables
from qwopus_agent.services.orchestration_models import AnswerContract, AnswerPlan
from qwopus_agent.skills import (
    BaseSkill,
    SkillRegistry,
    SkillRequest,
    SkillResponse,
    WorkflowSpec,
)


class FakeOpenAIModel:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = None
        self.calls = []
        self.call_kwargs = []
        FakeOpenAIModel.last_instance = self

    def generate(self, messages, **kwargs):
        self.messages = messages
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        return types.SimpleNamespace(content=f"reply: {messages[-1]['content']}")


class FakeCodeAgent:
    def __init__(self, tools, model, **kwargs):
        self.tools = tools
        self.model = model
        self.kwargs = kwargs

    def run(self, prompt):
        return f"ok: {prompt}"


class FakeTool:
    """Minimal smolagents Tool base used by request-scoped Skill adapters."""


class FakeToolCallingAgent:
    last_instance = None
    queued_results = []

    def __init__(self, tools, model, **kwargs):
        self.tools = tools
        self.model = model
        self.kwargs = kwargs
        self.prompt = None
        FakeToolCallingAgent.last_instance = self

    def run(self, prompt, **kwargs):
        self.prompt = prompt
        self.run_kwargs = kwargs
        if self.queued_results:
            return self.queued_results.pop(0)
        return f"agent reply: {prompt}"


def _grounded_report_prompt(section_five: str = "给出具体例子") -> str:
    return (
        "请逐一阅读所有文件并按以下结构完整输出：\n"
        "## 1. 文档理解与任务拆解\n"
        "## 2. 写作整体策略\n"
        "## 3. 详细写作框架（Outline）\n"
        "## 4. 逐段写作指导\n"
        f"## 5. {section_five}\n"
        "## 6. 生成完整报告 Draft\n"
        "## 7. Draft 后分析\n"
        "## 8. 最终检查 Checklist"
    )


def _grounded_collection_observation(
    manifest: tuple[str, ...] = (
        "method.pdf",
        "lesson-21.docx",
        "lesson-22.docx",
    ),
) -> str:
    return (
        "QWOPUS_SOURCE_COVERAGE="
        + json.dumps(
            list(manifest),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n"
        "QWOPUS_EXPLICIT_RUBRIC_FOUND=true\n\n"
        "# File: method.pdf\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Context and thought-flow\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=method]:\n"
        "Context changes what an observed detail contributes to an argument.\n\n"
        "# File: lesson-21.docx\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Lesson 21\n"
        "- topic_line: Title: Active humility\n"
        "- scripture_line: 经文：腓立比书2章8节\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=21-q]:\n"
        "材料不是把卑微解释为自我贬低，而是主动走向低处。\n"
        "APPLICATION_EVIDENCE [chunk_id=21-a]:\n"
        "请辨认一次表面让步、内里压抑的关系处境，并说明动机。\n\n"
        "# File: lesson-22.docx\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Lesson 22\n"
        "- topic_line: Title: Free obedience\n"
        "- scripture_line: 经文：腓立比书2章8节\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=22-q]:\n"
        "材料说明顺服不是盲从，而是理解之后仍然选择信靠和回应。\n"
        "APPLICATION_EVIDENCE [chunk_id=22-a]:\n"
        "请分析一次明知更正确却不愿行动的选择及其后果。"
    )


class SmolagentsRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeToolCallingAgent.queued_results = []
        FakeToolCallingAgent.last_instance = None
        self.previous_module = sys.modules.get("smolagents")
        fake_module = types.ModuleType("smolagents")
        fake_module.OpenAIModel = FakeOpenAIModel
        fake_module.CodeAgent = FakeCodeAgent
        fake_module.Tool = FakeTool
        fake_module.ToolCallingAgent = FakeToolCallingAgent
        sys.modules["smolagents"] = fake_module

    def tearDown(self) -> None:
        if self.previous_module is None:
            sys.modules.pop("smolagents", None)
        else:
            sys.modules["smolagents"] = self.previous_module

    def test_build_smolagents_model_uses_local_openai_compatible_settings(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="gemma-4-12B-it-qat-OptiQ-4bit",
            base_url="http://127.0.0.1:8080/v1",
            api_key="local_token",
            timeout_seconds=120,
            temperature=0.2,
            max_tokens=128,
        )

        model = build_smolagents_model(settings)

        self.assertEqual(model.kwargs["model_id"], "gemma-4-12B-it-qat-OptiQ-4bit")
        self.assertEqual(model.kwargs["api_base"], "http://127.0.0.1:8080/v1")
        self.assertEqual(model.kwargs["api_key"], "local_token")
        self.assertEqual(
            model.kwargs["client_kwargs"],
            {"timeout": 120, "max_retries": 1},
        )
        self.assertEqual(model.kwargs["max_tokens"], 128)

    def test_build_smolagents_code_agent_starts_without_tools(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        agent = build_smolagents_code_agent(settings=settings)

        self.assertEqual(agent.tools, [])
        self.assertEqual(agent.run("hello"), "ok: hello")

    def test_build_smolagents_tool_calling_agent_starts_with_tools(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        tools = [object()]

        agent = build_smolagents_tool_calling_agent(settings=settings, tools=tools)

        self.assertEqual(agent.tools, tools)
        self.assertIs(agent.model, FakeOpenAIModel.last_instance)

    def test_complex_answer_contract_uses_native_final_answer_check(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
            required_facets=("findings", "evidence", "limitations"),
        )
        checks = _build_answer_quality_checks(contract)

        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0]("结论可行。", None, agent=None))
        detailed = (
            "## 主要发现\n\n当前结构把规划、执行和持久化职责分开，"
            "因此组件可以独立测试，模型替换也不会改变业务接口。"
            "依赖注入让测试能够替换模型、知识库和外部搜索实现。\n\n"
            "## 证据与影响\n\n运行入口只依赖统一请求对象，后台任务使用版本化消息，"
            "失败状态不会覆盖最后一个成功任务。这个设计降低了跨进程状态漂移风险，"
            "也使 Debug 记录能够关联到真实请求。\n\n"
            "## 风险与限制\n\n仍需验证并发完成顺序、模型断线恢复和持久化迁移。"
            "对高风险工具还应增加权限校验与人工确认，避免文档中的指令改变工具行为。"
            "完成这些验证后，方案才适合进入稳定部署。"
            "例如，两个任务同时更新同一会话时，应以版本号拒绝旧写入，并用一次"
            "并发碰撞测试确认成功结果没有被迟到任务覆盖。模型断线测试还应分别检查"
            "连接错误、任务状态和恢复后的上下文，避免只验证页面是否显示错误提示。"
        )
        self.assertTrue(checks[0](detailed, None, agent=None))

        agent = build_smolagents_tool_calling_agent(
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            final_answer_checks=checks,
        )
        self.assertEqual(agent.kwargs["final_answer_checks"], checks)

    def test_run_agent_chat_turn_uses_smolagents_driver_without_web_tool(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        result = run_agent_chat_turn(
            user_message="你好",
            history=[],
            settings=settings,
            enable_web_search=False,
        )

        self.assertIn("agent reply:", result)
        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [])
        self.assertIn("Internet search is disabled", FakeToolCallingAgent.last_instance.prompt)

    def test_run_agent_chat_turn_receives_registry_discovered_skill(self) -> None:
        class RuntimeEchoSkill(BaseSkill):
            name = "runtime_echo"
            description = "Echo the current objective."

            async def run(self, request: SkillRequest) -> SkillResponse:
                return SkillResponse(success=True, content=request.query)

        registry = SkillRegistry()
        registry.register(RuntimeEchoSkill())
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        # 原因：适配器单测不能证明正式聊天入口实际使用自动发现结果。
        # 作用：锁定 Registry 是 smolagents 每轮普通扩展 Skill 的装配来源。
        with patch(
            "qwopus_agent.integrations.smolagents_runtime.SkillRegistry.discover",
            return_value=registry,
        ):
            run_agent_chat_turn(user_message="hello", history=[], settings=settings)

        self.assertEqual(
            [tool.name for tool in FakeToolCallingAgent.last_instance.tools],
            ["runtime_echo"],
        )

    def test_browser_tool_is_injected_only_with_explicit_permission(self) -> None:
        class RuntimeBrowserTool:
            name = "browser_open"

        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_browser_open_tool",
            return_value=RuntimeBrowserTool(),
        ):
            run_agent_chat_turn(
                user_message="Open https://example.com",
                history=[],
                settings=settings,
                enable_browser=True,
            )

        # 原因：Browser 不得因启用 Tavily 或普通聊天而隐式进入 Agent 工具列表。
        # 作用：锁定显式 enable_browser → browser_open 的单轮授权边界。
        self.assertEqual(
            [tool.name for tool in FakeToolCallingAgent.last_instance.tools],
            ["browser_open"],
        )
        self.assertIn("Use browser_open", FakeToolCallingAgent.last_instance.prompt)

    def test_promoted_workflow_is_injected_only_with_its_authorized_tool(self) -> None:
        class RuntimeSearchTool:
            name = "tavily_search"
            description = "Search current sources."

            def forward(self, query: str) -> str:
                return f"current evidence for {query}"

        workflow = WorkflowSpec(
            name="learned_web_search",
            version="0.1.0",
            description="Validated recurring web research.",
            intent_examples=("research the current topic",),
            steps=({"skill_name": "web_search"},),
            source_signature="signature",
        ).sealed()
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="A complete current answer based on the retrieved source.",
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {"function": {"name": "learned_web_search"}}
                        ],
                        "observations": "current evidence with https://example.com",
                    }
                ],
            )
        ]

        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_tavily_search_tool",
            return_value=RuntimeSearchTool(),
        ):
            result = run_agent_chat_turn_with_debug(
                user_message="research the current topic",
                history=[],
                settings=SmolagentsModelSettings(
                    model_id="any-model",
                    base_url="http://127.0.0.1:8080/v1",
                ),
                enable_web_search=True,
                promoted_workflows=(workflow,),
            )

        tool_names = [
            getattr(tool, "name", "")
            for tool in FakeToolCallingAgent.last_instance.tools
        ]
        # 原因：人工 promote 必须改变下一次实际运行，而不只是 Catalog 状态。
        # 作用：证明 smolagents 同时收到原始授权 Tool 和已校验的复用工作流。
        self.assertTrue(result.success)
        self.assertEqual(
            tool_names,
            ["tavily_search", "learned_web_search"],
        )

    def test_local_tool_factory_opens_global_store_only_when_authorized(self) -> None:
        class FakeMemory:
            instances = []

            def __init__(self, storage_path=None):
                self.storage_path = storage_path
                self.graph_index = object()
                self.instances.append(self)

        def rag_tool(memory, **kwargs):
            return kwargs.get("tool_name", "rag_search")

        def graph_tool(_index, **kwargs):
            return kwargs.get("tool_name", "graph_search")

        with (
            patch("qwopus_agent.memory.MiniRAG", FakeMemory),
            patch(
                "qwopus_agent.integrations.smolagents_tools.build_minirag_search_tool",
                side_effect=rag_tool,
            ),
            patch(
                "qwopus_agent.integrations.smolagents_tools.build_graph_search_tool",
                side_effect=graph_tool,
            ),
        ):
            private_tools = build_local_knowledge_tools(
                "conversation-1",
                knowledge_root=Path("private-root"),
            )
            global_tools = build_local_knowledge_tools(
                "conversation-2",
                knowledge_root=Path("private-root"),
                include_global_knowledge=True,
            )

        # 原因：Prompt 权限不足以隔离数据，未授权时全局 MiniRAG 对象本身就不应创建。
        # 作用：证明默认只装配两项私有工具，显式授权才额外装配两项全局工具。
        self.assertEqual(private_tools, ["rag_search", "graph_search"])
        self.assertEqual(
            global_tools,
            [
                "rag_search",
                "graph_search",
                "global_rag_search",
                "global_graph_search",
            ],
        )
        self.assertEqual(
            FakeMemory.instances[0].storage_path,
            Path("private-root/conversation-1/documents.jsonl"),
        )
        self.assertEqual(
            FakeMemory.instances[-1].storage_path,
            Path("documents.jsonl"),
        )

    def test_empty_private_store_promotes_authorized_global_tools_before_model_run(self) -> None:
        class FakeMemory:
            def __init__(self, storage_path=None):
                self.storage_path = Path(storage_path)
                self.graph_index = ("graph", self.storage_path)

            def list_sources(self):
                if "conversation-empty" in self.storage_path.as_posix():
                    return []
                return ["conversation:other/global-notes.md"]

        tool_bindings: list[tuple[str, Path]] = []

        def rag_tool(memory, **kwargs):
            name = kwargs.get("tool_name", "rag_search")
            tool_bindings.append((name, memory.storage_path))
            return name

        def graph_tool(index, **kwargs):
            name = kwargs.get("tool_name", "graph_search")
            tool_bindings.append((name, index[1]))
            return name

        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "全局资料 global-notes.md 提供了足够的项目证据，"
                    "当前回答只依据该已授权来源并保留引用。该资料说明了项目目标、"
                    "实施范围、关键责任和交付顺序，因此可以在不使用当前空私库的情况下"
                    "形成有来源约束的总结，同时不会把模型常识伪装成文件内容。"
                ),
                state="success",
                steps=[
                    {
                        "tool_calls": [{"function": {"name": "rag_search"}}],
                        "observations": "[global-notes.md] authorized global evidence",
                    }
                ],
            )
        ]
        with (
            patch("qwopus_agent.memory.MiniRAG", FakeMemory),
            patch(
                "qwopus_agent.integrations.smolagents_tools.build_minirag_search_tool",
                side_effect=rag_tool,
            ),
            patch(
                "qwopus_agent.integrations.smolagents_tools.build_graph_search_tool",
                side_effect=graph_tool,
            ),
        ):
            result = run_agent_chat_turn_with_debug(
                user_message="请根据我上传的所有文档总结项目",
                history=[],
                settings=SmolagentsModelSettings(
                    model_id="any-model",
                    base_url="http://127.0.0.1:8080/v1",
                ),
                enable_local_knowledge=True,
                include_global_knowledge=True,
                knowledge_scope="conversation-empty",
                knowledge_root=Path("private-root"),
            )

        # 原因：空私库曾让模型先反复调用必然零命中的本地 rag_search，再自行决定是否 fallback。
        # 作用：证明运行前已把授权全局库提升为标准工具，且没有安装空私库或重复 global_* 工具。
        self.assertTrue(result.success)
        self.assertEqual(
            FakeToolCallingAgent.last_instance.tools,
            ["rag_search", "graph_search"],
        )
        self.assertEqual(
            tool_bindings,
            [
                ("rag_search", Path("documents.jsonl")),
                ("graph_search", Path("documents.jsonl")),
            ],
        )
        self.assertIn(
            "deterministically bound to the global store",
            FakeToolCallingAgent.last_instance.prompt,
        )
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 2)

    def test_private_sources_keep_private_primary_and_global_secondary_tools(self) -> None:
        class FakeMemory:
            def __init__(self, storage_path=None):
                self.storage_path = Path(storage_path)
                self.graph_index = object()

            def list_sources(self):
                if "conversation-with-files" in self.storage_path.as_posix():
                    return ["private-notes.md"]
                return ["conversation:other/global-notes.md"]

        def rag_tool(_memory, **kwargs):
            return kwargs.get("tool_name", "rag_search")

        def graph_tool(_index, **kwargs):
            return kwargs.get("tool_name", "graph_search")

        with (
            patch("qwopus_agent.memory.MiniRAG", FakeMemory),
            patch(
                "qwopus_agent.integrations.smolagents_tools.build_minirag_search_tool",
                side_effect=rag_tool,
            ),
            patch(
                "qwopus_agent.integrations.smolagents_tools.build_graph_search_tool",
                side_effect=graph_tool,
            ),
        ):
            tools = build_local_knowledge_tools(
                "conversation-with-files",
                knowledge_root=Path("private-root"),
                include_global_knowledge=True,
            )

        self.assertEqual(
            tools,
            [
                "rag_search",
                "graph_search",
                "global_rag_search",
                "global_graph_search",
            ],
        )
        self.assertEqual(tools.primary_scope, "private")
        self.assertEqual(tools.private_sources, ("private-notes.md",))
        self.assertEqual(
            tools.global_sources,
            ("conversation:other/global-notes.md",),
        )

    def test_run_agent_chat_turn_injects_tavily_tool_when_web_enabled(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        fake_tool = object()

        # 原因：smolagents 是聊天驱动入口，联网搜索应作为 Tavily Tool 注入 Agent。
        # 作用：验证 Streamlit 不需要手动先搜索，Agent runtime 会持有工具。
        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_tavily_search_tool",
            return_value=fake_tool,
        ) as build_web_tool:
            run_agent_chat_turn(
                user_message="查一下米饭怎么做",
                history=[],
                settings=settings,
                enable_web_search=True,
                max_evidence_sources=17,
            )

        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [fake_tool])
        self.assertEqual(build_web_tool.call_args.kwargs["max_results"], 17)
        self.assertIn("Use tavily_search", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 2)

    def test_run_agent_chat_turn_injects_local_knowledge_tools_when_enabled(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        fake_tools = [object(), object()]

        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_local_knowledge_tools",
            return_value=fake_tools,
        ) as build_tools:
            run_agent_chat_turn(
                user_message="Company A 和 Company B 有什么关系？",
                history=[],
                settings=settings,
                enable_local_knowledge=True,
                knowledge_scope="conversation-1",
                max_evidence_sources=17,
            )

        # 原因：Streamlit 开关只负责授权，工具选择必须继续由 smolagents 驱动。
        # 作用：证明本地知识开启后同时提供语义检索和图路径检索，并保留步数上限。
        self.assertEqual(FakeToolCallingAgent.last_instance.tools, fake_tools)
        self.assertIn("Use rag_search", FakeToolCallingAgent.last_instance.prompt)
        self.assertIn("Use graph_search", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 2)
        # 原因：来源提示只能从当前用户问题提取，不能从历史或模型改写后的 query 猜测。
        # 作用：证明 runtime 把原始问题传给知识工具装配层。
        self.assertEqual(
            build_tools.call_args.kwargs["user_message"],
            "Company A 和 Company B 有什么关系？",
        )
        self.assertEqual(build_tools.call_args.kwargs["max_results"], 17)

    def test_run_agent_chat_turn_allows_both_web_and_local_tools(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        web_tool = object()
        local_tools = [object(), object()]

        with (
            patch(
                "qwopus_agent.integrations.smolagents_runtime.build_tavily_search_tool",
                return_value=web_tool,
            ),
            patch(
                "qwopus_agent.integrations.smolagents_runtime.build_local_knowledge_tools",
                return_value=local_tools,
            ),
        ):
            run_agent_chat_turn(
                user_message="Compare local project facts with current web information.",
                history=[],
                settings=settings,
                enable_web_search=True,
                enable_local_knowledge=True,
                knowledge_scope="conversation-1",
            )

        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [web_tool, *local_tools])
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 3)

    def test_global_knowledge_tools_require_explicit_turn_permission(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        scoped_and_global_tools = [object(), object(), object(), object()]

        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_local_knowledge_tools",
            return_value=scoped_and_global_tools,
        ) as build_tools:
            run_agent_chat_turn(
                user_message="Compare this chat with global project knowledge.",
                history=[],
                settings=settings,
                enable_local_knowledge=True,
                include_global_knowledge=True,
                knowledge_scope="conversation-1",
            )

        # 原因：Global 开关若只影响 Prompt，模型仍可能在未授权时拿到全局 Tool。
        # 作用：锁定授权位会进入 Tool 工厂，且增加一次可选检索所需的受限步数。
        self.assertTrue(build_tools.call_args.kwargs["include_global_knowledge"])
        self.assertEqual(
            FakeToolCallingAgent.last_instance.tools,
            scoped_and_global_tools,
        )
        self.assertIn(
            "explicitly allowed global knowledge",
            FakeToolCallingAgent.last_instance.prompt,
        )
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 3)

    def test_global_knowledge_is_rejected_without_private_knowledge_permission(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        with self.assertRaisesRegex(ValueError, "requires local knowledge"):
            run_agent_chat_turn(
                user_message="Search globally",
                history=[],
                settings=settings,
                include_global_knowledge=True,
                knowledge_scope="conversation-1",
            )

    def test_explicit_uploaded_document_request_with_knowledge_off_fails_before_model(
        self,
    ) -> None:
        result = run_agent_chat_turn_with_debug(
            user_message=(
                "我已经上传了一些文档，请你仔细阅读所有文件，"
                "并基于这些材料帮我完成写作任务。"
            ),
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            enable_local_knowledge=False,
            include_global_knowledge=False,
        )

        # 原因：仅靠 Prompt 提醒无法阻止无 Tool 模型根据历史或常识编造“文件分析”。
        # 作用：明确文档意图在没有附件或知识授权时于模型构造前失败。
        self.assertFalse(result.success)
        self.assertEqual(result.state, "preflight_rejected")
        self.assertIn("请开启 Knowledge", result.answer)
        self.assertIn("Document analysis", result.answer)
        self.assertIsNone(FakeToolCallingAgent.last_instance)
        self.assertEqual(result.debug_runs, ())

    def test_internal_ledger_stage_can_explicitly_bypass_document_preflight(self) -> None:
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    '{"agreements":["Ledger is usable"],"conflicts":[],'
                    '"unsupported_claims":[],"gaps":[],"resolution":"Use the ledger."}'
                ),
                state="success",
                steps=[],
            )
        ]

        result = run_agent_chat_turn_with_debug(
            user_message=(
                "Original request: review all uploaded documents.\n\n"
                "Independent evidence ledger: already supplied."
            ),
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            output_role="review",
            enforce_document_evidence=False,
        )

        # 原因：Review 已消费父阶段提供的 ledger，不应重新执行用户入口的附件可用性检查。
        # 作用：仅内部调用可显式跳过 preflight，默认用户入口测试仍锁定原有拒绝行为。
        self.assertTrue(result.success)
        self.assertNotEqual(result.state, "preflight_rejected")
        self.assertIn("Ledger is usable", result.answer)

    def test_run_agent_chat_turn_refines_short_local_knowledge_answer(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="Planner 路由到 work",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "tool_calls": [{"function": {"name": "graph_search"}}],
                        "observations": "[agent_notes.txt] Planner routes work.",
                    }
                ],
            ),
            types.SimpleNamespace(
                output=(
                    "Planner 通过 routes 关系把任务路由到 work。"
                    "该关系来自 agent_notes.txt 中的原文证据。"
                ),
                state="success",
                steps=[],
            ),
        ]

        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_local_knowledge_tools",
            return_value=[object(), object()],
        ):
            result = run_agent_chat_turn(
                user_message="说明 Planner 和 work 的关系并注明来源",
                history=[],
                settings=settings,
                enable_local_knowledge=True,
                knowledge_scope="conversation-1",
            )

        # 原因：真实模型曾正确检索图谱，却只返回一句没有来源的答案。
        # 作用：锁定短答案会基于既有 Observation 交给无工具 Agent 收敛，杜绝重复检索。
        self.assertIn("agent_notes.txt", result)
        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [])
        self.assertIn("[agent_notes.txt]", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 2)

    def test_knowledge_only_zero_hits_skip_open_ended_finalizer(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        local_tools = [object(), object()]
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="A plausible but unsupported answer from model knowledge.",
                state="max_steps_error",
                steps=[
                    {
                        "step_number": 1,
                        "tool_calls": [{"function": {"name": "rag_search"}}],
                        "observations": "No relevant MiniRAG results.",
                    }
                ],
            ),
            types.SimpleNamespace(
                output="This finalizer result must never be used.",
                state="success",
                steps=[],
            ),
        ]

        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_local_knowledge_tools",
            return_value=local_tools,
        ):
            result = run_agent_chat_turn_with_debug(
                user_message="对比我上传的两篇文章",
                history=[],
                settings=settings,
                enable_local_knowledge=True,
                knowledge_scope="conversation-1",
            )

        # 原因：真实问题在 MiniRAG 零命中后被无工具 finalizer 补成了泛化答案。
        # 作用：零证据直接产生可审计失败，且第二个伪造回答仍留在队列中证明未调用。
        self.assertFalse(result.success)
        self.assertIn("没有检索到足够的相关证据", result.answer)
        self.assertEqual(result.observations, ("No relevant MiniRAG results.",))
        self.assertEqual(len(FakeToolCallingAgent.queued_results), 1)
        self.assertEqual(FakeToolCallingAgent.last_instance.tools, local_tools)

    def test_detailed_chat_result_keeps_tool_metadata_without_thoughts(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="The relationship is supported by ownership.pdf.",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "model_output": "private reasoning",
                        "tool_calls": [{"function": {"name": "graph_search"}}],
                        "observations": "[ownership.pdf, page 4] Company A owns Company B",
                    },
                    {
                        "step_number": 2,
                        "tool_calls": [{"function": {"name": "final_answer"}}],
                        "observations": "The relationship is supported by ownership.pdf.",
                    },
                ],
            )
        ]

        with patch(
            "qwopus_agent.integrations.smolagents_runtime.build_local_knowledge_tools",
            return_value=[object(), object()],
        ):
            result = run_agent_chat_turn_with_debug(
                user_message="How are Company A and Company B related?",
                history=[],
                settings=settings,
                enable_local_knowledge=True,
                knowledge_scope="conversation-1",
            )

        # 原因：统一编排需要引用与 Tool 审计信息，但不能取得模型的 Thought。
        # 作用：确保详细结果只保留非 final_answer Tool 的 Observation 和名称。
        self.assertEqual(result.tool_calls, ("graph_search", "final_answer"))
        self.assertEqual(
            result.observations,
            ("[ownership.pdf, page 4] Company A owns Company B",),
        )
        self.assertNotIn("private reasoning", "".join(result.observations))
        # 原因：业务结果必须保持脱敏，但 Debug Console 需要定位模型为什么选择该 Tool。
        # 作用：锁定原始 model_output 只进入独立 debug_runs，不会污染最终答案或引用提取。
        self.assertEqual(result.debug_runs[0].label, "chat")
        self.assertEqual(
            result.debug_runs[0].steps[0]["model_output"],
            "private reasoning",
        )
        self.assertNotIn("private reasoning", result.answer)

    def test_format_agent_chat_prompt_keeps_recent_history(self) -> None:
        prompt = format_agent_chat_prompt(
            history=[{"role": "user", "content": "上一句"}],
            user_message="继续",
            enable_web_search=True,
        )

        self.assertIn("user: 上一句", prompt)
        self.assertIn("CURRENT USER QUESTION", prompt)
        self.assertIn("\n继续\n", prompt)
        self.assertIn("tavily_search", prompt)
        self.assertNotIn("800-1500 Chinese characters", prompt)
        self.assertIn("do not enforce a fixed minimum length", prompt)
        self.assertIn("call tavily_search only once", prompt)
        # 原因：历史对话语言不能覆盖当前搜索输入，且默认回答不应被压成短要点。
        # 作用：锁定多语言跟随与详细回答这两个 Prompt 行为约束。
        self.assertIn("Do not default to Chinese or English", prompt)
        self.assertIn("Do not compress the answer into a summary-like bullet list", prompt)
        self.assertIn("Local knowledge access is disabled", prompt)

    def test_format_agent_chat_prompt_applies_requested_detail_level(self) -> None:
        concise = format_agent_chat_prompt(
            history=[],
            user_message="Explain this",
            enable_web_search=False,
            response_detail="concise",
        )
        detailed = format_agent_chat_prompt(
            history=[],
            user_message="Explain this",
            enable_web_search=False,
            response_detail="detailed",
        )

        # 原因：只验证 API 接受参数不能证明模型实际收到不同的回答策略。
        # 作用：锁定简洁档与详细档生成不同 Prompt，且详细档不使用硬性字数。
        self.assertIn("Keep the final answer concise and direct", concise)
        self.assertIn("thorough, fully developed final answer", detailed)
        self.assertIn("Develop every central point", detailed)
        self.assertIn("concrete support or an example", detailed)
        self.assertIn("fixed length", detailed)

    def test_role_prompts_keep_evidence_and_review_out_of_final_writing(self) -> None:
        answer_plan = AnswerPlan(
            objective="Compare two implementations.",
            task_type="compare",
            response_detail="detailed",
            central_goal="Identify the material trade-offs.",
            required_sections=("direct answer", "trade-offs"),
            depth_questions=("When should each option be preferred?",),
        )

        evidence = format_agent_chat_prompt(
            history=[],
            user_message="Compare them",
            enable_web_search=True,
            output_role="evidence",
            answer_plan=answer_plan,
        )
        review = format_agent_chat_prompt(
            history=[],
            user_message="Review supplied evidence",
            enable_web_search=False,
            output_role="review",
            answer_plan=answer_plan,
        )

        # 原因：共享“完整最终答案”指令会让 Worker 先写多份散乱文章，再由综合器重复压缩。
        # 作用：锁定 Worker/Reviewer 只返回内部 JSON，用户语言与详细写作规则仅属于最终综合器。
        self.assertIn("evidence worker, not the final answer writer", evidence)
        self.assertIn('"facts"', evidence)
        self.assertIn('"plan_item_ids"', evidence)
        self.assertIn("ANSWER PLAN", evidence)
        self.assertNotIn("thorough, information-dense final answer", evidence)
        self.assertIn("evidence reviewer, not the final answer writer", review)
        self.assertIn('"unsupported_claims"', review)
        self.assertIn('"coverage"', review)
        self.assertIn("one coverage row for every ANSWER PLAN item", review)
        self.assertIn("without Tool-grounded sources as unsupported", review)
        self.assertNotIn("Now produce the complete final answer", review)

    def test_evidence_role_skips_user_facing_answer_quality_check(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    '{"facts":[{"claim":"Finding","support":"Evidence",'
                    '"sources":[],"confidence":"medium"}],"limitations":[]}'
                ),
                state="success",
                steps=[],
            )
        ]

        result = run_agent_chat_turn_with_debug(
            user_message="Analyze the evidence",
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            answer_contract=contract,
            output_role="evidence",
        )

        self.assertTrue(result.success)
        self.assertEqual(FakeToolCallingAgent.last_instance.kwargs["final_answer_checks"], [])
        self.assertIn("evidence worker", FakeToolCallingAgent.last_instance.prompt)

    def test_detailed_short_answer_gets_one_issue_aware_refinement(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
            required_facets=("findings", "risks", "actions"),
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="结论：方案可行。",
                state="success",
                steps=[],
            ),
            types.SimpleNamespace(
                output=(
                    "## 结论\n\n方案可行，但前提是职责边界、失败恢复和权限模型都被明确。"
                    "\n\n## 分析\n\n规划与执行分离让每个阶段可独立测试，结构化证据则减少"
                    "重复上下文。主要风险是额外模型调用、状态同步和来源可信度；应通过"
                    "有界步骤、失败降级和引用校验控制。\n\n## 行动\n\n先验证直接路径，"
                    "再测试证据审查和一次修正，最后用真实断线与多来源案例验收。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_agent_chat_turn_with_debug(
            user_message="详细分析这个方案",
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            answer_contract=contract,
            output_role="final",
        )

        # 原因：原生布尔 final_answer_check 会让弱模型看不到原因并重复同一句短答。
        # 作用：锁定首轮最多两步，随后只有一次包含具体问题和首稿的无工具修正。
        self.assertEqual(len(result.debug_runs), 2)
        self.assertEqual(result.debug_runs[0].max_steps, 2)
        self.assertIn("insufficient_depth", result.debug_runs[1].prompt)
        self.assertIn(
            "fully develop every relevant answer-plan item",
            result.debug_runs[1].prompt,
        )
        self.assertIn("Previous draft", result.debug_runs[1].prompt)
        self.assertIn("职责边界", result.answer)

    def test_unsourced_empirical_claim_gets_explicit_removal_instruction(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="研究报告显示任务成功率提升了 20%，因此方案很好。",
                state="success",
                steps=[],
            ),
            types.SimpleNamespace(
                output=(
                    "该方案的价值来自职责隔离，因为规划逻辑与执行副作用可以分别验证。"
                    "当前没有来源支持量化收益，应通过固定任务集、失败注入和延迟记录"
                    "建立本项目自己的基线。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_agent_chat_turn_with_debug(
            user_message="请详细分析这个方案",
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            answer_contract=contract,
            output_role="final",
        )

        self.assertEqual(len(result.debug_runs), 2)
        self.assertIn("unsupported_empirical_claims", result.debug_runs[1].prompt)
        self.assertIn("remove every percentage", result.debug_runs[1].prompt)
        self.assertNotIn("20%", result.answer)

    def test_retry_cannot_publish_fabricated_source_or_measurement(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="结论：这个方案可行。",
                state="success",
                steps=[],
            ),
            types.SimpleNamespace(
                output=(
                    "职责分离可以让规划逻辑和执行副作用分别测试。\n"
                    "研究报告显示成功率提升 20%。\n"
                    "根据 invented.md 第 2 页，该方案已完成验证。\n"
                    "主要风险是状态漂移，因此应通过失败注入验证恢复边界。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_agent_chat_turn_with_debug(
            user_message="请详细分析这个方案",
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            answer_contract=contract,
            output_role="final",
        )

        self.assertIn("职责分离", result.answer)
        self.assertIn("失败注入", result.answer)
        self.assertNotIn("20%", result.answer)
        self.assertNotIn("invented.md", result.answer)

    def test_retry_cannot_publish_fabricated_source_without_numbers(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="结论：这个方案可行。",
                state="success",
                steps=[],
            ),
            types.SimpleNamespace(
                output=(
                    "职责分离可以让规划逻辑和执行副作用分别测试。\n"
                    "根据 draft.txt 第 1 页，该方案已经完成验证。\n"
                    "主要风险是状态漂移，因此应通过失败注入验证恢复边界。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_agent_chat_turn_with_debug(
            user_message="请详细分析这个方案",
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            answer_contract=contract,
            output_role="final",
        )

        self.assertIn("职责分离", result.answer)
        self.assertIn("失败注入", result.answer)
        self.assertNotIn("draft.txt", result.answer)

    def test_retry_cannot_publish_unsourced_case_study_framing(self) -> None:
        contract = AnswerContract(
            task_type="analyze",
            complexity="complex",
            response_detail="detailed",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="结论：这个方案可行。",
                state="success",
                steps=[],
            ),
            types.SimpleNamespace(
                output=(
                    "职责分离可以让规划逻辑和执行副作用分别测试。\n"
                    "**证据**\n"
                    "- 经验案例：某 Agent 使用该架构后表现更好。\n"
                    "主要风险是状态漂移，因此应通过失败注入验证恢复边界。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_agent_chat_turn_with_debug(
            user_message="请详细分析这个方案",
            history=[],
            settings=SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            ),
            answer_contract=contract,
            output_role="final",
        )

        self.assertIn("职责分离", result.answer)
        self.assertIn("失败注入", result.answer)
        self.assertNotIn("证据", result.answer)
        self.assertNotIn("经验案例", result.answer)

    def test_format_agent_chat_prompt_separates_language_source_and_resolved_task(
        self,
    ) -> None:
        prompt = format_agent_chat_prompt(
            history=[],
            user_message=(
                "Previous objective: 分析上下文管理方案\n"
                "Current instruction: 再详细一点"
            ),
            response_language_source="再详细一点",
            enable_web_search=False,
            answer_contract=AnswerContract(
                task_type="analyze",
                required_facets=("findings", "evidence", "limitations"),
            ),
        )

        self.assertIn("CURRENT USER QUESTION", prompt)
        self.assertIn("RESOLVED TASK TO EXECUTE", prompt)
        self.assertIn("task type=analyze", prompt)
        self.assertIn("findings, evidence, limitations", prompt)

    def test_format_agent_chat_prompt_bounds_history_characters(self) -> None:
        prompt = format_agent_chat_prompt(
            history=[
                {"role": "user", "content": "old message"},
                {"role": "assistant", "content": "x" * 5000},
            ],
            user_message="current question",
            enable_web_search=False,
        )

        # 原因：最近一条长回答也可能让远程模型的首 token 延迟显著增加。
        # 作用：锁定 4000 字符预算，并确认更旧消息会在预算耗尽后被丢弃。
        self.assertIn("[truncated]", prompt)
        self.assertNotIn("old message", prompt)
        self.assertLess(len(prompt), 6000)

    def test_file_analysis_agent_requires_and_records_file_tool_call(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="完整文件总结",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_summary",
                                    "arguments": {"file_name": "notes.txt"},
                                }
                            }
                        ],
                    }
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["notes.txt"],
            spreadsheet_names=[],
            user_question="总结",
            tools=[object()],
            settings=settings,
        )

        # 原因：自然语言答案不能证明模型真的读取了上传文件。
        # 作用：断言 runtime 记录了文件 Tool 调用，并且只返回最终答案。
        self.assertEqual(result.answer, "完整文件总结")
        self.assertEqual(result.tool_calls, ["document_summary"])
        self.assertTrue(any("document_summary" in step for step in result.debug_steps))
        self.assertEqual(result.debug_runs[0].prompt.splitlines()[0], (
            "You are Qwopus-Agent's uploaded-file analysis agent."
        ))
        self.assertEqual(result.debug_runs[0].state, "success")
        self.assertEqual(result.debug_runs[0].steps[0]["step_number"], 1)

    def test_file_analysis_retries_broad_summary_without_document_summary(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="这是基于局部搜索的总结。",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "One matching paragraph.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {"file_name": "notes.txt"},
                                }
                            }
                        ],
                    }
                ],
            ),
            types.SimpleNamespace(
                output="这是基于全文摘要的最终总结。",
                state="success",
                steps=[
                    {
                        "step_number": 2,
                        "observations": "Hierarchical summary.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_summary",
                                    "arguments": {"file_name": "notes.txt"},
                                }
                            }
                        ],
                    }
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["notes.txt"],
            spreadsheet_names=[],
            user_question="总结这份文档",
            tools=[object()],
            settings=settings,
        )

        # 原因：弱模型会把 search 命中的局部片段当作全文总结。
        # 作用：证明文档级问题漏掉 document_summary 时会被 runtime 追补。
        self.assertEqual(result.answer, "这是基于全文摘要的最终总结。")
        self.assertEqual(result.tool_calls, ["document_search", "document_summary"])
        self.assertTrue(any("document_summary" in step for step in result.debug_steps))

    def test_file_analysis_agent_retries_raw_observation(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        tool_step = {
            "step_number": 1,
            "tool_calls": [
                {
                    "function": {
                        "name": "document_search",
                        "arguments": {"file_name": "notes.txt"},
                    }
                }
            ],
        }
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="Observation:\nDocument Analysis: raw",
                state="success",
                steps=[tool_step],
            ),
            types.SimpleNamespace(
                output="这是最终总结。",
                state="success",
                steps=[],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["notes.txt"],
            spreadsheet_names=[],
            user_question="这份文件提到什么？",
            tools=[object()],
            settings=settings,
        )

        self.assertEqual(result.answer, "这是最终总结。")
        self.assertNotIn("Observation", result.answer)

    def test_file_analysis_does_not_expose_final_generation_error_as_answer(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "[{'type': 'text', 'text': "
                    "'Error in generating final LLM output: Connection error.'}]"
                ),
                state="max_steps_error",
                steps=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {"file_name": "notes.txt"},
                                }
                            }
                        ]
                    }
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["notes.txt"],
            spreadsheet_names=[],
            user_question="总结",
            tools=[object()],
            settings=settings,
        )

        # 原因：smolagents 的最终生成错误使用普通文本形状，旧代码会把它显示为成功答案。
        # 作用：业务答案保持为空以触发 fail-closed，原错误仅留在 Debug Console 运行记录。
        self.assertEqual(result.answer, "")
        self.assertEqual(result.inspected_file_names, ("notes.txt",))
        self.assertIn("Connection error", result.debug_runs[0].output)

    def test_file_analysis_retries_when_requested_draft_is_only_a_placeholder(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        tool_step = {
            "step_number": 1,
            "tool_calls": [
                {
                    "function": {
                        "name": "document_search",
                        "arguments": {"file_name": "lesson.txt"},
                    }
                }
            ],
        }
        complete_draft = " ".join(f"evidence-{index}" for index in range(340))
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="## 1. 文档理解\n简单说明。\n\n## 2. 完整报告 Draft\n略。",
                state="success",
                steps=[tool_step],
            ),
            types.SimpleNamespace(
                output=(
                    "## 1. 文档理解\n"
                    "这里给出足够具体的文档理解、任务拆解、证据和逻辑说明。"
                    "每一个判断都明确连接回当前课程材料。\n\n"
                    f"## 2. 完整报告 Draft\n{complete_draft}"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["lesson.txt"],
            spreadsheet_names=[],
            user_question=(
                "请按以下结构输出：\n"
                "## 1. 文档理解\n"
                "## 2. 完整报告 Draft"
            ),
            tools=[object()],
            settings=settings,
        )

        self.assertIn("evidence-339", result.answer)
        self.assertIn("Never use placeholders", FakeToolCallingAgent.last_instance.prompt)

    def test_file_analysis_accepts_compact_chinese_section_with_nested_headings(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "## 1. 文档理解\n"
                    "材料主题明确，任务需要把核心论点连接到具体证据并解释原因。\n\n"
                    "## 7. Draft 后分析\n"
                    "结构合理；严格评分时需补强引文。\n\n"
                    "### 1. 为什么 opening 有效\n"
                    "它先建立背景。\n\n"
                    "### 2. 可能扣分处\n"
                    "证据格式仍需统一。\n\n"
                    "## 8. Post-draft analysis\n"
                    "The opening works, but source citations still need a consistent format "
                    "and one stricter counterargument."
                ),
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {"file_name": "lesson.txt"},
                                }
                            }
                        ]
                    }
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["lesson.txt"],
            spreadsheet_names=[],
            user_question=(
                "请按以下结构输出：\n"
                "## 1. 文档理解\n"
                "## 7. Draft 后分析\n"
                "## 8. Post-draft analysis"
            ),
            tools=[object()],
            settings=settings,
        )

        # 原因：第 7 节的简洁中文正文及其 ### 编号子标题曾被误判为顶层缺节。
        # 作用：锁定语言友好阈值和同级标题边界，避免无意义的整篇重写。
        self.assertIn("严格评分时需补强引文", result.answer)
        self.assertIn("one stricter counterargument", result.answer)
        self.assertEqual(len(result.debug_runs), 1)
        self.assertEqual(FakeToolCallingAgent.queued_results, [])

    def test_file_analysis_merges_only_deficient_sections(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        accepted_section = (
            "ACCEPTED-SECTION-MUST-STAY：材料核心概念、关键证据和任务边界"
            "已经逐项解释清楚，并说明了这些证据为什么支持结论。"
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    f"## 1. 文档理解\n{accepted_section}\n\n"
                    "## 7. Draft 后分析\n略。"
                ),
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {"file_name": "lesson.txt"},
                                }
                            }
                        ]
                    }
                ],
            ),
            types.SimpleNamespace(
                output=(
                    "以下是整份答案的新版本，但这段前言不应进入合并结果。\n\n"
                    "## 1. 文档理解\n模型试图覆盖已经验收的章节。\n\n"
                    "## 7. Draft 后分析\n"
                    "结构合理；严格评分时需补强引文，并统一引用格式。\n\n"
                    "## 1. 文档理解\n这段尾随覆盖也不应进入合并结果。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["lesson.txt"],
            spreadsheet_names=[],
            user_question=(
                "请按以下结构输出：\n"
                "## 1. 文档理解\n"
                "## 7. Draft 后分析"
            ),
            tools=[object()],
            settings=settings,
        )

        self.assertIn(accepted_section, result.answer)
        self.assertNotIn("模型试图覆盖", result.answer)
        self.assertNotIn("尾随覆盖", result.answer)
        self.assertNotIn("整份答案的新版本", result.answer)
        self.assertIn("统一引用格式", result.answer)
        self.assertEqual(
            result.debug_runs[1].label,
            "file_analysis_section_refinement",
        )
        self.assertIn("Return ONLY", result.debug_runs[1].prompt)
        self.assertIn("Do not rewrite", result.debug_runs[1].prompt)

    def test_file_analysis_section_refinement_remains_fail_closed(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        tool_step = {
            "tool_calls": [
                {
                    "function": {
                        "name": "document_search",
                        "arguments": {"file_name": "lesson.txt"},
                    }
                }
            ]
        }
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "## 1. 文档理解\n"
                    "材料核心概念、证据和任务边界已经解释清楚。\n\n"
                    "## 7. Draft 后分析\n略。"
                ),
                state="success",
                steps=[tool_step],
            ),
            types.SimpleNamespace(
                output="## 7. Draft 后分析\n待补充。",
                state="success",
                steps=[],
            ),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "did not complete requested report sections",
        ):
            run_smolagents_file_analysis_with_debug(
                file_names=["lesson.txt"],
                spreadsheet_names=[],
                user_question=(
                    "请按以下结构输出：\n"
                    "## 1. 文档理解\n"
                    "## 7. Draft 后分析"
                ),
                tools=[object()],
                settings=settings,
            )

    def test_file_analysis_repairs_exhaustive_draft_and_grounding_contract(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        file_names = [
            "bible-method.pdf",
            "腓立比书查经第二十一课.docx",
            "腓立比书查经第二十二课.docx",
        ]
        collection_observation = (
            'QWOPUS_SOURCE_COVERAGE=["bible-method.pdf",'
            '"腓立比书查经第二十一课.docx","腓立比书查经第二十二课.docx"]\n\n'
            "QWOPUS_EXPLICIT_RUBRIC_FOUND=false\n\n"
            "# File: bible-method.pdf\n"
            "SOURCE_FACTS:\n- document_heading: Bible study method\n\n"
            "# File: 腓立比书查经第二十一课.docx\n"
            "SOURCE_FACTS:\n"
            "- document_heading: 腓立比书查经第二十一课\n"
            "- scripture_line: 经文：腓立比书2章8节\n\n"
            "# File: 腓立比书查经第二十二课.docx\n"
            "SOURCE_FACTS:\n"
            "- document_heading: 腓立比书查经第二十二课\n"
            "- scripture_line: 经文：腓立比书2章9-11节"
        )
        collection_step = {
            "tool_calls": [
                {"function": {"name": "document_collection_summary"}}
            ],
            "observations": collection_observation,
        }
        draft_detail = (
            "本段依据对应文件解释经文、核心主题、具体例子与生活应用，"
            "并逐步说明为什么原文证据能够支持结论。"
        ) * 18
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "## 1. 文档理解与任务拆解\n"
                    "只总结腓立比书查经第二十一课，并把总分设为5分。"
                    "经文写成以弗所书2章8节（腓立比书2章8节）。\n\n"
                    "## 6. 生成完整报告 Draft\n"
                    "以下为示例：第二十一课内容。其余章节按同样格式展开。\n\n"
                    "## 7. Draft 后分析\n"
                    "开篇有背景，论证顺序清楚，满足5分标准。"
                ),
                state="success",
                steps=[collection_step],
            ),
            types.SimpleNamespace(
                output=(
                    "## 1. 文档理解与任务拆解\n"
                    "所有材料均已检查。材料没有提供显式 rubric，因此不虚构分值。\n\n"
                    "## 6. 生成完整报告 Draft\n"
                    "### 第21课\n"
                    f"经文：腓立比书2章8节\n{draft_detail}\n\n"
                    "### 第22课\n"
                    f"经文：腓立比书2章9-11节\n{draft_detail}\n\n"
                    "## 7. Draft 后分析\n"
                    "开篇先建立材料背景，再按经文、解释、证据和应用推进。"
                    "材料没有提供显式 rubric；严格评分时仍应检查逐项引证。"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=file_names,
            spreadsheet_names=[],
            user_question=(
                "请逐一阅读所有文件并按以下结构输出：\n"
                "## 1. 文档理解与任务拆解\n"
                "## 6. 生成完整报告 Draft\n"
                "## 7. Draft 后分析"
            ),
            tools=[types.SimpleNamespace(name="document_collection_summary")],
            settings=settings,
        )

        self.assertIn("bible-method", result.answer)
        self.assertIn("### 第二十一课", result.answer)
        self.assertIn("### 第二十二课", result.answer)
        self.assertNotIn("以弗所书", result.answer)
        self.assertNotIn("其余章节", result.answer)
        self.assertNotIn("5分标准", result.answer)
        self.assertEqual(
            result.debug_runs[1].label,
            "file_analysis_section_refinement",
        )
        self.assertIn("missing source labels", result.debug_runs[1].prompt)
        self.assertIn("QWOPUS_EXPLICIT_RUBRIC_FOUND", result.debug_runs[1].prompt)

    def test_file_analysis_rebuilds_unique_ordered_grounded_lesson_slots(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        file_names = [
            "腓立比书查经第二十三课.docx",
            "腓立比书查经第二十一课.docx",
            "腓立比书查经第二十二课.docx",
        ]
        collection_observation = (
            'QWOPUS_SOURCE_COVERAGE=["腓立比书查经第二十三课.docx",'
            '"腓立比书查经第二十一课.docx","腓立比书查经第二十二课.docx"]\n\n'
            "QWOPUS_EXPLICIT_RUBRIC_FOUND=false\n\n"
            "# File: 腓立比书查经第二十三课.docx\n"
            "SOURCE_FACTS:\n"
            "- document_heading: 腓立比书查经第二十三课\n"
            "- scripture_line: 经文：腓立比书2章9-11节\n"
            "QUERY_RELEVANT_EVIDENCE [chunk_id=23]:\n"
            "材料说明上帝高举基督，并把荣耀归给父上帝。\n\n"
            "# File: 腓立比书查经第二十一课.docx\n"
            "SOURCE_FACTS:\n"
            "- document_heading: 腓立比书查经第二十一课\n"
            "- scripture_line: 经文：腓立比书2章8节\n"
            "QUERY_RELEVANT_EVIDENCE [chunk_id=21]:\n"
            "材料围绕基督主动卑微、走向低处及其生活应用展开。\n\n"
            "# File: 腓立比书查经第二十二课.docx\n"
            "SOURCE_FACTS:\n"
            "- document_heading: 腓立比书查经第二十二课\n"
            "- scripture_line: 经文：腓立比书2章8节\n"
            "QUERY_RELEVANT_EVIDENCE [chunk_id=22]:\n"
            "材料从关系和自由的角度解释顺服，并提供讨论与应用问题。"
        )
        collection_step = {
            "tool_calls": [
                {"function": {"name": "document_collection_summary"}}
            ],
            "observations": collection_observation,
        }
        detail = (
            "本段逐步观察对应经文，解释材料主题为什么重要，"
            "并把证据连接到具体生活处境和可执行的回应。"
        ) * 4
        accepted_understanding = (
            "腓立比书查经第二十三课、腓立比书查经第二十一课、"
            "腓立比书查经第二十二课都已逐一说明。"
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    f"## 1. 文档理解\n{accepted_understanding}\n\n"
                    "## 6. 生成完整报告 Draft\n"
                    "### 第二十三课\n"
                    f"经文：腓立比书2章9-11节\n{detail}\n\n"
                    "### 第二十一课\n"
                    f"经文：腓立比书2章8节\n{detail}"
                ),
                state="success",
                steps=[collection_step],
            ),
            types.SimpleNamespace(
                output=(
                    "补写内容如下；模型没有遵守顶层 Section 标题格式。\n"
                    "### 第23课\n"
                    f"经文：腓立比书2章9-11节\n{detail}\n\n"
                    "### 第21课\n"
                    f"经文：以弗所书2章8节\n{detail}\n\n"
                    "### 第二十三课\n"
                    f"经文：腓立比书2章9-11节\n{detail}"
                ),
                state="success",
                steps=[],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=file_names,
            spreadsheet_names=[],
            user_question=(
                "请逐一阅读所有文件并按以下结构输出：\n"
                "## 1. 文档理解\n"
                "## 6. 生成完整报告 Draft"
            ),
            tools=[types.SimpleNamespace(name="document_collection_summary")],
            settings=settings,
        )

        self.assertIn(accepted_understanding, result.answer)
        self.assertEqual(result.answer.count("### 第二十一课"), 1)
        self.assertEqual(result.answer.count("### 第二十二课"), 1)
        self.assertEqual(result.answer.count("### 第二十三课"), 1)
        self.assertLess(
            result.answer.index("### 第二十一课"),
            result.answer.index("### 第二十二课"),
        )
        self.assertLess(
            result.answer.index("### 第二十二课"),
            result.answer.index("### 第二十三课"),
        )
        self.assertNotIn("以弗所书", result.answer)
        self.assertIn("材料从关系和自由的角度解释顺服", result.answer)
        self.assertEqual(FakeToolCallingAgent.queued_results, [])

    def test_file_analysis_agent_requires_each_uploaded_file_to_be_inspected(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="Only the first file was read.",
                state="max_steps_error",
                steps=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": '{"file_name": "first.txt"}',
                                }
                            }
                        ]
                    }
                ],
            ),
            types.SimpleNamespace(
                output="Both files were summarized.",
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {"file_name": "second.txt"},
                                }
                            }
                        ]
                    }
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["first.txt", "second.txt"],
            spreadsheet_names=[],
            user_question="Compare the setup details in both files.",
            tools=[object()],
            settings=settings,
        )

        # 原因：只检查 Tool 名称会把“重复读取同一文件”误判为完成多文件分析。
        # 作用：验证补充轮明确读取遗漏文件后，runtime 才接受最终答案。
        self.assertEqual(result.answer, "Both files were summarized.")
        self.assertEqual(result.inspected_file_names, ("first.txt", "second.txt"))
        self.assertIn("second.txt", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 3)

    def test_grounded_report_composer_uses_all_sources_without_model(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        file_names = ["method.pdf", "lesson-21.docx", "lesson-22.docx"]
        collection_tool = types.SimpleNamespace(
            name="document_collection_summary",
            forward=lambda: _grounded_collection_observation(),
        )

        result = run_smolagents_file_analysis_with_debug(
            file_names=file_names,
            spreadsheet_names=[],
            user_question=_grounded_report_prompt(),
            tools=[collection_tool],
            settings=settings,
        )

        self.assertIsNone(FakeToolCallingAgent.last_instance)
        self.assertEqual(result.generation_mode, "grounded_composer")
        self.assertEqual(result.tool_calls, ["document_collection_summary"])
        self.assertEqual(result.inspected_file_names, tuple(file_names))
        self.assertEqual(result.debug_runs[0].label, "grounded_report_composer")
        self.assertTrue(all(f"## {number}." in result.answer for number in range(1, 9)))
        draft = result.answer.split("## 6.", 1)[1].split("## 7.", 1)[0]
        self.assertEqual(draft.count("### lesson-21"), 1)
        self.assertEqual(draft.count("### lesson-22"), 1)
        self.assertLess(draft.index("### lesson-21"), draft.index("### lesson-22"))
        self.assertIn("### 引言", draft)
        self.assertIn("### 综合结论", draft)
        self.assertIn("表面让步、内里压抑", draft)
        self.assertIn("明知更正确却不愿行动", draft)
        self.assertNotIn("✅ 已对照材料中的显式 rubric", result.answer)
        self.assertIn("当前证据包未安全复述细则", result.answer)

    def test_grounded_report_composer_rejects_manifest_mismatch(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        collection_tool = types.SimpleNamespace(
            name="document_collection_summary",
            forward=lambda: _grounded_collection_observation(
                ("method.pdf", "lesson-21.docx"),
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "manifest does not exactly match",
        ):
            run_smolagents_file_analysis_with_debug(
                file_names=["method.pdf", "lesson-21.docx", "lesson-22.docx"],
                spreadsheet_names=[],
                user_question=_grounded_report_prompt(),
                tools=[collection_tool],
                settings=settings,
            )

        self.assertIsNone(FakeToolCallingAgent.last_instance)

    def test_grounded_report_composer_rejects_unknown_section_contract(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        collection_tool = types.SimpleNamespace(
            name="document_collection_summary",
            forward=lambda: _grounded_collection_observation(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Unsupported grounded report section: 5",
        ):
            run_smolagents_file_analysis_with_debug(
                file_names=["method.pdf", "lesson-21.docx", "lesson-22.docx"],
                spreadsheet_names=[],
                user_question=_grounded_report_prompt("自由发挥"),
                tools=[collection_tool],
                settings=settings,
            )

    def test_collection_summary_requires_a_complete_tool_coverage_manifest(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        collection_tool = types.SimpleNamespace(name="document_collection_summary")
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="All selected lessons were compared.",
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {"function": {"name": "document_collection_summary"}}
                        ],
                        "observations": (
                            'QWOPUS_SOURCE_COVERAGE=["lesson-21.docx","lesson-22.docx"]\n\n'
                            "# File: lesson-21.docx\nEvidence A\n\n"
                            "# File: lesson-22.docx\nEvidence B"
                        ),
                    }
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["lesson-21.docx", "lesson-22.docx"],
            spreadsheet_names=[],
            user_question="Read and compare every file.",
            tools=[collection_tool],
            settings=settings,
        )

        self.assertEqual(
            result.inspected_file_names,
            ("lesson-21.docx", "lesson-22.docx"),
        )
        self.assertIn("document_collection_summary", result.tool_calls)

    def test_specific_multi_file_question_does_not_require_collection_summary(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="Alpha and beta provide the requested facts.",
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {
                                        "file_name": "alpha.md",
                                        "query": "owner",
                                    },
                                }
                            }
                        ],
                        "observations": "[Source: alpha.md] Owner is Mira.",
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_search",
                                    "arguments": {
                                        "file_name": "beta.md",
                                        "query": "budget",
                                    },
                                }
                            }
                        ],
                        "observations": "[Source: beta.md] Budget is USD 4.2 million.",
                    },
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["alpha.md", "beta.md"],
            spreadsheet_names=[],
            user_question=(
                "State the owner and budget, cite the source file for each fact, "
                "and do not cite unrelated material."
            ),
            tools=[types.SimpleNamespace(name="document_collection_summary")],
            settings=settings,
        )

        # 原因：特定事实提取不需要先生成整个文件集合的平衡摘要。
        # 作用：证明两份文件均已检查时可直接完成，同时保留全文件任务的 coverage 规则。
        self.assertNotIn("document_collection_summary", result.tool_calls)
        self.assertEqual(result.inspected_file_names, ("alpha.md", "beta.md"))

    def test_collection_summary_tool_name_without_manifest_does_not_fake_coverage(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        collection_tool = types.SimpleNamespace(name="document_collection_summary")
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="Only a generic summary.",
                state="success",
                steps=[
                    {
                        "tool_calls": [
                            {"function": {"name": "document_collection_summary"}}
                        ],
                        "observations": "# File: lesson-21.docx\nOnly one source.",
                    }
                ],
            ),
            types.SimpleNamespace(
                output="Still generic.",
                state="success",
                steps=[],
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "did not inspect uploaded files"):
            run_smolagents_file_analysis_with_debug(
                file_names=["lesson-21.docx", "lesson-22.docx"],
                spreadsheet_names=[],
                user_question="Read every file.",
                tools=[collection_tool],
                settings=settings,
            )

    def test_file_analysis_agent_rejects_answer_without_file_inspection(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(output="猜测答案", state="success", steps=[]),
            types.SimpleNamespace(output="仍然猜测", state="success", steps=[]),
        ]

        with self.assertRaisesRegex(RuntimeError, "did not inspect uploaded files"):
            run_smolagents_file_analysis_with_debug(
                file_names=["notes.txt"],
                spreadsheet_names=[],
                user_question="这份文件提到什么？",
                tools=[object()],
                settings=settings,
            )

    def test_file_analysis_agent_forces_excel_sandbox_after_schema(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="根据样本猜测 East 为 40。",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "sales.xlsx"},
                                }
                            }
                        ],
                    }
                ],
            ),
            types.SimpleNamespace(
                output=(
                    "本地计算结果：\n\n"
                    "| region | revenue |\n"
                    "| --- | --- |\n"
                    "| East | 40 |\n"
                    "| West | 20 |"
                ),
                state="success",
                steps=[
                    {
                        "step_number": 2,
                        "observations": (
                            "| region | revenue |\n"
                            "| --- | --- | --- |\n"
                            "| East | 40 |\n"
                            "| West | 20 |"
                        ),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_analysis",
                                    "arguments": {"file_name": "sales.xlsx"},
                                }
                            }
                        ],
                    }
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["sales.xlsx"],
            spreadsheet_names=["sales.xlsx"],
            user_question="按地区汇总收入",
            tools=[object(), object()],
            settings=settings,
        )

        # 原因：schema sample 可能刚好包含全部小表，模型会跳过真实 pandas 计算。
        # 作用：验证 runtime 必须补调 excel_analysis 后才接受最终答案。
        self.assertEqual(result.tool_calls, ["excel_schema", "excel_analysis"])
        self.assertIn("| East | 40 |", result.answer)
        self.assertIn("| region | revenue |", result.answer)
        self.assertTrue(any("excel_analysis" in step for step in result.debug_steps))

    def test_file_analysis_accepts_reviewed_statistics_skill_after_schema(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "IQR 异常值：\n\n"
                    "| student | metric | upper_bound |\n"
                    "| --- | --- | --- |\n"
                    "| E | 100 | 16 |"
                ),
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "scores.xlsx"},
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 2,
                        "observations": (
                            "| student | metric | upper_bound |\n"
                            "| --- | --- | --- |\n"
                            "| E | 100 | 16 |"
                        ),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "scores.xlsx",
                                        "method": "iqr_outliers",
                                    },
                                }
                            }
                        ],
                    },
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["scores.xlsx"],
            spreadsheet_names=["scores.xlsx"],
            user_question="outlier 是什么",
            tools=[object(), object(), object()],
            settings=settings,
        )

        # 原因：常规统计不应为了满足完成条件再执行一次任意 pandas 代码。
        # 作用：锁定 schema + reviewed statistics 是完整且可接受的 Excel 工具链。
        self.assertEqual(result.tool_calls, ["excel_schema", "excel_statistics"])
        self.assertIn("| E | 100 | 16 |", result.answer)

    def test_file_analysis_accepts_modeling_skill_after_schema(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "回归结果：\n\n"
                    "| term | estimate | p_value |\n"
                    "| --- | --- | --- |\n"
                    "| x | 1.97 | 0.0001 |"
                ),
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "model.xlsx"},
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 2,
                        "observations": (
                            "| term | estimate | p_value |\n"
                            "| --- | --- | --- |\n"
                            "| x | 1.97 | 0.0001 |"
                        ),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_modeling",
                                    "arguments": {
                                        "file_name": "model.xlsx",
                                        "method": "linear_regression",
                                    },
                                }
                            }
                        ],
                    },
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["model.xlsx"],
            spreadsheet_names=["model.xlsx"],
            user_question="做 y 对 x 的回归",
            tools=[object(), object(), object(), object()],
            settings=settings,
        )

        # 原因：回归已经由审核 Skill 完成，不应再强制执行任意 pandas 代码。
        # 作用：锁定 schema + excel_modeling 是合法完整的 Excel 工具链。
        self.assertEqual(result.tool_calls, ["excel_schema", "excel_modeling"])
        self.assertIn("| x | 1.97 | 0.0001 |", result.answer)

    def test_file_analysis_routes_abstract_spreadsheet_statistics_for_weak_models(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="我看到了表结构，但没有计算。",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "iris.xlsx"},
                                }
                            }
                        ],
                    },
                ],
            ),
            types.SimpleNamespace(
                output=(
                    "正态性检验：\n\n"
                    "| column | p_value | decision_at_0.05 |\n"
                    "| --- | --- | --- |\n"
                    "| Sepal.Length | 0.056824 | do not reject normality |"
                ),
                state="success",
                steps=[
                    {
                        "step_number": 2,
                        "observations": (
                            "| column | p_value | decision_at_0.05 |\n"
                            "| --- | --- | --- |\n"
                            "| Sepal.Length | 0.056824 | do not reject normality |"
                        ),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "iris.xlsx",
                                        "method": "normality_test",
                                    },
                                }
                            }
                        ],
                    },
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["iris.xlsx"],
            spreadsheet_names=["iris.xlsx"],
            user_question="这个数据的分布正态吗？",
            tools=[object(), object(), object()],
            settings=settings,
        )

        # 原因：弱模型可能只看 schema 就作答，抽象统计问题必须被 runtime 拉回本地计算。
        # 作用：锁定“正态/分布”等问题会补调确定性 normality_test，而不是接受空泛回答。
        self.assertEqual(result.tool_calls, ["excel_schema", "excel_statistics"])
        self.assertIn("| Sepal.Length | 0.056824 |", result.answer)
        self.assertTrue(any("excel_statistics" in step for step in result.debug_steps))

    def test_file_analysis_routes_broad_spreadsheet_diagnostics_for_weak_models(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="我看到了表结构，数据大概正常。",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "scores.xlsx"},
                                }
                            }
                        ],
                    },
                ],
            ),
            types.SimpleNamespace(
                output="数据质量、概况、分位数和异常值已经本地计算。",
                state="success",
                steps=[
                    {
                        "step_number": 2,
                        "observations": "| column | missing |\n| --- | --- |\n| score | 0 |",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "scores.xlsx",
                                        "method": "missing",
                                    },
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 3,
                        "observations": "| column | mean |\n| --- | --- |\n| score | 75 |",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "scores.xlsx",
                                        "method": "describe",
                                    },
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 4,
                        "observations": "| column | p50 |\n| --- | --- |\n| score | 74 |",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "scores.xlsx",
                                        "method": "quantiles",
                                    },
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 5,
                        "observations": "| column | outlier_count |\n| --- | --- |\n| score | 1 |",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "scores.xlsx",
                                        "method": "iqr_outliers",
                                    },
                                }
                            }
                        ],
                    },
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["scores.xlsx"],
            spreadsheet_names=["scores.xlsx"],
            user_question="这个数据有什么问题？",
            tools=[object(), object(), object()],
            settings=settings,
        )

        # 原因：泛化诊断问题需要多项本地统计证据，否则弱模型会只给表面评价。
        # 作用：锁定 runtime 会补齐缺失、概况、分位数和异常值四类检查。
        self.assertEqual(result.tool_calls, ["excel_schema", "excel_statistics"])
        self.assertIn("| column | missing |", result.answer)
        self.assertIn("| column | mean |", result.answer)
        self.assertIn("| column | p50 |", result.answer)
        self.assertIn("| column | outlier_count |", result.answer)
        self.assertIn("IQR 异常值检查发现 1 个候选离群点", result.answer)

    def test_file_analysis_explains_empty_outlier_result_in_narrative(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="异常值检查完成。",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "iris.xlsx"},
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 2,
                        "observations": (
                            "## Statistical result: iqr_outliers\n"
                            "- outlier_count: 0\n\n"
                            "| metric | rule | outlier_count |\n"
                            "| --- | --- | --- |\n"
                            "| row_mean | 1.5 x IQR | 0 |"
                        ),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "iris.xlsx",
                                        "method": "iqr_outliers",
                                    },
                                }
                            }
                        ],
                    },
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["iris.xlsx"],
            spreadsheet_names=["iris.xlsx"],
            user_question="outlier 是什么？",
            tools=[object(), object(), object()],
            settings=settings,
        )

        # 原因：模型可能只说“已检查”，却漏掉 0 个异常值这个关键解释。
        # 作用：保证本地统计结果在自然语言正文和核验表中都可见。
        self.assertIn("未发现离群点", result.answer)
        self.assertIn("| metric | rule | outlier_count |", result.answer)

    def test_file_analysis_routes_single_spreadsheet_item_lookup_for_weak_models(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="我看到了表结构，但没有查具体项。",
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "character.xlsx"},
                                }
                            }
                        ],
                    },
                ],
            ),
            types.SimpleNamespace(
                output="STR 的值如下。",
                state="success",
                steps=[
                    {
                        "step_number": 2,
                        "observations": (
                            "| row_index | key | value |\n"
                            "| --- | --- | --- |\n"
                            "| 3 | STR | 40 |"
                        ),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_statistics",
                                    "arguments": {
                                        "file_name": "character.xlsx",
                                        "table_name": "Sheet1_key_values",
                                        "method": "lookup",
                                        "lookup_value": "STR",
                                    },
                                }
                            }
                        ],
                    },
                ],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["character.xlsx"],
            spreadsheet_names=["character.xlsx"],
            user_question="STR 是多少？",
            tools=[object(), object(), object()],
            settings=settings,
        )

        # 原因：单项问题需要命中原始行或 key-value 派生表，不能只给整表概况。
        # 作用：锁定 runtime 会强制补调 lookup，并保留本地命中的结果表。
        self.assertEqual(result.tool_calls, ["excel_schema", "excel_statistics"])
        self.assertIn("| 3 | STR | 40 |", result.answer)
        self.assertIn("excel_statistics.lookup", FakeToolCallingAgent.last_instance.prompt)
        self.assertIn("lookup_value", FakeToolCallingAgent.last_instance.prompt)

    def test_file_analysis_rejects_lookup_value_not_requested_by_user(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        wrong_lookup_steps = [
            {
                "step_number": 1,
                "observations": "Schema result",
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_schema",
                            "arguments": {"file_name": "character.xlsx"},
                        }
                    }
                ],
            },
            {
                "step_number": 2,
                "observations": (
                    "| row_index | STR | 40 |\n"
                    "| --- | --- | --- |\n"
                    "| 10 | CON | 55 |"
                ),
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_statistics",
                            "arguments": {
                                "file_name": "character.xlsx",
                                "table_name": "COC Character::table_2",
                                "method": "lookup",
                                "lookup_value": "CON",
                            },
                        }
                    }
                ],
            },
        ]
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="STR 是 55。",
                state="success",
                steps=wrong_lookup_steps,
            ),
            types.SimpleNamespace(
                output="STR 是 55。",
                state="success",
                steps=wrong_lookup_steps,
            ),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "did not call required file tools: excel_statistics",
        ):
            run_smolagents_file_analysis_with_debug(
                file_names=["character.xlsx"],
                spreadsheet_names=["character.xlsx"],
                user_question="STR 是多少？",
                tools=[object(), object(), object()],
                settings=settings,
            )

    def test_file_analysis_falls_back_when_lookup_value_is_wrong_but_path_is_authorized(
        self,
    ) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        wrong_lookup_steps = [
            {
                "step_number": 1,
                "observations": "Schema result",
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_schema",
                            "arguments": {"file_name": "character.xlsx"},
                        }
                    }
                ],
            },
            {
                "step_number": 2,
                "observations": (
                    "| row_index | STR | 40 |\n"
                    "| --- | --- | --- |\n"
                    "| 10 | CON | 55 |"
                ),
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_statistics",
                            "arguments": {
                                "file_name": "character.xlsx",
                                "table_name": "Sheet1",
                                "method": "lookup",
                                "lookup_value": "CON",
                            },
                        }
                    }
                ],
            },
        ]
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="STR 是 55。",
                state="success",
                steps=wrong_lookup_steps,
            ),
            types.SimpleNamespace(
                output="STR 是 55。",
                state="success",
                steps=wrong_lookup_steps,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "character.xlsx"
            pd.DataFrame({"STR": ["CON"], "40": [55]}).to_excel(path, index=False)

            result = run_smolagents_file_analysis_with_debug(
                file_names=["character.xlsx"],
                spreadsheet_names=["character.xlsx"],
                user_question="STR 是多少？",
                tools=[object(), object(), object()],
                settings=settings,
                spreadsheet_paths={"character.xlsx": path},
            )

        # 原因：真实 UI 中弱模型可能两轮都错把样例值 CON 当查询目标。
        # 作用：已授权路径内的本地 fallback 能纠正为 STR | 40，而不是把错误答案返回用户。
        self.assertIn("| STR | 40 |", result.answer)
        self.assertIn("本地 lookup 兜底完成", "\n".join(result.debug_steps))

    def test_file_analysis_rejects_failed_regression_even_with_model_table(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        failed_steps = [
            {
                "step_number": 1,
                "observations": "Schema result",
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_schema",
                            "arguments": {"file_name": "model.xlsx"},
                        }
                    }
                ],
            },
            {
                "step_number": 2,
                "error": "SVD did not converge",
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_modeling",
                            "arguments": {
                                "file_name": "model.xlsx",
                                "method": "linear_regression",
                            },
                        }
                    }
                ],
            },
        ]
        fabricated_answer = (
            "| term | estimate |\n"
            "| --- | --- |\n"
            "| x | 999 |"
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=fabricated_answer,
                state="success",
                steps=failed_steps,
            ),
            types.SimpleNamespace(
                output=fabricated_answer,
                state="success",
                steps=failed_steps,
            ),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "did not call required file tools: excel_modeling",
        ):
            run_smolagents_file_analysis_with_debug(
                file_names=["model.xlsx"],
                spreadsheet_names=["model.xlsx"],
                user_question="做 summary(lm()) 回归",
                tools=[object(), object()],
                settings=settings,
            )

    def test_format_file_analysis_prompt_explains_excel_tool_order(self) -> None:
        prompt = format_file_analysis_agent_prompt(
            file_names=["sales.xlsx"],
            spreadsheet_names=["sales.xlsx"],
            user_question="按地区汇总收入",
        )

        self.assertIn("excel_schema first", prompt)
        self.assertIn("excel_statistics", prompt)
        self.assertIn("excel_analysis", prompt)
        self.assertIn("already loaded in dfs", prompt)
        self.assertIn("Do not import, read files", prompt)
        self.assertIn("Markdown table", prompt)
        self.assertIn("count, mean, standard deviation", prompt)
        self.assertIn("按地区汇总收入", prompt)

    def test_format_file_analysis_prompt_includes_spreadsheet_intent_decomposition(self) -> None:
        prompt = format_file_analysis_agent_prompt(
            file_names=["character.xlsx"],
            spreadsheet_names=["character.xlsx"],
            user_question="STR 是多少？",
        )

        self.assertIn("Spreadsheet intent decomposition", prompt)
        self.assertIn("excel_statistics.lookup", prompt)
        self.assertIn("lookup_value", prompt)
        self.assertIn("exact item label", prompt)

    def test_format_file_analysis_prompt_treats_mean_as_required_statistics(self) -> None:
        prompt = format_file_analysis_agent_prompt(
            file_names=["iris.xlsx"],
            spreadsheet_names=["iris.xlsx"],
            user_question="Calculate the mean Sepal.Length and return a table.",
        )

        # 原因：均值问题如果只要求 schema，弱模型会直接猜表格并跳过本地统计 Tool。
        # 作用：锁定 mean/average 这类计算请求首轮就要求 excel_statistics.describe。
        self.assertIn("excel_statistics.describe", prompt)
        self.assertIn("do not call final_answer until excel_schema", prompt)
        self.assertNotIn("document-level understanding", prompt)

    def test_remove_markdown_tables_removes_malformed_model_pipe_tables(self) -> None:
        cleaned = remove_markdown_tables(
            "The mean is 5.84.\n\n"
            "| column | mean |\n"
            "| Sepal.Length | 5.843333 |\n\n"
            "Use the local table below."
        )

        # 原因：本地模型会生成没有 delimiter 的伪表格，直接展示会和核验表重复。
        # 作用：清理模型自写表格块，最终只附加本地 Tool 的规范 Markdown 表格。
        self.assertIn("The mean is 5.84.", cleaned)
        self.assertIn("Use the local table below.", cleaned)
        self.assertNotIn("| Sepal.Length |", cleaned)

    def test_file_analysis_preserves_computed_table_when_model_omits_it(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        computed_table = (
            "| region | average_revenue |\n"
            "| --- | --- |\n"
            "| East | 20.0 |\n"
            "| West | 20.0 |"
        )
        tool_steps = [
            {
                "step_number": 1,
                "observations": "Schema result",
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_schema",
                            "arguments": {"file_name": "sales.xlsx"},
                        }
                    }
                ],
            },
            {
                "step_number": 2,
                "tool_calls": [
                    {
                        "function": {
                            "name": "excel_statistics",
                            "arguments": {
                                "file_name": "sales.xlsx",
                                "method": "describe",
                            },
                        }
                    }
                ],
                "observations": computed_table,
            },
        ]
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output="East and West have the same average.",
                state="success",
                steps=tool_steps,
            ),
            types.SimpleNamespace(
                output="East and West have the same average.",
                state="success",
                steps=[],
            ),
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["sales.xlsx"],
            spreadsheet_names=["sales.xlsx"],
            user_question="Compare average revenue by region.",
            tools=[object(), object()],
            settings=settings,
        )

        self.assertIn("## Local calculation table", result.answer)
        self.assertIn(computed_table, result.answer)

    def test_file_analysis_replaces_model_table_with_verified_tool_table(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        verified_table = (
            "| group | ci_upper |\n"
            "| --- | --- |\n"
            "| virginica | 6.768715 |"
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "The groups differ.\n\n"
                    "| group | ci_upper |\n"
                    "| --- | --- |\n"
                    "| virginica | 6.589 |"
                ),
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "iris.xlsx"},
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 2,
                        "observations": verified_table,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_modeling",
                                    "arguments": {
                                        "file_name": "iris.xlsx",
                                        "method": "one_way_anova",
                                    },
                                }
                            }
                        ],
                    },
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["iris.xlsx"],
            spreadsheet_names=["iris.xlsx"],
            user_question="Run ANOVA.",
            tools=[object(), object()],
            settings=settings,
        )

        # 原因：模型可能在最终排版时改动一个数字，肉眼仍很难发现。
        # 作用：最终答案只保留受信任 Tool 表，错误的模型重抄表不会进入 UI。
        self.assertNotIn("6.589", result.answer)
        self.assertIn("6.768715", result.answer)
        self.assertIn("Tukey HSD assumes", result.answer)

    def test_file_analysis_removes_invalid_unequal_variance_tukey_claim(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        verified_table = (
            "| source | f_statistic |\n"
            "| --- | --- |\n"
            "| between_groups | 119.264502 |"
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(
                output=(
                    "方差不齐，但使用了不等方差的 Tukey HSD。\n\n"
                    "ANOVA 显示组间存在差异。\n\n"
                    "可考虑 Welch‑Tukey 或 Games-Howell。"
                ),
                state="success",
                steps=[
                    {
                        "step_number": 1,
                        "observations": "Schema result",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_schema",
                                    "arguments": {"file_name": "iris.xlsx"},
                                }
                            }
                        ],
                    },
                    {
                        "step_number": 2,
                        "observations": verified_table,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "excel_modeling",
                                    "arguments": {
                                        "file_name": "iris.xlsx",
                                        "method": "one_way_anova",
                                    },
                                }
                            }
                        ],
                    },
                ],
            )
        ]

        result = run_smolagents_file_analysis_with_debug(
            file_names=["iris.xlsx"],
            spreadsheet_names=["iris.xlsx"],
            user_question="进行方差分析",
            tools=[object(), object()],
            settings=settings,
        )

        # 原因：Tukey HSD 没有“不等方差版本”，弱模型可能将其与 Games-Howell 混淆。
        # 作用：删除错误方法声明，并用不可变限制说明保留正确的解释边界。
        self.assertNotIn("不等方差的 Tukey", result.answer)
        self.assertNotIn("Welch‑Tukey", result.answer)
        self.assertIn("本次未计算 Games-Howell", result.answer)

    def test_file_analysis_prompt_applies_requested_detail_level(self) -> None:
        concise = format_file_analysis_agent_prompt(
            file_names=["lesson.pdf"],
            spreadsheet_names=[],
            user_question="总结这份课程",
            response_detail="concise",
        )
        detailed = format_file_analysis_agent_prompt(
            file_names=["lesson.pdf"],
            spreadsheet_names=[],
            user_question="总结这份课程",
            response_detail="detailed",
        )

        # 原因：文档分析曾忽略聊天页的 Detailed 选择，导致相同设置产生概述式短答。
        # 作用：证明文件 Agent 与聊天 Agent 使用同一套详略语义。
        self.assertIn("Keep the final answer concise and direct", concise)
        self.assertIn("Develop every central point", detailed)
        self.assertNotEqual(concise, detailed)

    def test_format_file_analysis_prompt_routes_full_summary_to_hierarchy(self) -> None:
        prompt = format_file_analysis_agent_prompt(
            file_names=["manual.pdf"],
            spreadsheet_names=[],
            user_question="",
            analysis_mode="full",
        )

        # 原因：全文模式不能通过扩大 prompt 把完整大文档直接交给模型。
        # 作用：锁定 Agent 先使用分层摘要 Tool、再按需检索证据的调用策略。
        self.assertIn("document_summary first", prompt)
        self.assertIn("Summarize the uploaded files.", prompt)

    def test_format_file_analysis_prompt_routes_broad_question_to_summary(self) -> None:
        prompt = format_file_analysis_agent_prompt(
            file_names=["manual.pdf"],
            spreadsheet_names=[],
            user_question="请总结这份文档的整体观点",
            analysis_mode="question",
        )

        # 原因：用户不一定切到全文模式，也会用自然语言提出整体总结。
        # 作用：锁定 question 模式下的文档级请求先走分层摘要。
        self.assertIn("document-level understanding", prompt)
        self.assertIn("Call document_summary first", prompt)

    def test_format_chat_prompt_includes_history_and_latest_user_message(self) -> None:
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是本地助手。"},
        ]

        prompt = format_chat_prompt(history=history, user_message="上一句你说了什么？")

        self.assertIn("用户：你好", prompt)
        self.assertIn("助手：你好，我是本地助手。", prompt)
        self.assertIn("用户：上一句你说了什么？", prompt)
        self.assertTrue(prompt.endswith("助手："))

    def test_run_smolagents_chat_turn_uses_formatted_prompt(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        history = [{"role": "user", "content": "你好"}]

        result = run_smolagents_chat_turn(
            user_message="请继续",
            history=history,
            settings=settings,
        )

        self.assertEqual(result, "reply: 请继续")

    def test_build_chat_messages_uses_plain_chat_roles(self) -> None:
        history = [{"role": "user", "content": "你好"}]

        messages = build_chat_messages(history=history, user_message="请继续")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("无法自动访问用户之前上传的文件", messages[0]["content"])
        self.assertIn("不要编造文件内容", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "你好"})
        self.assertEqual(messages[2], {"role": "user", "content": "请继续"})

    @patch("qwopus_agent.integrations.smolagents_model.urllib.request.urlopen")
    def test_check_model_connection_reports_online(self, mock_urlopen) -> None:
        mock_response = mock_urlopen.return_value.__enter__.return_value
        mock_response.status = 200
        mock_response.read.return_value = b'{"data": [{"id": "live-model.gguf"}]}'

        online, message = check_model_connection(
            SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            )
        )

        self.assertTrue(online)
        self.assertIn("模型服务在线", message)
        self.assertIn("live-model.gguf", message)

    @patch("qwopus_agent.integrations.smolagents_model.urllib.request.urlopen")
    def test_resolve_model_settings_uses_live_server_model(self, mock_urlopen) -> None:
        mock_response = mock_urlopen.return_value.__enter__.return_value
        mock_response.status = 200
        mock_response.read.return_value = (
            b'{"data": [{"id": "C:\\\\models\\\\current-model.gguf"}]}'
        )
        settings = SmolagentsModelSettings(
            model_id="stale-model.gguf",
            base_url="http://127.0.0.1:8080/v1",
        )

        # 原因：.env 模型名可能与服务器刚切换的模型不同。
        # 作用：证明解析结果使用服务器实时 id，并保留原连接配置。
        resolved = resolve_model_settings(settings)

        self.assertEqual(resolved.model_id, "C:\\models\\current-model.gguf")
        self.assertEqual(resolved.base_url, settings.base_url)

    @patch("qwopus_agent.integrations.smolagents_model.urllib.request.urlopen")
    def test_check_model_connection_reports_offline(self, mock_urlopen) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        online, message = check_model_connection(
            SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            )
        )

        self.assertFalse(online)
        self.assertIn("无法连接模型服务", message)
