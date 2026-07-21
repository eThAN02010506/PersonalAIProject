import sys
import types
import unittest
from unittest.mock import patch

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    build_chat_messages,
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


class SmolagentsRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeToolCallingAgent.queued_results = []
        self.previous_module = sys.modules.get("smolagents")
        fake_module = types.ModuleType("smolagents")
        fake_module.OpenAIModel = FakeOpenAIModel
        fake_module.CodeAgent = FakeCodeAgent
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
        self.assertEqual(model.kwargs["client_kwargs"], {"timeout": 120})
        self.assertNotIn("max_tokens", model.kwargs)

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
        ):
            run_agent_chat_turn(
                user_message="查一下米饭怎么做",
                history=[],
                settings=settings,
                enable_web_search=True,
            )

        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [fake_tool])
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
        ):
            run_agent_chat_turn(
                user_message="Company A 和 Company B 有什么关系？",
                history=[],
                settings=settings,
                enable_local_knowledge=True,
            )

        # 原因：Streamlit 开关只负责授权，工具选择必须继续由 smolagents 驱动。
        # 作用：证明本地知识开启后同时提供语义检索和图路径检索，并保留步数上限。
        self.assertEqual(FakeToolCallingAgent.last_instance.tools, fake_tools)
        self.assertIn("Use rag_search", FakeToolCallingAgent.last_instance.prompt)
        self.assertIn("Use graph_search", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 2)

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
            )

        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [web_tool, *local_tools])
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 3)

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
            )

        # 原因：真实模型曾正确检索图谱，却只返回一句没有来源的答案。
        # 作用：锁定短答案会基于既有 Observation 交给无工具 Agent 收敛，杜绝重复检索。
        self.assertIn("agent_notes.txt", result)
        self.assertEqual(FakeToolCallingAgent.last_instance.tools, [])
        self.assertIn("[agent_notes.txt]", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 2)

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
            )

        # 原因：统一编排需要引用与 Tool 审计信息，但不能取得模型的 Thought。
        # 作用：确保详细结果只保留非 final_answer Tool 的 Observation 和名称。
        self.assertEqual(result.tool_calls, ("graph_search", "final_answer"))
        self.assertEqual(
            result.observations,
            ("[ownership.pdf, page 4] Company A owns Company B",),
        )
        self.assertNotIn("private reasoning", "".join(result.observations))

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
        self.assertIn("Do not reduce the answer to a short bullet list", prompt)
        self.assertIn("Local knowledge access is disabled", prompt)

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
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "document_parser",
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
        self.assertEqual(result.tool_calls, ["document_parser"])
        self.assertTrue(any("document_parser" in step for step in result.debug_steps))

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
                        "name": "document_parser",
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
            user_question="总结",
            tools=[object()],
            settings=settings,
        )

        self.assertEqual(result.answer, "这是最终总结。")
        self.assertNotIn("Observation", result.answer)

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
                                    "name": "document_parser",
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
                                    "name": "document_parser",
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
            user_question="Summarize both files.",
            tools=[object()],
            settings=settings,
        )

        # 原因：只检查 Tool 名称会把“重复读取同一文件”误判为完成多文件分析。
        # 作用：验证补充轮明确读取遗漏文件后，runtime 才接受最终答案。
        self.assertEqual(result.answer, "Both files were summarized.")
        self.assertIn("second.txt", FakeToolCallingAgent.last_instance.prompt)
        self.assertEqual(FakeToolCallingAgent.last_instance.run_kwargs["max_steps"], 3)

    def test_file_analysis_agent_rejects_answer_without_file_inspection(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        FakeToolCallingAgent.queued_results = [
            types.SimpleNamespace(output="猜测答案", state="success", steps=[]),
            types.SimpleNamespace(output="仍然猜测", state="success", steps=[]),
        ]

        with self.assertRaisesRegex(RuntimeError, "required file tools"):
            run_smolagents_file_analysis_with_debug(
                file_names=["notes.txt"],
                spreadsheet_names=[],
                user_question="总结",
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
                        "tool_calls": [{"function": {"name": "excel_schema"}}],
                    }
                ],
            ),
            types.SimpleNamespace(
                output="本地计算结果：East 为 40，West 为 20。",
                state="success",
                steps=[
                    {
                        "step_number": 2,
                        "tool_calls": [{"function": {"name": "excel_analysis"}}],
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
        self.assertIn("East 为 40", result.answer)
        self.assertTrue(any("excel_analysis" in step for step in result.debug_steps))

    def test_format_file_analysis_prompt_explains_excel_tool_order(self) -> None:
        prompt = format_file_analysis_agent_prompt(
            file_names=["sales.xlsx"],
            spreadsheet_names=["sales.xlsx"],
            user_question="按地区汇总收入",
        )

        self.assertIn("excel_schema first", prompt)
        self.assertIn("excel_analysis", prompt)
        self.assertIn("按地区汇总收入", prompt)

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

    @patch("qwopus_agent.integrations.smolagents_runtime.urllib.request.urlopen")
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

    @patch("qwopus_agent.integrations.smolagents_runtime.urllib.request.urlopen")
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

    @patch("qwopus_agent.integrations.smolagents_runtime.urllib.request.urlopen")
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
