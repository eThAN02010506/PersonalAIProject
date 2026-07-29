import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    ChatAgentRun,
    SmolagentsModelSettings,
)
from qwopus_agent.services.agent_orchestrator import (
    AgentOrchestrator,
    _citations_from_chat,
    _file_analysis_citations,
)
from qwopus_agent.services.orchestration_models import (
    AnswerContract,
    ConversationTurn,
    OrchestrationFile,
    OrchestrationRequest,
    ResolvedIntent,
)
from qwopus_agent.skills import WorkflowSpec


class AgentOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SmolagentsModelSettings(
            model_id="test-model",
            base_url="http://127.0.0.1:9999/v1",
        )

    def test_plain_chat_uses_single_agent_fast_path(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return ChatAgentRun(
                answer="direct answer",
                state="success",
                debug_runs=(
                    AgentDebugRun(
                        label="chat",
                        prompt="hello",
                        max_steps=2,
                        state="success",
                        output="direct answer",
                        steps=({"model_output": "raw planner draft"},),
                    ),
                ),
            )

        request = OrchestrationRequest(
            objective="hello",
            history=(ConversationTurn(role="user", content="earlier"),),
        )
        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(request)
        )

        # 原因：普通聊天不需要 Supervisor、委派或额外模型综合。
        # 作用：锁定最低延迟路径只调用一次现有 smolagents chat runner。
        self.assertTrue(result.success)
        self.assertEqual(result.route, "single_agent")
        self.assertEqual(result.final_answer, "direct answer")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["history"], [{"role": "user", "content": "earlier"}])
        # 原因：Debug Console 依赖 Orchestrator 传递 raw run，而正式答案仍是独立字符串。
        # 作用：锁定单 Agent 快速路径不会丢失调试步骤或把草稿拼进 final_answer。
        self.assertEqual(result.debug_runs[0].steps[0]["model_output"], "raw planner draft")
        self.assertNotIn("raw planner draft", result.final_answer)

    def test_active_workflow_snapshot_reaches_smolagents_runtime(self) -> None:
        calls: list[dict[str, object]] = []
        workflow = WorkflowSpec(
            name="learned_web_search",
            version="0.1.0",
            description="Validated web research workflow.",
            intent_examples=("research the current topic",),
            steps=({"skill_name": "web_search"},),
            source_signature="signature",
        ).sealed()

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return ChatAgentRun(answer="current sourced answer", state="success")

        result = asyncio.run(
            AgentOrchestrator(
                self.settings,
                chat_runner=fake_chat,
                workflow_specs=(workflow,),
            ).run(
                OrchestrationRequest(
                    objective="research the current topic",
                    enable_web_search=True,
                )
            )
        )

        # 原因：Catalog 中 active 不代表子 Agent 已拥有该能力。
        # 作用：锁定 Orchestrator 把父进程选定的不可变版本传给真实 chat runtime。
        self.assertTrue(result.success)
        self.assertEqual(calls[0]["promoted_workflows"], (workflow,))

    def test_resolved_objective_drives_agent_while_original_controls_language(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return ChatAgentRun(answer="更完整的分析", state="success")

        intent = ResolvedIntent(
            original_request="再详细一点",
            operational_objective=(
                "Previous objective: 分析上下文管理方案\n"
                "Current instruction: 再详细一点"
            ),
            task_type="analyze",
            answer_contract=AnswerContract(
                task_type="analyze",
                required_facets=("findings", "evidence", "limitations"),
            ),
        )
        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="再详细一点",
                    resolved_intent=intent,
                )
            )
        )

        self.assertTrue(result.success)
        self.assertIn("分析上下文管理方案", calls[0]["user_message"])
        self.assertEqual(calls[0]["response_language_source"], "再详细一点")
        self.assertEqual(calls[0]["answer_contract"], intent.answer_contract)
        self.assertEqual(result.resolved_intent, intent)

    def test_web_and_local_knowledge_use_supervisor_and_merge_citations(self) -> None:
        calls: list[tuple[bool, bool, str]] = []

        def fake_chat(**kwargs):
            web = bool(kwargs["enable_web_search"])
            local = bool(kwargs["enable_local_knowledge"])
            self.assertEqual(kwargs["min_source_relevance"], 0.8)
            self.assertEqual(bool(kwargs["include_global_knowledge"]), local)
            question = str(kwargs["user_message"])
            calls.append((web, local, question))
            if web:
                return ChatAgentRun(
                    answer="current external fact",
                    tool_calls=("tavily_search", "final_answer"),
                    observations=("https://example.com/current",),
                    state="success",
                )
            if local:
                return ChatAgentRun(
                    answer="stored relationship",
                    tool_calls=("graph_search", "final_answer"),
                    observations=("[ownership.pdf, page 4] Company A owns Company B",),
                    state="success",
                )
            self.assertIn("current external fact", question)
            self.assertIn("stored relationship", question)
            return ChatAgentRun(answer="combined final answer", state="success")

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="Compare Company A ownership with the current web status.",
                    conversation_id="conversation-1",
                    enable_web_search=True,
                    enable_local_knowledge=True,
                    include_global_knowledge=True,
                    min_source_relevance=0.8,
                )
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.route, "multi_agent")
        self.assertIn("combined final answer", result.final_answer)
        self.assertIn("https://example.com/current", result.final_answer)
        self.assertIn("ownership.pdf, page 4", result.final_answer)
        self.assertEqual(
            [task.agent_name for task in result.multi_agent_run.delegation_plan.tasks],
            ["research_agent", "knowledge_agent", "synthesis_agent"],
        )
        self.assertTrue(any(event.tool == "tavily_search" for event in result.trace))
        self.assertTrue(any(event.tool == "graph_search" for event in result.trace))
        self.assertEqual(len(calls), 3)

    def test_final_answer_citations_exclude_unused_retrieval_candidates(self) -> None:
        run = ChatAgentRun(
            answer=(
                "The frontend uses React. "
                "[Source: README.md | Section: Project Layout]"
            ),
            observations=(
                "[Source: unrelated.md | Section: Other] http://127.0.0.1:8010/debug`.",
                "[Source: README.md | Section: Project Layout]",
            ),
        )

        citations = _citations_from_chat(run)

        # 原因：RAG Observation 是候选证据集，不等于最终回答真正采用的来源。
        # 作用：锁定正式答案只显示被 final answer 引用的文件，并去掉 Section 元数据。
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].kind, "local")
        self.assertEqual(citations[0].source, "README.md")

    def test_plain_filename_reference_excludes_observation_urls(self) -> None:
        run = ChatAgentRun(
            answer="According to README.md, the frontend uses React and the backend uses FastAPI.",
            observations=(
                "[Source: README.md | Section: Project Layout] "
                "Examples: http://127.0.0.1:8010/ and http://127.0.0.1:8010/debug",
                "[Source: unrelated.md | Section: Other]",
            ),
        )

        citations = _citations_from_chat(run)

        # 原因：本地知识回答可能只写文件名，Observation 还可能包含文档正文中的示例 URL。
        # 作用：证明 UI 只收到回答实际采用的 README，而不会显示无关 localhost 链接。
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].kind, "local")
        self.assertEqual(citations[0].source, "README.md")

    def test_multiple_filename_references_exclude_unused_candidate(self) -> None:
        run = ChatAgentRun(
            answer="Facts are supported by alpha.md and beta.md.",
            observations=(
                "[Source: alpha.md] First fact.",
                "[Source: beta.md] Second fact.",
                "[Source: decoy.md] Unused candidate.",
            ),
        )

        citations = _citations_from_chat(run)

        # 原因：跨文件检索会返回多个候选来源，但答案可能只综合其中两份。
        # 作用：锁定多文件引用按最终答案筛选、保持顺序，并排除未采用的候选文件。
        self.assertEqual(
            [citation.source for citation in citations],
            ["alpha.md", "beta.md"],
        )

    def test_file_analysis_citations_exclude_uploaded_but_unused_file(self) -> None:
        citations = _file_analysis_citations(
            "The date is in *alpha.md* and the budget is in *beta.md*.",
            ["alpha.md", "beta.md", "decoy.md"],
        )

        # 原因：文件分析路径以前把“成功上传”错误等同于“最终答案采用”。
        # 作用：锁定上传分析与知识对话使用相同的最终答案来源原则。
        self.assertEqual(
            [citation.source for citation in citations],
            ["alpha.md", "beta.md"],
        )

    def test_failed_evidence_is_not_misreported_as_success(self) -> None:
        def fake_chat(**kwargs):
            if kwargs["enable_web_search"]:
                raise TimeoutError("web unavailable")
            if kwargs["enable_local_knowledge"]:
                return ChatAgentRun(answer="local evidence", state="success")
            return ChatAgentRun(answer="answer from remaining local evidence", state="success")

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="Compare available evidence.",
                    conversation_id="conversation-1",
                    enable_web_search=True,
                    enable_local_knowledge=True,
                )
            )
        )

        # 原因：异常文本过去沿用 success=True，导致 synthesis 和整体请求伪装成成功。
        # 作用：保留已取得的本地证据供降级展示，但 terminal 未完成时状态必须为失败。
        self.assertFalse(result.success)
        self.assertIn("local evidence", result.final_answer)
        self.assertTrue(
            any(
                event.status == "warning" and event.agent == "research_agent"
                for event in result.trace
            )
        )
        runs = {item.task_id: item for item in result.multi_agent_run.runs}
        self.assertFalse(runs["research"].success)
        self.assertFalse(runs["synthesis"].success)

    def test_knowledge_only_no_evidence_preserves_refusal_and_failure(self) -> None:
        def fake_chat(**_kwargs):
            return ChatAgentRun(
                answer=(
                    "当前会话的本地知识中没有检索到足够的相关证据，"
                    "因此我没有使用常识补全答案。"
                ),
                tool_calls=("rag_search",),
                observations=("No relevant MiniRAG results.",),
                state="max_steps_error",
                success=False,
                error="No relevant local knowledge evidence was found for this request.",
            )

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="对比我上传的两篇文章",
                    conversation_id="conversation-1",
                    enable_local_knowledge=True,
                )
            )
        )

        # 原因：安全拒答是有内容的失败结果，不能因文本非空而被整体状态判为成功。
        # 作用：单 Agent 仲裁仍向用户展示明确说明，调用方同时收到 success=False。
        self.assertFalse(result.success)
        self.assertIn("没有检索到足够的相关证据", result.final_answer)
        self.assertTrue(
            any(
                event.status == "failed" and event.agent == "knowledge_agent"
                for event in result.trace
            )
        )

    def test_knowledge_connection_failure_returns_safe_localized_error(self) -> None:
        def failing_chat(**_kwargs):
            raise RuntimeError(
                "AgentGenerationError: Error while generating output: Connection error."
            )

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=failing_chat).run(
                OrchestrationRequest(
                    objective="总结知识库",
                    conversation_id="conversation-1",
                    enable_local_knowledge=True,
                )
            )
        )

        # 原因：运行期间断线无法通过启动前探测完全避免，底层 Agent 异常不能直接展示。
        # 作用：用户看到中文恢复建议，Debug trace 仍保留 AgentGenerationError 供排查。
        self.assertFalse(result.success)
        self.assertIn("模型服务连接已中断", result.final_answer)
        self.assertNotIn("knowledge_agent was unavailable", result.final_answer)
        self.assertNotIn("AgentGenerationError", result.final_answer)
        self.assertTrue(
            any(
                "AgentGenerationError" in event.message
                for event in result.trace
                if event.agent == "knowledge_agent"
            )
        )

    def test_document_report_flow_preserves_analysis_tables_and_artifact(self) -> None:
        local_result = AnalysisResult(
            markdown_summary="# Local summary",
            tables={"summary": pd.DataFrame([{"value": 1}])},
            metadata={"source_type": "text"},
            markdown_document="local content",
            llm_analysis="document answer",
        )

        def fake_analysis(**_kwargs):
            return SimpleNamespace(
                result=local_result,
                debug_steps=["local parse completed"],
                analyzed_file_names=["notes.txt"],
            )

        def fake_chat(**kwargs):
            self.assertIn("document answer", kwargs["user_message"])
            return ChatAgentRun(answer="synthesized document answer", state="success")

        with tempfile.TemporaryDirectory() as tmpdir:
            report = SimpleNamespace(
                artifacts=[SimpleNamespace(kind="markdown", path=Path(tmpdir) / "report.md")]
            )

            class FakeReportGenerator:
                def generate(self, **kwargs):
                    self.kwargs = kwargs
                    return report

            generator = FakeReportGenerator()
            result = asyncio.run(
                AgentOrchestrator(
                    self.settings,
                    minirag=object(),
                    chat_runner=fake_chat,
                    analysis_runner=fake_analysis,
                    report_generator=generator,
                ).run(
                    OrchestrationRequest(
                        objective="summarize notes",
                        uploaded_files=(
                            OrchestrationFile(name="notes.txt", content=b"notes"),
                        ),
                        generate_report=True,
                    )
                )
            )

        self.assertEqual(result.route, "multi_agent")
        self.assertIs(result.analysis_result, local_result)
        self.assertIs(result.report, report)
        self.assertIn("notes.txt", result.final_answer)
        self.assertEqual(list(generator.kwargs["tables"]), ["summary"])
        self.assertEqual(
            [task.agent_name for task in result.multi_agent_run.delegation_plan.tasks],
            ["document_agent", "synthesis_agent", "report_agent"],
        )


if __name__ == "__main__":
    unittest.main()
