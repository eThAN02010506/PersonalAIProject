import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import (
    ChatAgentRun,
    SmolagentsModelSettings,
)
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import (
    ConversationTurn,
    OrchestrationFile,
    OrchestrationRequest,
)


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
            return ChatAgentRun(answer="direct answer", state="success")

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

    def test_web_and_local_knowledge_use_supervisor_and_merge_citations(self) -> None:
        calls: list[tuple[bool, bool, str]] = []

        def fake_chat(**kwargs):
            web = bool(kwargs["enable_web_search"])
            local = bool(kwargs["enable_local_knowledge"])
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
                    enable_web_search=True,
                    enable_local_knowledge=True,
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

    def test_one_failed_evidence_agent_does_not_cancel_synthesis(self) -> None:
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
                    enable_web_search=True,
                    enable_local_knowledge=True,
                )
            )
        )

        # 原因：一个外部 provider 故障不应跳过依赖它的最终综合任务。
        # 作用：验证降级贡献进入共享状态，其他证据仍能生成最终答案。
        self.assertTrue(result.success)
        self.assertIn("remaining local evidence", result.final_answer)
        self.assertTrue(
            any(
                event.status == "warning" and event.agent == "research_agent"
                for event in result.trace
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
