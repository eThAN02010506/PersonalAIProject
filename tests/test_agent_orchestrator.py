import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pydantic import ValidationError

from qwopus_agent.agents.multi_agent import MultiAgentRun, NamedAgentRun
from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    ChatAgentRun,
    SmolagentsModelSettings,
)
from qwopus_agent.reports.recipe import default_recipe
from qwopus_agent.services.agent_orchestrator import (
    AgentOrchestrator,
    _CapabilityResult,
    _citations_from_chat,
    _fallback_from_dependencies_for_empty_answer,
    _file_analysis_citations,
    _is_spreadsheet_only_local_computation_request,
    _preserve_requested_table_output,
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

    def test_orchestration_request_normalizes_and_rejects_blank_objectives(
        self,
    ) -> None:
        request = OrchestrationRequest(objective="  Analyze the documents.  ")

        self.assertEqual(request.objective, "Analyze the documents.")
        with self.assertRaisesRegex(ValidationError, "must not be blank"):
            OrchestrationRequest(objective="   ")

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

    def test_synthesis_preserves_requested_verified_table(self) -> None:
        table_answer = (
            "The mean Sepal.Length is 5.843333.\n\n"
            "## Local calculation table\n\n"
            "| column | mean |\n"
            "| --- | --- |\n"
            "| Sepal.Length | 5.843333 |"
        )
        context = {
            "multi_agent": {
                "dependency_results": {"document": table_answer},
                "shared_state": {
                    "contributions": {
                        "document": SimpleNamespace(
                            raw=_CapabilityResult(content=table_answer, success=True)
                        )
                    }
                },
            }
        }

        result = _preserve_requested_table_output(
            "The mean Sepal.Length is 5.843333.",
            "Calculate the mean Sepal.Length and return a table.",
            context,
        )

        # 原因：Synthesizer 会把本地计算表压成纯文字，正式聊天页因此看不到用户要求的表格。
        # 作用：只在用户明确要表格且最终答案缺表时，从成功依赖中补回 verified table。
        self.assertIn("## Local calculation table", result)
        self.assertIn("| Sepal.Length | 5.843333 |", result)

    def test_synthesis_empty_scaffold_falls_back_to_dependency_answer(self) -> None:
        dependency_answer = (
            "The mean Sepal.Length is 5.843333.\n\n"
            "| column | mean |\n"
            "| --- | --- |\n"
            "| Sepal.Length | 5.843333 |"
        )
        context = {
            "multi_agent": {
                "dependency_results": {"document": dependency_answer},
                "shared_state": {
                    "contributions": {
                        "document": SimpleNamespace(
                            raw=_CapabilityResult(
                                content=dependency_answer,
                                success=True,
                            )
                        )
                    }
                },
            }
        }

        result = _fallback_from_dependencies_for_empty_answer(
            "**Direct answer**\n\n**Supporting explanation**\n\n**Practical implication**",
            context,
        )

        # 原因：真实模型偶尔只输出章节壳，直接发布会让用户看到空答案。
        # 作用：锁定这种失败形态会回退到已成功执行的 Tool/Worker 结果。
        self.assertIn("The mean Sepal.Length is 5.843333.", result)
        self.assertIn("| Sepal.Length | 5.843333 |", result)

    def test_spreadsheet_computation_can_skip_local_knowledge_planning(self) -> None:
        request = OrchestrationRequest(
            objective="Calculate the mean score and return a table.",
            uploaded_files=(
                OrchestrationFile(name="scores.xlsx", content=b"placeholder"),
            ),
            enable_local_knowledge=True,
        )
        global_request = request.model_copy(update={"include_global_knowledge": True})

        # 原因：Excel 本地 Tool 已经能给出可核验计算结果，继续 RAG 综合会明显拖慢回答。
        # 作用：只让纯表格计算走快速 document_agent 路径，显式全局知识请求仍保留完整链路。
        self.assertTrue(_is_spreadsheet_only_local_computation_request(request))
        self.assertFalse(
            _is_spreadsheet_only_local_computation_request(global_request)
        )

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

    def test_browser_permission_routes_to_browser_only_agent(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return ChatAgentRun(answer="Rendered page answer", state="success")

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="Open https://example.com and summarize it.",
                    enable_browser=True,
                )
            )
        )

        # 原因：浏览器授权若只到达 Runtime、不进入 Planner，会退回普通 chat_agent。
        # 作用：证明单能力请求由 browser_agent 执行，且 Tavily 保持关闭。
        self.assertTrue(result.success)
        self.assertEqual(result.route, "single_agent")
        self.assertEqual(
            result.multi_agent_run,
            None,
        )
        self.assertTrue(calls[0]["enable_browser"])
        self.assertFalse(calls[0]["enable_web_search"])

    def test_web_and_local_knowledge_use_supervisor_and_merge_citations(self) -> None:
        calls: list[tuple[bool, bool, str]] = []
        output_roles: list[str] = []
        synthesis_materials: list[str] = []

        def fake_chat(**kwargs):
            web = bool(kwargs["enable_web_search"])
            local = bool(kwargs["enable_local_knowledge"])
            self.assertEqual(kwargs["min_source_relevance"], 0.8)
            self.assertEqual(bool(kwargs["include_global_knowledge"]), local)
            question = str(kwargs["user_message"])
            calls.append((web, local, question))
            output_roles.append(str(kwargs["output_role"]))
            if web:
                return ChatAgentRun(
                    answer=(
                        '{"facts":[{"claim":"current external fact",'
                        '"support":"current external support",'
                        '"sources":["https://invented.example"],'
                        '"confidence":"high"}],"limitations":[]}'
                    ),
                    tool_calls=("tavily_search", "final_answer"),
                    observations=("https://example.com/current",),
                    state="success",
                )
            if local:
                return ChatAgentRun(
                    answer=(
                        '{"facts":[{"claim":"stored relationship",'
                        '"support":"stored relationship support",'
                        '"sources":["ownershp.pdf"],'
                        '"confidence":"high"}],"limitations":[]}'
                    ),
                    tool_calls=("graph_search", "final_answer"),
                    observations=("[ownership.pdf, page 4] Company A owns Company B",),
                    state="success",
                )
            self.assertIn("current external fact", question)
            self.assertIn("stored relationship", question)
            self.assertNotIn("https://invented.example", question)
            self.assertNotIn("ownershp.pdf", question)
            if "Audit this evidence" in question:
                return ChatAgentRun(
                    answer="The sources agree on ownership but differ in recency.",
                    state="success",
                )
            self.assertIn("differ in recency", question)
            synthesis_materials.append(question)
            return ChatAgentRun(
                answer=(
                    "combined final answer "
                    "https://example.com/current [ownership.pdf, page 4]"
                ),
                state="success",
            )

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
            [citation.source for citation in result.citations],
            ["https://example.com/current", "ownership.pdf"],
        )
        self.assertEqual(
            [task.agent_name for task in result.multi_agent_run.delegation_plan.tasks],
            [
                "research_agent",
                "knowledge_agent",
                "review_agent",
                "synthesis_agent",
            ],
        )
        self.assertTrue(any(event.tool == "tavily_search" for event in result.trace))
        self.assertTrue(any(event.tool == "graph_search" for event in result.trace))
        self.assertTrue(
            any(
                event.phase == "reflection" and event.status == "completed"
                for event in result.trace
            )
        )
        self.assertEqual(len(calls), 4)
        # 原因：能得到综合答案并不能证明中间 Agent 没有各自生成一篇用户答案。
        # 作用：锁定两路 Worker 只产证据、Reviewer 只审证据、最后一次模型调用才负责写作。
        self.assertEqual(output_roles[:2].count("evidence"), 2)
        self.assertEqual(output_roles[-2:], ["review", "final"])
        self.assertIn("Evidence ledger", synthesis_materials[0])
        self.assertIn("Evidence review", synthesis_materials[0])
        self.assertIn("provide the specific support", synthesis_materials[0])
        self.assertIn("practical implication", synthesis_materials[0])
        self.assertNotIn("Agent evidence:", synthesis_materials[0])

    def test_review_failure_cannot_publish_worker_content_or_citations(self) -> None:
        review_debug = AgentDebugRun(
            label="review",
            prompt="internal review prompt",
            max_steps=2,
            state="generation_error",
            output='{"agreements":[],"resolution":"partial review"}',
        )

        def fake_chat(**kwargs):
            if kwargs["output_role"] == "review":
                return ChatAgentRun(
                    answer='{"agreements":[],"resolution":"partial review"}',
                    state="generation_error",
                    success=False,
                    error="review model failed",
                    debug_runs=(review_debug,),
                )
            if kwargs["enable_web_search"]:
                return ChatAgentRun(
                    answer=(
                        '{"facts":[{"claim":"Web fact","support":"Web support",'
                        '"sources":["https://invented.example"],"confidence":"high"}],'
                        '"limitations":["web limitation"]}'
                    ),
                    observations=("https://example.com/verified",),
                    state="success",
                )
            return ChatAgentRun(
                answer=(
                    '{"facts":[{"claim":"Knowledge fact","support":"Local support",'
                    '"sources":["misspelled.pdf"],"confidence":"high"}],'
                    '"limitations":["knowledge limitation"]}'
                ),
                observations=("[verified.pdf, page 2] Local support",),
                state="success",
            )

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="Compare the uploaded evidence with current research.",
                    conversation_id="conversation-1",
                    enable_web_search=True,
                    enable_local_knowledge=True,
                )
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.citations, ())
        self.assertNotIn('"facts"', result.final_answer)
        self.assertNotIn('"limitations"', result.final_answer)
        self.assertNotIn("partial review", result.final_answer)
        self.assertNotIn("Observation", result.final_answer)
        self.assertNotIn("https://example.com/verified", result.final_answer)
        runs = {item.task_id: item for item in result.multi_agent_run.runs}
        self.assertFalse(runs["review"].success)
        self.assertFalse(runs["synthesis"].success)
        self.assertEqual(runs["review"].error, "review model failed")
        self.assertEqual(runs["review"].result.error, "review model failed")
        self.assertEqual(result.debug_runs[-1].output, review_debug.output)

    def test_synthesis_failure_cannot_publish_review_or_evidence(self) -> None:
        synthesis_debug = AgentDebugRun(
            label="synthesis",
            prompt="internal synthesis prompt",
            max_steps=2,
            state="generation_error",
            output="Observation: incomplete finalization",
        )

        def fake_chat(**kwargs):
            role = kwargs["output_role"]
            if role == "review":
                return ChatAgentRun(
                    answer=(
                        '{"agreements":["Evidence agrees"],"conflicts":[],'
                        '"unsupported_claims":[],"gaps":[],"resolution":"Use it."}'
                    ),
                    state="success",
                )
            if role == "final":
                return ChatAgentRun(
                    answer="Observation: incomplete finalization",
                    state="generation_error",
                    success=False,
                    error="synthesis model failed",
                    debug_runs=(synthesis_debug,),
                )
            observation = (
                "https://example.com/verified"
                if kwargs["enable_web_search"]
                else "[verified.pdf, page 2] Local support"
            )
            return ChatAgentRun(
                answer=(
                    '{"facts":[{"claim":"Intermediate fact","support":"Internal support",'
                    '"sources":[],"confidence":"high"}],"limitations":[]}'
                ),
                observations=(observation,),
                state="success",
            )

        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective="Compare all evidence.",
                    conversation_id="conversation-1",
                    enable_web_search=True,
                    enable_local_knowledge=True,
                )
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.citations, ())
        self.assertNotIn("Intermediate fact", result.final_answer)
        self.assertNotIn("Evidence agrees", result.final_answer)
        self.assertNotIn("Observation", result.final_answer)
        runs = {item.task_id: item for item in result.multi_agent_run.runs}
        self.assertFalse(runs["synthesis"].success)
        self.assertEqual(runs["synthesis"].error, "synthesis model failed")
        self.assertEqual(runs["synthesis"].result.error, "synthesis model failed")
        self.assertEqual(result.debug_runs[-1].output, synthesis_debug.output)

    def test_success_state_with_internal_synthesis_payload_is_rejected(self) -> None:
        internal_payload = (
            '{"facts":[{"claim":"Still internal","support":"Do not publish",'
            '"sources":[],"confidence":"medium"}],"limitations":[]}'
        )

        def fake_chat(**kwargs):
            role = kwargs["output_role"]
            if role == "review":
                return ChatAgentRun(
                    answer=(
                        '{"agreements":[],"conflicts":[],"unsupported_claims":[],'
                        '"gaps":[],"resolution":"Synthesize."}'
                    ),
                    state="success",
                )
            if role == "final":
                return ChatAgentRun(answer=internal_payload, state="success")
            return ChatAgentRun(
                answer=(
                    '{"facts":[{"claim":"Evidence","support":"Support",'
                    '"sources":[],"confidence":"medium"}],"limitations":[]}'
                ),
                state="success",
            )

        intent = ResolvedIntent(
            original_request="Provide a detailed comparison.",
            operational_objective="Provide a detailed comparison.",
            task_type="compare",
            answer_contract=AnswerContract(
                task_type="compare",
                complexity="complex",
            ),
        )
        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective=intent.original_request,
                    resolved_intent=intent,
                    enable_web_search=True,
                )
            )
        )

        self.assertFalse(result.success)
        self.assertNotIn("Still internal", result.final_answer)
        self.assertEqual(result.citations, ())
        synthesis = {
            item.task_id: item for item in result.multi_agent_run.runs
        }["synthesis"]
        self.assertFalse(synthesis.success)
        self.assertIn("internal pipeline payload", synthesis.error)

    def test_internal_review_and_synthesis_bypass_only_user_document_preflight(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            role = kwargs["output_role"]
            if role == "review":
                return ChatAgentRun(
                    answer=(
                        '{"agreements":["Ledger is usable"],"conflicts":[],'
                        '"unsupported_claims":[],"gaps":[],"resolution":"Synthesize it."}'
                    ),
                    state="success",
                )
            if role == "final":
                return ChatAgentRun(
                    answer="Final document comparison [verified.pdf, page 2]",
                    state="success",
                )
            observation = (
                "https://example.com/verified"
                if kwargs["enable_web_search"]
                else "[verified.pdf, page 2] Local support"
            )
            return ChatAgentRun(
                answer=(
                    '{"facts":[{"claim":"Verified fact","support":"Verified support",'
                    '"sources":[],"confidence":"high"}],"limitations":[]}'
                ),
                observations=(observation,),
                state="success",
            )

        intent = ResolvedIntent(
            original_request="总结我上传的文档并与网页资料对比",
            operational_objective="总结我上传的文档并与网页资料对比",
            task_type="compare",
            answer_contract=AnswerContract(task_type="compare"),
        )
        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective=intent.original_request,
                    resolved_intent=intent,
                    conversation_id="conversation-1",
                    enable_web_search=True,
                    enable_local_knowledge=True,
                )
            )
        )

        self.assertTrue(result.success)
        evidence_calls = [call for call in calls if call["output_role"] == "evidence"]
        internal_calls = [call for call in calls if call["output_role"] != "evidence"]
        self.assertTrue(all(call["enforce_document_evidence"] for call in evidence_calls))
        self.assertTrue(
            all(not call["enforce_document_evidence"] for call in internal_calls)
        )
        self.assertTrue(
            all(
                call["response_language_source"] == intent.original_request
                for call in internal_calls
            )
        )

    def test_terminal_failure_states_never_promote_arbiter_fallback(self) -> None:
        worker_json = (
            '{"facts":[{"claim":"Internal worker result","support":"Do not expose",'
            '"sources":[],"confidence":"medium"}],"limitations":[]}'
        )

        class TerminalStateExecutor:
            def __init__(self, terminal_error: str | None) -> None:
                self.terminal_error = terminal_error

            async def execute(self, plan, **_kwargs):
                runs = [
                    NamedAgentRun(
                        name="research_agent",
                        result=worker_json,
                        task_id="research",
                        success=True,
                    )
                ]
                if self.terminal_error is not None:
                    runs.append(
                        NamedAgentRun(
                            name="synthesis_agent",
                            result=None,
                            task_id=plan.terminal_task_id,
                            success=False,
                            error=self.terminal_error,
                        )
                    )
                return MultiAgentRun(
                    objective=plan.delegation.objective,
                    runs=runs,
                    delegation_plan=plan.delegation,
                    final_answer=worker_json,
                )

        for terminal_error in (
            None,
            "Skipped because dependencies failed: review",
            "terminal task was skipped",
            "terminal task was cancelled",
            "RuntimeError: synthesis exploded",
        ):
            with self.subTest(terminal_error=terminal_error):
                result = asyncio.run(
                    AgentOrchestrator(
                        self.settings,
                        executor=TerminalStateExecutor(terminal_error),
                    ).run(
                        OrchestrationRequest(
                            objective="Compare all available sources.",
                            conversation_id="conversation-1",
                            enable_web_search=True,
                            enable_local_knowledge=True,
                        )
                    )
                )

                self.assertFalse(result.success)
                self.assertEqual(result.citations, ())
                self.assertNotIn("Internal worker result", result.final_answer)
                self.assertNotIn('"facts"', result.final_answer)

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

    def test_complex_detailed_route_fills_one_structured_review_gap(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_chat(**kwargs):
            role = str(kwargs["output_role"])
            question = str(kwargs["user_message"])
            calls.append((role, question))
            if role == "review":
                return ChatAgentRun(
                    answer=(
                        '{"agreements":["Base evidence is usable"],"conflicts":[],'
                        '"unsupported_claims":[],"gaps":["Find an independent deployment '
                        'measurement"],"resolution":"Verify the missing measurement."}'
                    ),
                    state="success",
                )
            if role == "evidence" and "Fill only these reviewed evidence gaps" in question:
                return ChatAgentRun(
                    answer=(
                        '{"facts":[{"claim":"Independent measurement exists",'
                        '"support":"A second source measured deployment.",'
                        '"sources":["https://example.com/measurement"],'
                        '"confidence":"high"}],"limitations":[]}'
                    ),
                    state="success",
                )
            if role == "evidence":
                return ChatAgentRun(
                    answer=(
                        '{"facts":[{"claim":"Base finding","support":"Primary evidence",'
                        '"sources":["https://example.com/base"],'
                        '"confidence":"medium"}],"limitations":[]}'
                    ),
                    state="success",
                )
            self.assertIn("Independent measurement exists", question)
            return ChatAgentRun(answer="Integrated detailed answer", state="success")

        intent = ResolvedIntent(
            original_request="Investigate this claim in detail",
            operational_objective="Investigate this claim in detail",
            task_type="analyze",
            answer_contract=AnswerContract(
                task_type="analyze",
                complexity="complex",
                response_detail="detailed",
            ),
        )
        result = asyncio.run(
            AgentOrchestrator(self.settings, chat_runner=fake_chat).run(
                OrchestrationRequest(
                    objective=intent.original_request,
                    resolved_intent=intent,
                    enable_web_search=True,
                )
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [role for role, _question in calls],
            ["evidence", "review", "evidence", "final"],
        )
        self.assertEqual(
            [task.task_id for task in result.multi_agent_run.delegation_plan.tasks],
            ["research", "review", "gap_fill", "synthesis"],
        )

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

        # 原因：依赖失败后 Arbiter 仍可能选择成功 Worker 的内部 Evidence 作为答案。
        # 作用：terminal 未完成时 fail closed，同时在 MultiAgentRun 中保留各节点诊断。
        self.assertFalse(result.success)
        self.assertNotIn("local evidence", result.final_answer)
        self.assertEqual(result.citations, ())
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

        captured: dict[str, object] = {}

        def fake_analysis(**kwargs):
            captured["recipe"] = kwargs.get("recipe")
            return SimpleNamespace(
                result=local_result,
                debug_steps=["local parse completed"],
                analyzed_file_names=["notes.txt"],
            )

        def fake_chat(**kwargs):
            self.assertIn("document answer", kwargs["user_message"])
            return ChatAgentRun(
                answer="synthesized document answer [Source: notes.txt]",
                state="success",
            )

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
        # 编排层总是把默认 recipe 传给文件分析 runner。
        self.assertIs(captured["recipe"], default_recipe())
        self.assertIn("notes.txt", result.final_answer)
        self.assertEqual(list(generator.kwargs["tables"]), ["summary"])
        self.assertEqual(
            [task.agent_name for task in result.multi_agent_run.delegation_plan.tasks],
            ["document_agent", "synthesis_agent", "report_agent"],
        )


if __name__ == "__main__":
    unittest.main()
