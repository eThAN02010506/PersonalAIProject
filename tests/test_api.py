import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.api.app import SPAStaticFiles, create_app
from qwopus_agent.api.model_runtime import ModelRuntimeError, RuntimeModelStatus
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.routes.analysis import analysis_view
from qwopus_agent.code_workspace.models import CodeWorkspaceAgentRun
from qwopus_agent.code_workspace.repository import CodeChangeRepository
from qwopus_agent.documents import DocumentStore, build_document_structure, chunk_document_structure
from qwopus_agent.integrations.smolagents_runtime import AgentDebugRun, SmolagentsModelSettings
from qwopus_agent.integrations.tavily_credentials import TavilyCredentialStore
from qwopus_agent.llm import BaseLLM, ChatMessage, LLMResponse, ModelCapabilities
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.memory.graph_extraction import (
    CompositeGraphExtractor,
    LLMGraphExtractor,
    RuleBasedGraphExtractor,
)
from qwopus_agent.services.code_workspace_service import CodeWorkspaceService
from qwopus_agent.services.orchestration_models import OrchestrationResult
from qwopus_agent.services.skill_growth_service import SkillRunTrace, SkillTraceStep
from qwopus_agent.utils.debug_store import load_debug_records


class ApiStaticLLM(BaseLLM):
    """Deterministic authoring model that never reaches the configured server."""

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                '{"name":"debug_report","description":"Inspect then summarize.",'
                '"intent_examples":["prepare a report"],'
                '"steps":[{"skill_name":"excel_schema",'
                '"query_template":"Inspect {query}","arguments":{}}]}'
            ),
            model="api-authoring-model",
        )


class ApiConversationSkillLLM(BaseLLM):
    """Return one workflow draft followed by one independent approval."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        content = (
            '{"name":"sourced_research","description":"Research current sources.",'
            '"intent_examples":["research this topic"],'
            '"steps":[{"skill_name":"web_search",'
            '"query_template":"Research {query}","arguments":{}}]}'
            if self.calls == 1
            else '{"approved":true,"issues":[]}'
        )
        return LLMResponse(content=content, model="conversation-authoring-model")


class ApiCodeProposalLLM(BaseLLM):
    """Return a deterministic source replacement without contacting a server."""

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                '{"summary":"Clarify value","reason":"The new value is explicit.",'
                '"verification_plan":["Run git diff check"],'
                '"changes":[{"path":"sample.py","replacements":['
                '{"old_text":"VALUE = 1","new_text":"VALUE = 2"}]}]}'
            ),
            model="api-code-model",
        )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "qwopus.db"
        self.debug_directory = Path(self.temp_directory.name) / "debug_runs"
        self.runtime_log_path = Path(self.temp_directory.name) / "qwopus_agent.log"
        self.knowledge_root = Path(self.temp_directory.name) / "minirag"
        self.document_directory = Path(self.temp_directory.name) / "documents"
        self.code_root = Path(self.temp_directory.name) / "code-repository"
        self.code_root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.code_root)], check=True)
        (self.code_root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.code_workspace_service = CodeWorkspaceService(
            CodeChangeRepository(Path(self.temp_directory.name) / "code_changes"),
            llm_factory=ApiCodeProposalLLM,
            code_chat_runner=lambda _root, _transcript, _paths, _selected: (
                CodeWorkspaceAgentRun(
                    content=(
                        '{"mode":"ready","message":"I inspected sample.py and found the '
                        'current configuration value. It can be clarified without changing '
                        'the module structure and verified through the existing diff check.",'
                        '"objective":"Change the sample value while preserving module structure.",'
                        '"selected_files":["sample.py"]}'
                    ),
                    inspected_files=["sample.py"],
                    tool_calls=["code_search", "code_read"],
                    state="success",
                )
            ),
        )
        self.tavily_credentials = TavilyCredentialStore(
            path=Path(self.temp_directory.name) / "secrets" / "tavily.key",
            legacy_env_path=Path(self.temp_directory.name) / "missing.env",
        )
        # 原因：API 测试必须验证 SQLite 持久化，但不能读取或修改用户现有对话。
        # 作用：每个测试使用独立数据库，并关闭旧 JSONL 自动导入。
        self.repository = ConversationRepository(database_path, import_legacy=False)
        # 原因：API 边界单元测试不应加载或写入用户真实的 storage/minirag 索引。
        # 作用：注入隔离替身；真实 MinerU/MiniRAG 链路由端到端用例单独验证。
        self.model_settings = SmolagentsModelSettings(
            model_id="test-model",
            base_url="http://127.0.0.1:9999/v1",
        )
        self.model_status = RuntimeModelStatus(
            mode="remote",
            settings=self.model_settings,
            online=True,
            message="online",
        )
        self.model_runtime = MagicMock()
        self.model_runtime.current_settings.return_value = self.model_settings
        self.model_runtime.require_online_settings.return_value = self.model_settings
        self.model_runtime.status.return_value = self.model_status
        self.model_runtime.configure_remote.return_value = self.model_status
        self.model_runtime.configure_local.return_value = self.model_status
        self.knowledge_manager = ConversationKnowledgeManager(
            root=self.knowledge_root,
            factory=lambda _path: MagicMock(),
        )
        self.client_context = TestClient(
            create_app(
                self.repository,
                knowledge_manager=self.knowledge_manager,
                model_runtime=self.model_runtime,
                debug_directory=self.debug_directory,
                runtime_log_path=self.runtime_log_path,
                document_store=DocumentStore(self.document_directory),
                tavily_credentials=self.tavily_credentials,
                code_workspace_service=self.code_workspace_service,
            )
        )
        self.client = self.client_context.__enter__()
        initialized = self.client.post(
            "/api/auth/bootstrap",
            json={
                "username": "test-admin",
                "display_name": "Test Admin",
                "password": "test-password-123",
            },
        )
        self.assertEqual(initialized.status_code, 201)
        self.user_id = initialized.json()["user"]["id"]

    def test_code_workspace_requires_proposal_approval_and_supports_rollback(self) -> None:
        scanned = self.client.post(
            "/api/code-workspaces/scan",
            json={"path": str(self.code_root)},
        )
        proposed = self.client.post(
            "/api/code-changes/propose",
            json={
                "root": str(self.code_root),
                "objective": "Change the sample value.",
                "selected_files": ["sample.py"],
            },
        )

        self.assertEqual(scanned.status_code, 200)
        self.assertEqual(scanned.json()["file_count"], 1)
        self.assertEqual(proposed.status_code, 200)
        self.assertEqual(proposed.json()["status"], "proposed")
        self.assertEqual(
            (self.code_root / "sample.py").read_text(encoding="utf-8"),
            "VALUE = 1\n",
        )

        change_id = proposed.json()["id"]
        applied = self.client.post(f"/api/code-changes/{change_id}/apply")
        rolled_back = self.client.post(f"/api/code-changes/{change_id}/rollback")

        self.assertEqual(applied.json()["status"], "applied")
        self.assertEqual(rolled_back.json()["status"], "rolled_back")
        self.assertEqual(
            (self.code_root / "sample.py").read_text(encoding="utf-8"),
            "VALUE = 1\n",
        )
        records = load_debug_records(directory=self.debug_directory)
        self.assertTrue(any(record["source"] == "code_workspace" for record in records))

    def test_code_workspace_chat_prepares_grounded_change_without_writing(self) -> None:
        discussed = self.client.post(
            "/api/code-workspaces/chat",
            json={
                "root": str(self.code_root),
                "message": "Make this configuration value clearer.",
                "history": [],
                "selected_files": [],
            },
        )

        self.assertEqual(discussed.status_code, 200)
        self.assertEqual(discussed.json()["mode"], "ready")
        self.assertEqual(discussed.json()["selected_files"], ["sample.py"])
        self.assertEqual(
            (self.code_root / "sample.py").read_text(encoding="utf-8"),
            "VALUE = 1\n",
        )

    def test_admin_can_manage_and_test_tavily_without_key_disclosure(self) -> None:
        api_key = "tvly-test-admin-secret-123456"
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": ""}),
            patch(
                "qwopus_agent.api.routes.web_search_settings._test_connection",
                return_value={
                    "success": True,
                    "message": "Tavily search is ready.",
                },
            ),
        ):
            initial = self.client.get("/api/web-search-settings")
            saved = self.client.put(
                "/api/web-search-settings",
                json={"api_key": api_key},
            )
            tested = self.client.post(
                "/api/web-search-settings/test",
                json={},
            )
            deleted = self.client.delete("/api/web-search-settings")

        self.assertFalse(initial.json()["configured"])
        self.assertTrue(saved.json()["configured"])
        self.assertEqual(saved.json()["source"], "managed")
        self.assertNotIn(api_key, saved.text)
        self.assertTrue(tested.json()["success"])
        self.assertFalse(deleted.json()["configured"])
        self.assertFalse(self.tavily_credentials.path.exists())

    def test_web_run_is_rejected_before_model_use_when_tavily_is_missing(self) -> None:
        conversation = self.client.post(
            "/api/conversations",
            json={"title": "Web test"},
        ).json()
        self.model_runtime.require_online_settings.reset_mock()

        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/runs",
                json={"content": "Search current information", "enable_web_search": True},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("administrator configures Tavily", response.json()["detail"])
        self.model_runtime.require_online_settings.assert_not_called()

    def test_member_can_see_availability_but_cannot_manage_tavily(self) -> None:
        api_key = "tvly-test-member-hidden-123456"
        self.tavily_credentials.save(api_key)
        created = self.client.post(
            "/api/users",
            json={
                "username": "member",
                "display_name": "Member",
                "password": "member-password-123",
                "role": "member",
            },
        )
        self.assertEqual(created.status_code, 201)
        self.client.post("/api/auth/logout")
        logged_in = self.client.post(
            "/api/auth/login",
            json={
                "username": "member",
                "password": "member-password-123",
            },
        )
        self.assertEqual(logged_in.status_code, 200)

        status = self.client.get("/api/web-search-settings")
        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json()["configured"])
        self.assertFalse(status.json()["can_manage"])
        self.assertIsNone(status.json()["source"])
        self.assertIsNone(status.json()["masked_key"])
        self.assertNotIn(api_key, status.text)
        self.assertEqual(
            self.client.put(
                "/api/web-search-settings",
                json={"api_key": "tvly-member-replacement-123456"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post("/api/web-search-settings/test", json={}).status_code,
            403,
        )
        self.assertEqual(
            self.client.delete("/api/web-search-settings").status_code,
            403,
        )

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temp_directory.cleanup()

    def test_default_knowledge_composition_uses_current_model_for_graph_extraction(
        self,
    ) -> None:
        with patch("qwopus_agent.memory.minirag.MiniRAG") as minirag_class:
            app = create_app(
                self.repository,
                model_runtime=self.model_runtime,
                debug_directory=self.debug_directory,
                runtime_log_path=self.runtime_log_path,
                document_store=DocumentStore(self.document_directory),
            )
            factory = app.state.knowledge_manager.factory
            self.assertIsNotNone(factory)
            factory(self.knowledge_root / "documents.jsonl")

        extractor = minirag_class.call_args.kwargs["graph_extractor"]
        self.assertIsInstance(extractor, CompositeGraphExtractor)
        self.assertIsInstance(extractor.extractors[0], RuleBasedGraphExtractor)
        self.assertIsInstance(extractor.extractors[1], LLMGraphExtractor)

        # 原因：应用启动后用户仍可切换模型地址和模型 ID，图抽取不能捕获启动时快照。
        # 作用：调用 extractor 的工厂时才读取 RuntimeModelController 当前在线设置。
        extractor.extractors[1].llm_factory()
        self.model_runtime.require_online_settings.assert_called()

    def test_conversation_crud_and_message_isolation(self) -> None:
        first = self.client.post("/api/conversations", json={"title": "First"})
        second = self.client.post("/api/conversations", json={"title": "Second"})

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        first_id = first.json()["id"]
        second_id = second.json()["id"]
        self.repository.add_message(first_id, "user", "first question")

        first_messages = self.client.get(f"/api/conversations/{first_id}/messages")
        second_messages = self.client.get(f"/api/conversations/{second_id}/messages")
        self.assertEqual(first_messages.json()[0]["content"], "first question")
        self.assertEqual(second_messages.json(), [])

        renamed = self.client.patch(
            f"/api/conversations/{first_id}",
            json={"title": "Renamed"},
        )
        self.assertEqual(renamed.json()["title"], "Renamed")
        scoped_directory = self.knowledge_manager.storage_path(first_id).parent
        scoped_directory.mkdir(parents=True)
        (scoped_directory / "marker").write_text("private", encoding="utf-8")
        self.assertEqual(self.client.delete(f"/api/conversations/{first_id}").status_code, 204)
        self.assertFalse(scoped_directory.exists())
        self.assertEqual(
            self.client.get(f"/api/conversations/{first_id}/messages").status_code,
            404,
        )

    def test_ambiguous_continuation_clarifies_without_model_service(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "New chat"},
        ).json()["id"]
        self.model_runtime.require_online_settings.reset_mock()
        self.model_runtime.require_online_settings.side_effect = ModelRuntimeError(
            "offline"
        )

        started = self.client.post(
            f"/api/conversations/{conversation_id}/runs",
            json={
                "content": "继续",
                "interpretation_mode": "contextual",
            },
        )
        result = self.client.get(f"/api/runs/{started.json()['run_id']}")
        messages = self.client.get(
            f"/api/conversations/{conversation_id}/messages"
        ).json()

        # 原因：没有上一任务时让模型猜“继续什么”既不可靠，也不应依赖模型在线。
        # 作用：锁定意图层直接返回同语言澄清，并沿用标准 run/message 协议。
        self.assertEqual(started.status_code, 200)
        self.assertEqual(result.json()["status"], "completed")
        self.assertEqual(result.json()["phase"], "clarification")
        self.assertIn("具体任务", result.json()["answer"])
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.model_runtime.require_online_settings.assert_not_called()

    def test_deleting_conversation_cancels_its_active_runs_first(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Running"},
        ).json()["id"]
        runs = self.client.app.state.runs

        with patch("qwopus_agent.api.runs.start_chat_task") as start_task:
            task = start_task.return_value
            run_id = runs.start(
                conversation_id,
                "long task",
                self.model_settings,
                enable_web_search=False,
                enable_local_knowledge=False,
            )

            response = self.client.delete(f"/api/conversations/{conversation_id}")

        # 原因：浏览器可以关闭或切换聊天，删除一致性不能依靠 UI 的 isRunning。
        # 作用：证明 worker 在 SQLite 和私有知识目录删除前被取消，并保留可轮询终态。
        self.assertEqual(response.status_code, 204)
        task.cancel.assert_called_once()
        self.assertEqual(runs.poll(run_id).status, "cancelled")

    def test_openapi_exposes_agent_and_document_boundaries(self) -> None:
        schema = self.client.get("/openapi.json").json()

        self.assertIn("/api/conversations/{conversation_id}/runs", schema["paths"])
        self.assertIn("/api/runs/{run_id}", schema["paths"])
        self.assertIn("/api/analysis", schema["paths"])
        self.assertIn("/api/local-folders/scan", schema["paths"])
        self.assertIn("/api/local-folders/analyze", schema["paths"])
        self.assertIn("/api/documents", schema["paths"])
        self.assertIn("/api/model-settings", schema["paths"])
        self.assertIn("/api/reports/{filename}", schema["paths"])
        self.assertIn("/api/debug", schema["paths"])
        self.assertIn("/api/skills", schema["paths"])
        self.assertIn("/api/debug/skills/generate", schema["paths"])
        self.assertIn("/api/debug/skills/from-runs", schema["paths"])
        self.assertIn("/api/debug/skills/{name}/{version}/test", schema["paths"])

    def test_skill_api_promotes_and_rolls_back_reviewed_versions(self) -> None:
        growth = self.client.app.state.skill_growth
        first_trace = SkillRunTrace(
            success=True,
            output=(
                "A complete reusable research result with sources, context, "
                "and a clear conclusion."
            ),
            steps=(SkillTraceStep("web_search"),),
        )
        for objective in ("research current prices", "research current schedules"):
            growth.observe_trace(objective, first_trace)

        listed = self.client.get("/api/skills")
        promoted = self.client.post(
            "/api/skills/learned_web_search/0.1.0/promote"
        )

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["status"], "candidate")
        self.assertTrue(listed.json()[0]["spec_valid"])
        self.assertEqual(promoted.status_code, 200)
        self.assertEqual(promoted.json()["status"], "active")
        self.assertNotIn("spec_path", promoted.json())

        second_trace = SkillRunTrace(
            success=True,
            output=(
                "A complete reusable research summary with sources, context, "
                "and a clear conclusion."
            ),
            steps=(SkillTraceStep("web_search", {"mode": "summary"}),),
        )
        for objective in ("summarize current prices", "summarize current schedules"):
            growth.observe_trace(objective, second_trace)
        promoted_second = self.client.post(
            "/api/skills/learned_web_search/0.1.1/promote"
        )
        rolled_back = self.client.post(
            "/api/skills/learned_web_search/0.1.0/rollback"
        )

        self.assertEqual(promoted_second.status_code, 200)
        self.assertEqual(rolled_back.status_code, 200)
        self.assertEqual(rolled_back.json()["version"], "0.1.0")
        self.assertEqual(
            self.client.app.state.skill_catalog.active(
                "learned_web_search"
            ).version,
            "0.1.0",
        )

    def test_debug_console_generates_reviews_and_tests_workflow_candidate(self) -> None:
        authoring = self.client.app.state.skill_authoring
        authoring.llm_factory = lambda: ApiStaticLLM()

        capabilities = self.client.get("/api/debug/skills/capabilities")
        generated = self.client.post(
            "/api/debug/skills/generate",
            json={
                "goal": "Prepare a spreadsheet report",
                "requested_name": "spreadsheet_report",
                "intent_examples": ["analyze this workbook"],
                "allowed_skills": ["excel_schema"],
            },
        )

        self.assertEqual(capabilities.status_code, 200)
        self.assertIn(
            "excel_schema",
            [item["name"] for item in capabilities.json()],
        )
        self.assertEqual(generated.status_code, 200)
        payload = generated.json()
        self.assertEqual(payload["skill"]["name"], "learned_spreadsheet_report")
        self.assertEqual(payload["skill"]["status"], "candidate")
        self.assertEqual(payload["skill"]["source_model"], "api-authoring-model")
        self.assertIn("+++ learned_spreadsheet_report@0.1.0", payload["diff"])
        self.assertTrue(all(check["passed"] for check in payload["checks"]))
        self.assertNotIn(
            "learned_spreadsheet_report",
            self.client.app.state.skill_registry.list_names(),
        )

        detail = self.client.get(
            "/api/debug/skills/learned_spreadsheet_report/0.1.0"
        )
        tested = self.client.post(
            "/api/debug/skills/learned_spreadsheet_report/0.1.0/test",
            json={"query": "sales.xlsx"},
        )
        promoted = self.client.post(
            "/api/skills/learned_spreadsheet_report/0.1.0/promote"
        )

        self.assertEqual(detail.status_code, 200)
        self.assertIsNone(detail.json()["model_output"])
        self.assertEqual(tested.status_code, 200)
        self.assertTrue(tested.json()["success"])
        self.assertEqual(
            tested.json()["steps"][0]["query"],
            "Inspect sales.xlsx",
        )
        self.assertEqual(promoted.status_code, 200)
        self.assertIn(
            "learned_spreadsheet_report",
            self.client.app.state.skill_registry.list_names(),
        )

    def test_debug_console_generates_candidate_from_conversation_run(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Reusable research"},
        ).json()["id"]
        user_message = self.repository.add_message(
            conversation_id,
            "user",
            "Research current rice prices",
        )
        assistant_message = self.repository.add_message(
            conversation_id,
            "assistant",
            "Current rice price findings with cited sources.",
        )
        self.repository.save_conversation_run(
            run_id="source-run-1",
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            requested_by_user_id=self.user_id,
            objective="Research current rice prices",
            operational_objective="Find and summarize current rice price sources",
            status="completed",
            model_id="source-model",
            reusable_skills=("web_search",),
        )
        llm = ApiConversationSkillLLM()
        self.client.app.state.skill_authoring.llm_factory = lambda: llm

        conversations = self.client.get(
            "/api/debug/skills/source-conversations"
        )
        runs = self.client.get(
            f"/api/debug/skills/source-conversations/{conversation_id}/runs"
        )
        generated = self.client.post(
            "/api/debug/skills/from-runs",
            json={
                "conversation_id": conversation_id,
                "run_ids": ["source-run-1"],
                "requested_name": "current_research",
            },
        )

        self.assertEqual(conversations.status_code, 200)
        self.assertEqual(conversations.json()[0]["id"], conversation_id)
        self.assertEqual(runs.status_code, 200)
        self.assertEqual(runs.json()[0]["reusable_skills"], ["web_search"])
        self.assertIn("cited sources", runs.json()[0]["answer_preview"])
        self.assertEqual(generated.status_code, 200)
        self.assertEqual(generated.json()["skill"]["status"], "candidate")
        self.assertEqual(
            generated.json()["skill"]["source_run_id"],
            "conversation-runs:source-run-1",
        )
        self.assertEqual(llm.calls, 2)

    def test_spa_static_files_do_not_serve_parent_files(self) -> None:
        root = Path(self.temp_directory.name)
        frontend = root / "frontend"
        frontend.mkdir()
        (frontend / "index.html").write_text("safe spa", encoding="utf-8")
        (root / "secret.txt").write_text("private value", encoding="utf-8")
        static_app = FastAPI()
        static_app.mount("/", SPAStaticFiles(directory=frontend), name="frontend")

        with TestClient(static_app) as client:
            response = client.get("/..%2Fsecret.txt")
            spa_response = client.get("/debug")

        # 原因：SPA fallback 必须支持前端路由，但不能把编码后的父目录当作静态文件。
        # 作用：确认越界路径被拒绝，同时普通 React 路由仍得到安全入口。
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("private value", response.text)
        self.assertEqual(spa_response.status_code, 200)
        self.assertEqual(spa_response.text, "safe spa")

    def test_saved_document_inventory_lists_only_complete_parsed_documents(self) -> None:
        original = Path(self.temp_directory.name) / "paper.md"
        original.write_text("# One\nBody\n\n# Two\nBody", encoding="utf-8")
        structure = chunk_document_structure(
            build_document_structure(original.read_text(encoding="utf-8"), source="paper.md")
        )
        store = DocumentStore(self.document_directory)
        complete = store.persist(
            original_path=original,
            markdown=original.read_text(encoding="utf-8"),
            structure=structure,
            metadata={"parser": "markdown"},
        )
        (complete / "document_summary.md").write_text("Summary", encoding="utf-8")
        incomplete = self.document_directory / "document-incomplete"
        incomplete.mkdir()
        (incomplete / "metadata.json").write_text("{}", encoding="utf-8")
        self.repository.register_document(
            structure.document_id,
            owner_user_id=self.user_id,
        )

        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["source"], "paper.md")
        self.assertEqual(response.json()[0]["section_count"], 2)
        self.assertTrue(response.json()[0]["summary_available"])

    def test_chat_receives_attached_spreadsheet_only_with_local_knowledge(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Spreadsheet chat"},
        ).json()["id"]
        original = Path(self.temp_directory.name) / "sales.xlsx"
        original.write_bytes(b"test workbook placeholder")
        structure = chunk_document_structure(
            build_document_structure(
                "# Sales workbook\nRevenue and region columns.",
                source=original.name,
            )
        )
        store = DocumentStore(self.document_directory)
        store.persist(
            original_path=original,
            markdown="# Sales workbook\nRevenue and region columns.",
            structure=structure,
            metadata={"parser": "spreadsheet"},
        )
        self.repository.register_document(
            structure.document_id,
            owner_user_id=self.user_id,
            conversation_id=conversation_id,
        )

        with patch("qwopus_agent.api.runs.start_chat_task") as start_task:
            start_task.return_value.refresh_phase.return_value = "executing"
            start_task.return_value.poll_result.return_value = None
            enabled = self.client.post(
                f"/api/conversations/{conversation_id}/runs",
                json={
                    "content": "Calculate average revenue and show details in a table.",
                    "enable_local_knowledge": True,
                },
            )
            enabled_files = start_task.call_args.kwargs["uploaded_files"]
            disabled = self.client.post(
                f"/api/conversations/{conversation_id}/runs",
                json={
                    "content": "Explain how averages are calculated.",
                    "enable_local_knowledge": False,
                },
            )
            disabled_files = start_task.call_args.kwargs["uploaded_files"]
            knowledge_only = self.client.post(
                f"/api/conversations/{conversation_id}/runs",
                json={
                    "content": "What is the uploaded note about?",
                    "enable_local_knowledge": True,
                },
            )
            knowledge_only_files = start_task.call_args.kwargs["uploaded_files"]

        # 原因：MiniRAG 中的摘要不能支持聊天时才提出的任意表格计算。
        # 作用：只有计算型表格问题交给本地 Excel Tool，普通知识问答避免无关表格污染证据。
        self.assertEqual(enabled.status_code, 200)
        self.assertEqual(disabled.status_code, 200)
        self.assertEqual(knowledge_only.status_code, 200)
        self.assertEqual(len(enabled_files), 1)
        self.assertEqual(enabled_files[0].name, "sales.xlsx")
        self.assertEqual(
            enabled_files[0].local_path,
            store.load_original_path(structure.document_id),
        )
        self.assertEqual(disabled_files, ())
        self.assertEqual(knowledge_only_files, ())

    def test_debug_overview_exposes_complete_local_diagnostics(self) -> None:
        self.runtime_log_path.write_text(
            "line one\nline two\nline three\n",
            encoding="utf-8",
        )
        debug_record = self.debug_directory / "record.json"
        self.debug_directory.mkdir(parents=True)
        debug_record.write_text(
            (
                '{"id":"record-1","timestamp":"2026-07-23T00:00:00+00:00",'
                '"source":"chat","status":"completed","run_id":"run-1",'
                '"result":"answer","trace":[{"phase":"planning"}],'
                '"debug_runs":[{"label":"chat","prompt":"question",'
                '"steps":[{"observations":"raw evidence"}]}]}'
            ),
            encoding="utf-8",
        )

        response = self.client.get("/api/debug?limit=10&log_lines=2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["record_count"], 1)
        self.assertEqual(payload["source_counts"], {"chat": 1})
        self.assertEqual(payload["status_counts"], {"completed": 1})
        self.assertNotIn("debug_runs", payload["records"][0])
        self.assertEqual(payload["records"][0]["trace_events"], 1)
        detail = self.client.get("/api/debug/records/record-1")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["debug_runs"][0]["prompt"], "question")
        self.assertEqual(
            detail.json()["debug_runs"][0]["steps"][0]["observations"],
            "raw evidence",
        )
        self.assertEqual(payload["runtime_log"]["lines"], ["line two", "line three"])
        self.assertTrue(payload["model"]["model_online"])

    def test_analysis_rejects_malformed_section_scope(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Documents"},
        ).json()["id"]
        response = self.client.post(
            "/api/analysis",
            files={"files": ("sample.txt", b"content", "text/plain")},
            data={
                "conversation_id": conversation_id,
                "selected_sections": '{"document":"section-as-string"}',
            },
        )

        # 原因：宽松解析会把字符串 section id 拆成字符列表并绕过预期范围。
        # 作用：在创建 OrchestrationRequest 前拒绝形状错误的 multipart 字段。
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Invalid selected sections.")

    def test_upload_question_analysis_rejects_blank_before_reading_files(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Documents"},
        ).json()["id"]

        with (
            patch(
                "qwopus_agent.api.routes.analysis._read_uploads",
                new_callable=AsyncMock,
            ) as read_uploads,
            patch(
                "qwopus_agent.api.routes.analysis.AgentOrchestrator"
            ) as orchestrator,
        ):
            response = self.client.post(
                "/api/analysis",
                files={"files": ("sample.txt", b"content", "text/plain")},
                data={
                    "conversation_id": conversation_id,
                    "question": "   ",
                    "analysis_mode": "question",
                },
            )

        self.assertEqual(response.status_code, 422)
        read_uploads.assert_not_awaited()
        orchestrator.assert_not_called()

    def test_model_settings_switches_remote_endpoint_without_env_changes(self) -> None:
        response = self.client.put(
            "/api/model-settings",
            json={"mode": "remote", "base_url": "http://192.168.1.97:8001/v1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["base_url"], self.model_settings.base_url)
        self.model_runtime.configure_remote.assert_called_once_with(
            "http://192.168.1.97:8001/v1",
            ModelCapabilities(),
            timeout_seconds=120,
            max_retries=1,
            run_timeout_seconds=600,
        )

    def test_analysis_reads_uploaded_bytes_before_orchestration(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Documents"},
        ).json()["id"]
        structure = chunk_document_structure(
            build_document_structure("# Overview\nReal body", source="sample.txt")
        )
        orchestrator = MagicMock()
        orchestrator.run_sync.return_value = OrchestrationResult(
            success=True,
            final_answer="Analyzed",
            route="single_agent",
            analysis_result=AnalysisResult(
                markdown_summary="# Summary",
                markdown_document="# Overview\nReal body",
                document_structures=(structure,),
            ),
            debug_runs=(
                AgentDebugRun(
                    label="document",
                    prompt="Summarize",
                    max_steps=2,
                    state="success",
                    output="Analyzed",
                    steps=({"observations": "raw document"},),
                ),
            ),
        )

        # 原因：该测试只验证 FastAPI 上传边界，真实 MinerU/MiniRAG 由端到端测试负责。
        # 作用：防止 await 生成式再次把 UploadFile 变成不可迭代的 async_generator。
        with patch(
            "qwopus_agent.api.routes.analysis.AgentOrchestrator",
            return_value=orchestrator,
        ):
            response = self.client.post(
                "/api/analysis",
                files={"files": ("sample.txt", b"real bytes", "text/plain")},
                data={
                    "conversation_id": conversation_id,
                    "question": "Summarize",
                    "min_source_relevance": "0.8",
                    "analysis_mode": "section",
                    "selected_sections": (
                        f'{{"{structure.document_id}":["{structure.sections[0].id}"]}}'
                    ),
                },
            )

        self.assertEqual(response.status_code, 200)
        # 原因：raw debug 可能包含完整文件正文，只允许本机 Debug Console 读取。
        # 作用：锁定 FastAPI 正式响应不会把内部 debug_runs 暴露给 React 或其他客户端。
        self.assertNotIn("debug_runs", response.json())
        self.assertEqual(response.json()["documents"][0]["source"], "sample.txt")
        self.assertEqual(
            response.json()["documents"][0]["sections"][0]["title"],
            "Overview",
        )
        # 原因：正式响应必须脱敏，但独立 Console 仍要接收同一次文档运行的原始记录。
        # 作用：验证 API 边界同时满足“不外泄”和“可调试”两个方向。
        debug_records = load_debug_records(directory=self.debug_directory)
        self.assertEqual(debug_records[0]["source"], "document")
        self.assertEqual(
            debug_records[0]["debug_runs"][0]["steps"][0]["observations"],
            "raw document",
        )
        request = orchestrator.run_sync.call_args.args[0]
        self.assertEqual(request.conversation_id, conversation_id)
        self.assertEqual(request.uploaded_files[0].name, "sample.txt")
        self.assertEqual(request.uploaded_files[0].content, b"real bytes")
        self.assertEqual(request.min_source_relevance, 0.8)
        self.assertEqual(request.analysis_mode, "section")
        self.assertEqual(
            request.selected_sections[structure.document_id],
            (structure.sections[0].id,),
        )

    def test_analysis_view_exposes_bounded_workbook_structure(self) -> None:
        columns = [f"column-{index}" for index in range(45)]
        result = OrchestrationResult(
            success=True,
            final_answer="Analyzed",
            route="single_agent",
            analysis_result=AnalysisResult(
                markdown_summary="# Workbook",
                metadata={
                    "files": [
                        {
                            "file_name": "sales.xlsx",
                            "metadata": {
                                "source_type": "spreadsheet",
                                "workbook_profile": {
                                    "sheet_count": 1,
                                    "formula_count": 2,
                                    "merged_range_count": 1,
                                    "chart_count": 1,
                                    "image_count": 0,
                                    "data_validation_count": 3,
                                    "sheets": [
                                        {
                                            "name": "Data",
                                            "kind": "multi_table",
                                            "table_regions": [{}, {}],
                                            "formula_count": 2,
                                            "merged_range_count": 1,
                                            "chart_count": 1,
                                            "image_count": 0,
                                            "data_validation_count": 3,
                                        }
                                    ],
                                },
                                "analysis_tables": {
                                    "Data": {
                                        "rows": 10,
                                        "columns": 45,
                                        "column_names": columns,
                                    }
                                },
                            },
                        }
                    ]
                },
            ),
        )

        view = analysis_view(result)

        self.assertEqual(view.spreadsheets[0].source, "sales.xlsx")
        self.assertEqual(view.spreadsheets[0].sheets[0].region_count, 2)
        self.assertEqual(len(view.spreadsheets[0].tables[0].column_names), 40)
        self.assertTrue(view.spreadsheets[0].tables[0].columns_truncated)

    def test_analysis_rejects_unknown_conversation_scope(self) -> None:
        response = self.client.post(
            "/api/analysis",
            files={"files": ("sample.txt", b"content", "text/plain")},
            data={"conversation_id": "missing-conversation"},
        )

        # 原因：允许任意 scope 会让客户端绕过对话所有权并创建不可见知识库。
        # 作用：上传只接受 SQLite 中真实存在的 conversation_id。
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Conversation not found.")

    def test_analysis_rejects_unsupported_upload_type(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Documents"},
        ).json()["id"]

        response = self.client.post(
            "/api/analysis",
            files={"files": ("archive.zip", b"not a document", "application/zip")},
            data={"conversation_id": conversation_id, "question": "Summarize"},
        )

        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported upload type", response.json()["detail"])

    def test_local_folder_api_scans_and_analyzes_only_selected_files(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Local folder"},
        ).json()["id"]
        root = Path(self.temp_directory.name) / "source_folder"
        root.mkdir()
        selected = root / "selected.md"
        ignored = root / "ignored.md"
        selected.write_text("# Selected\nChosen evidence", encoding="utf-8")
        ignored.write_text("# Ignored\nOther evidence", encoding="utf-8")

        scan_response = self.client.post(
            "/api/local-folders/scan",
            json={"path": str(root)},
        )
        self.assertEqual(scan_response.status_code, 200)
        self.assertEqual(scan_response.json()["file_count"], 2)
        self.assertEqual(scan_response.json()["max_selection"], 100)

        structure = chunk_document_structure(
            build_document_structure(selected.read_text(), source="selected.md")
        )
        orchestrator = MagicMock()
        orchestrator.run_sync.return_value = OrchestrationResult(
            success=True,
            final_answer="Selected file analyzed.",
            route="single_agent",
            analysis_result=AnalysisResult(
                markdown_summary="# Selected",
                markdown_document="# Selected\nChosen evidence",
                document_structures=(structure,),
            ),
        )
        with patch(
            "qwopus_agent.api.routes.local_folders.AgentOrchestrator",
            return_value=orchestrator,
        ):
            response = self.client.post(
                "/api/local-folders/analyze",
                json={
                    "conversation_id": conversation_id,
                    "root": str(root),
                    "selected_files": ["selected.md"],
                    "question": "Summarize",
                },
            )

        self.assertEqual(response.status_code, 200)
        request = orchestrator.run_sync.call_args.args[0]
        self.assertEqual(len(request.uploaded_files), 1)
        self.assertEqual(request.uploaded_files[0].name, "selected.md")
        self.assertEqual(request.uploaded_files[0].local_path, selected.resolve())
        self.assertIsNone(request.uploaded_files[0].content)

    def test_local_folder_question_analysis_rejects_blank_before_file_access(
        self,
    ) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Local documents"},
        ).json()["id"]

        with (
            patch(
                "qwopus_agent.api.routes.local_folders.resolve_selected_files"
            ) as resolve_files,
            patch(
                "qwopus_agent.api.routes.local_folders.AgentOrchestrator"
            ) as orchestrator,
        ):
            response = self.client.post(
                "/api/local-folders/analyze",
                json={
                    "conversation_id": conversation_id,
                    "root": "/missing",
                    "selected_files": ["missing.md"],
                    "question": "",
                    "analysis_mode": "question",
                },
            )

        self.assertEqual(response.status_code, 422)
        resolve_files.assert_not_called()
        orchestrator.assert_not_called()

    def test_chat_rejects_global_knowledge_without_local_permission(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Knowledge"},
        ).json()["id"]

        response = self.client.post(
            f"/api/conversations/{conversation_id}/runs",
            json={
                "content": "Search global documents",
                "enable_local_knowledge": False,
                "include_global_knowledge": True,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_offline_model_before_starting_worker(self) -> None:
        conversation_id = self.client.post(
            "/api/conversations",
            json={"title": "Knowledge"},
        ).json()["id"]
        self.model_runtime.require_online_settings.side_effect = ModelRuntimeError(
            "host is down"
        )

        response = self.client.post(
            f"/api/conversations/{conversation_id}/runs",
            json={
                "content": "总结知识库",
                "enable_local_knowledge": True,
            },
        )

        # 原因：离线模型不是请求格式错误，也不应启动一个最终必然失败的 Agent 进程。
        # 作用：前端立即得到可展示的 503，且未成功的消息不会写入对话记录。
        self.assertEqual(response.status_code, 503)
        self.assertIn("Model service is unavailable", response.json()["detail"])
        self.assertEqual(
            self.client.get(
                f"/api/conversations/{conversation_id}/messages"
            ).json(),
            [],
        )


if __name__ == "__main__":
    unittest.main()
