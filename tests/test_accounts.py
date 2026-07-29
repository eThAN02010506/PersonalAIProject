import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwopus_agent.api.app import create_app
from qwopus_agent.api.auth import (
    SESSION_COOKIE_NAME,
    AccountAuthMiddleware,
    AuthService,
)
from qwopus_agent.api.model_runtime import RuntimeModelStatus
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.documents import (
    DocumentStore,
    build_document_structure,
    chunk_document_structure,
)
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.utils.debug_store import append_debug_record
from tests.minirag_fakes import make_test_minirag


class AccountIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = ConversationRepository(
            self.root / "qwopus.db",
            import_legacy=False,
        )
        self.documents = DocumentStore(self.root / "documents")
        self.reports = self.root / "reports"
        settings = SmolagentsModelSettings(
            model_id="account-test-model",
            base_url="http://127.0.0.1:9999/v1",
        )
        runtime = MagicMock()
        runtime.current_settings.return_value = settings
        runtime.require_online_settings.return_value = settings
        runtime.status.return_value = RuntimeModelStatus(
            mode="remote",
            settings=settings,
            online=True,
            message="online",
        )
        self.app = create_app(
            repository=self.repository,
            knowledge_manager=ConversationKnowledgeManager(
                root=self.root / "minirag" / "conversations",
                factory=make_test_minirag,
            ),
            model_runtime=runtime,
            debug_directory=self.root / "debug",
            runtime_log_path=self.root / "runtime.log",
            document_store=self.documents,
            report_directory=self.reports,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_accounts_isolate_then_share_conversation_files_and_reports(self) -> None:
        denied = self.client.get("/api/conversations")
        self.assertEqual(denied.status_code, 401)

        bootstrap = self.client.post(
            "/api/auth/bootstrap",
            json={
                "username": "owner",
                "display_name": "Owner",
                "password": "owner-password-123",
            },
        )
        self.assertEqual(bootstrap.status_code, 201)
        owner = bootstrap.json()["user"]
        owner_token = self._take_cookie()

        member = self.client.post(
            "/api/users",
            headers=_session_header(owner_token),
            json={
                "username": "member",
                "display_name": "Member",
                "password": "member-password-123",
                "role": "member",
            },
        ).json()
        outsider = self.client.post(
            "/api/users",
            headers=_session_header(owner_token),
            json={
                "username": "outsider",
                "display_name": "Outsider",
                "password": "outsider-password-123",
                "role": "member",
            },
        ).json()
        member_token = self._login("member", "member-password-123")
        outsider_token = self._login("outsider", "outsider-password-123")

        conversation = self.client.post(
            "/api/conversations",
            headers=_session_header(owner_token),
            json={"title": "Private project"},
        ).json()
        conversation_id = conversation["id"]
        self.repository.add_message(conversation_id, "user", "owner-only fact")

        self.assertEqual(
            self.client.get(
                f"/api/conversations/{conversation_id}/messages",
                headers=_session_header(member_token),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                "/api/conversations",
                headers=_session_header(member_token),
            ).json(),
            [],
        )
        with patch("qwopus_agent.api.runs.start_chat_task") as start_task:
            start_task.return_value.refresh_phase.return_value = "executing"
            start_task.return_value.poll_result.return_value = None
            started = self.client.post(
                f"/api/conversations/{conversation_id}/runs",
                headers=_session_header(owner_token),
                json={"content": "Explain the project architecture."},
            )
        self.assertEqual(started.status_code, 200)
        run_id = started.json()["run_id"]
        self.assertEqual(
            self.client.get(
                f"/api/runs/{run_id}",
                headers=_session_header(member_token),
            ).status_code,
            404,
        )

        shared = self.client.post(
            f"/api/conversations/{conversation_id}/members",
            headers=_session_header(owner_token),
            json={"username": "member"},
        )
        self.assertEqual(shared.status_code, 201)
        self.assertEqual(shared.json()["user_id"], member["id"])
        member_conversations = self.client.get(
            "/api/conversations",
            headers=_session_header(member_token),
        ).json()
        self.assertEqual(member_conversations[0]["id"], conversation_id)
        self.assertFalse(member_conversations[0]["is_owner"])
        self.assertEqual(
            self.client.get(
                f"/api/runs/{run_id}",
                headers=_session_header(member_token),
            ).status_code,
            200,
        )
        member_messages = self.client.get(
            f"/api/conversations/{conversation_id}/messages",
            headers=_session_header(member_token),
        )
        self.assertEqual(member_messages.json()[0]["content"], "owner-only fact")
        self.assertEqual(
            self.client.patch(
                f"/api/conversations/{conversation_id}",
                headers=_session_header(member_token),
                json={"title": "Unauthorized rename"},
            ).status_code,
            404,
        )

        saved = self._save_document("shared.md", "shared-file-fact")
        self.repository.register_document(
            saved.document_id,
            owner_user_id=owner["id"],
            conversation_id=conversation_id,
        )
        self.assertEqual(
            [item["document_id"] for item in self.client.get(
                "/api/documents",
                headers=_session_header(member_token),
            ).json()],
            [saved.document_id],
        )
        self.assertEqual(
            self.client.get(
                "/api/documents",
                headers=_session_header(outsider_token),
            ).json(),
            [],
        )

        self.reports.mkdir()
        report = self.reports / "shared-report.md"
        report.write_text("# Shared report", encoding="utf-8")
        self.repository.register_report(
            report.name,
            created_by_user_id=owner["id"],
            conversation_id=conversation_id,
        )
        self.assertEqual(
            self.client.get(
                f"/api/reports/{report.name}",
                headers=_session_header(member_token),
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                f"/api/reports/{report.name}",
                headers=_session_header(outsider_token),
            ).status_code,
            404,
        )

        with patch("qwopus_agent.api.runs.start_chat_task") as start_task:
            start_task.return_value.refresh_phase.return_value = "executing"
            start_task.return_value.poll_result.return_value = None
            member_started = self.client.post(
                f"/api/conversations/{conversation_id}/runs",
                headers=_session_header(member_token),
                json={
                    "content": (
                        "Read the attached shared.md file, quote the exact "
                        "shared-file-fact phrase, and explain its meaning."
                    )
                },
            )
            start_task.assert_called_once()
        self.assertEqual(member_started.status_code, 200)
        member_run_id = member_started.json()["run_id"]

        revoked = self.client.delete(
            f"/api/conversations/{conversation_id}/members/{member['id']}",
            headers=_session_header(owner_token),
        )
        self.assertEqual(revoked.status_code, 204)
        # 原因：撤销共享只应停止已失去权限的成员任务，不能中断所有者自己的任务。
        # 作用：同时锁定即时撤权与最小权限影响范围。
        self.assertEqual(self.app.state.runs.poll(member_run_id).status, "cancelled")
        self.assertEqual(self.app.state.runs.poll(run_id).status, "running")
        self.assertEqual(
            self.client.get(
                f"/api/conversations/{conversation_id}/messages",
                headers=_session_header(member_token),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                "/api/documents",
                headers=_session_header(member_token),
            ).json(),
            [],
        )
        self.assertEqual(
            self.client.get(
                f"/api/reports/{report.name}",
                headers=_session_header(member_token),
            ).status_code,
            404,
        )

        append_debug_record(
            source="chat",
            status="completed",
            result="member audit result",
            trace=(),
            debug_runs=(),
            run_id="member-audit-run",
            user_id=member["id"],
            username=member["username"],
            directory=self.root / "debug",
        )
        member_debug = self.client.get(
            "/api/debug",
            headers=_session_header(member_token),
        )
        administrator_debug = self.client.get(
            "/api/debug",
            headers=_session_header(owner_token),
        )
        # 原因：全局审计若过滤成当前管理员自身记录，就无法排查其他使用者的 Agent 运行。
        # 作用：普通成员仍被拒绝，主机管理员则能看到记录对应的真实账号。
        self.assertEqual(member_debug.status_code, 403)
        self.assertEqual(administrator_debug.status_code, 200)
        audit_record = next(
            record
            for record in administrator_debug.json()["records"]
            if record["run_id"] == "member-audit-run"
        )
        self.assertEqual(audit_record["username"], "member")
        self.assertEqual(outsider["role"], "member")

        with sqlite3.connect(self.repository.database_path) as connection:
            password_hash = connection.execute(
                "SELECT password_hash FROM users WHERE id = ?",
                (owner["id"],),
            ).fetchone()[0]
            stored_token = connection.execute(
                "SELECT token_hash FROM sessions WHERE user_id = ?",
                (owner["id"],),
            ).fetchone()[0]
        self.assertNotIn("owner-password-123", password_hash)
        self.assertTrue(password_hash.startswith("$argon2id$"))
        self.assertNotEqual(stored_token, owner_token)

    def test_first_admin_claims_legacy_conversations_and_files(self) -> None:
        legacy_conversation = self.repository.create_conversation("Legacy")
        saved = self._save_document("legacy.md", "legacy fact")
        self.reports.mkdir()
        (self.reports / "legacy.md").write_text("# Legacy", encoding="utf-8")

        bootstrap = self.client.post(
            "/api/auth/bootstrap",
            json={
                "username": "legacy-admin",
                "display_name": "Legacy Admin",
                "password": "legacy-password-123",
            },
        )

        self.assertEqual(bootstrap.status_code, 201)
        conversations = self.client.get("/api/conversations").json()
        documents = self.client.get("/api/documents").json()
        self.assertEqual(conversations[0]["id"], legacy_conversation.id)
        self.assertTrue(conversations[0]["is_owner"])
        self.assertEqual(documents[0]["document_id"], saved.document_id)
        self.assertEqual(
            self.client.get("/api/reports/legacy.md").status_code,
            200,
        )

    def test_administrator_creates_member_and_admin_accounts(self) -> None:
        bootstrap = self.client.post(
            "/api/auth/bootstrap",
            json={
                "username": "root-admin",
                "display_name": "Root Admin",
                "password": "root-password-123",
            },
        )
        self.assertEqual(bootstrap.status_code, 201)
        administrator_token = self._take_cookie()

        created_accounts = []
        for username, role in (("new-member", "member"), ("new-admin", "admin")):
            created = self.client.post(
                "/api/users",
                headers=_session_header(administrator_token),
                json={
                    "username": username,
                    "display_name": username.replace("-", " ").title(),
                    "password": f"{username}-password",
                    "role": role,
                },
            )
            self.assertEqual(created.status_code, 201)
            created_accounts.append(created.json())

        # 原因：账号创建是管理员操作，开放给普通账号会绕过本地邀请式开户。
        # 作用：同时锁定两种可创建角色，以及普通账号无法继续扩散账号权限。
        self.assertEqual(
            [account["role"] for account in created_accounts],
            ["member", "admin"],
        )
        member_token = self._login("new-member", "new-member-password")
        denied = self.client.post(
            "/api/users",
            headers=_session_header(member_token),
            json={
                "username": "unauthorized",
                "display_name": "Unauthorized",
                "password": "unauthorized-password",
                "role": "member",
            },
        )
        self.assertEqual(denied.status_code, 403)

    def test_debug_page_rejects_non_loopback_even_for_an_admin_session(self) -> None:
        repository = ConversationRepository(
            self.root / "debug-host.db",
            import_legacy=False,
        )
        repository.initialize()
        auth = AuthService(repository)
        administrator = auth.bootstrap(
            username="host-admin",
            display_name="Host Admin",
            password="host-password-123",
        )
        self.assertIsNotNone(administrator)
        assert administrator is not None
        grant = auth.issue_session(administrator)
        app = FastAPI()
        app.add_middleware(AccountAuthMiddleware, auth=auth)

        @app.get("/debug")
        def debug() -> dict[str, str]:
            return {"status": "visible"}

        with TestClient(app, client=("192.168.1.44", 53000)) as lan_client:
            response = lan_client.get(
                "/debug",
                headers=_session_header(grant.token),
            )

        self.assertEqual(response.status_code, 403)
        self.assertIn("host machine", response.json()["detail"])

    def _login(self, username: str, password: str) -> str:
        response = self.client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(response.status_code, 200)
        return self._take_cookie()

    def _take_cookie(self) -> str:
        token = self.client.cookies.get(SESSION_COOKIE_NAME)
        self.assertIsNotNone(token)
        self.client.cookies.clear()
        return str(token)

    def _save_document(self, source: str, body: str):
        original = self.root / source
        markdown = f"# {source}\n\n{body}"
        original.write_text(markdown, encoding="utf-8")
        structure = chunk_document_structure(
            build_document_structure(markdown, source=source)
        )
        self.documents.persist(
            original_path=original,
            markdown=markdown,
            structure=structure,
            metadata={"parser": "markdown"},
        )
        return structure


def _session_header(token: str) -> dict[str, str]:
    return {"Cookie": f"{SESSION_COOKIE_NAME}={token}"}


if __name__ == "__main__":
    unittest.main()
