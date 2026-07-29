import base64
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwopus_agent.api.app import create_app
from qwopus_agent.api.lan_auth import LanAuthConfig, LanAuthMiddleware
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.documents import DocumentStore
from qwopus_agent.memory import ConversationKnowledgeManager

_LAN_CLIENT = ("192.168.1.50", 50000)


def _authorization(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {encoded}"


def _minimal_app(config: LanAuthConfig) -> FastAPI:
    app = FastAPI()
    app.add_middleware(LanAuthMiddleware, config=config)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


class LanAuthTests(unittest.TestCase):
    def test_loopback_client_remains_available_without_credentials(self) -> None:
        with TestClient(_minimal_app(LanAuthConfig())) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_lan_fails_closed_when_password_is_not_configured(self) -> None:
        with TestClient(
            _minimal_app(LanAuthConfig()),
            client=_LAN_CLIENT,
        ) as client:
            response = client.get("/health")

        # 原因：仅监听私网并不能保证同一网络中的客户端可信。
        # 作用：证明缺少密码时非本机入口不会退回匿名访问。
        self.assertEqual(response.status_code, 503)
        self.assertIn("QWOPUS_LAN_PASSWORD", response.json()["detail"])

    def test_lan_rejects_wrong_credentials_and_challenges_browser(self) -> None:
        with TestClient(
            _minimal_app(LanAuthConfig(username="qwopus", password="correct")),
            client=_LAN_CLIENT,
        ) as client:
            response = client.get(
                "/health",
                headers={"Authorization": _authorization("qwopus", "wrong")},
            )

        self.assertEqual(response.status_code, 401)
        self.assertIn("Basic", response.headers["www-authenticate"])

    def test_lan_accepts_utf8_credentials(self) -> None:
        config = LanAuthConfig(username="用户", password="本地密码")
        with TestClient(
            _minimal_app(config),
            client=_LAN_CLIENT,
        ) as client:
            response = client.get(
                "/health",
                headers={"Authorization": _authorization("用户", "本地密码")},
            )

        self.assertEqual(response.status_code, 200)

    def test_create_app_applies_auth_to_the_complete_http_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                repository=ConversationRepository(
                    root / "qwopus.db",
                    import_legacy=False,
                ),
                knowledge_manager=ConversationKnowledgeManager(
                    root=root / "minirag",
                ),
                document_store=DocumentStore(root / "documents"),
                runtime_log_path=root / "qwopus.log",
                lan_auth=LanAuthConfig(password="private-lan"),
            )
            with TestClient(app, client=_LAN_CLIENT) as client:
                denied = client.get("/openapi.json")
                allowed = client.get(
                    "/openapi.json",
                    headers={
                        "Authorization": _authorization(
                            "qwopus",
                            "private-lan",
                        )
                    },
                )

        # 原因：独立中间件测试不能证明 create_app 真的完成了生产装配。
        # 作用：锁定 API 文档和同进程 React 入口之前存在同一认证边界。
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
