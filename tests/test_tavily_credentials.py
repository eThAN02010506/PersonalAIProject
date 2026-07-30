import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwopus_agent.integrations.tavily_credentials import (
    TavilyCredentialError,
    TavilyCredentialStore,
    resolve_tavily_api_key,
)


class TavilyCredentialStoreTests(unittest.TestCase):
    def test_managed_key_is_persistent_masked_and_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = TavilyCredentialStore(
                path=root / "secrets" / "tavily.key",
                legacy_env_path=root / "missing.env",
            )
            key = "tvly-test-managed-123456"

            with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
                status = store.save(key)
                reloaded = TavilyCredentialStore(
                    path=store.path,
                    legacy_env_path=store.legacy_env_path,
                )

                self.assertEqual(reloaded.resolve(), key)
                self.assertTrue(status.configured)
                self.assertEqual(status.source, "managed")
                self.assertNotEqual(status.masked_key, key)
                self.assertNotIn(key, status.masked_key or "")
                self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(store.path.parent.stat().st_mode & 0o777, 0o700)

    def test_delete_falls_back_to_environment_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = TavilyCredentialStore(
                path=root / "secrets" / "tavily.key",
                legacy_env_path=root / "missing.env",
            )
            with patch.dict(
                os.environ,
                {"TAVILY_API_KEY": "tvly-environment-123456"},
            ):
                store.save("tvly-managed-123456")
                status = store.delete()

                self.assertFalse(store.path.exists())
                self.assertTrue(status.configured)
                self.assertEqual(status.source, "environment")

    def test_invalid_key_is_rejected_without_creating_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "secrets" / "tavily.key"
            store = TavilyCredentialStore(
                path=path,
                legacy_env_path=Path(temporary_directory) / "missing.env",
            )

            with self.assertRaises(TavilyCredentialError):
                store.save("bad key")

            self.assertFalse(path.exists())

    def test_process_resolver_observes_key_rotation_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "secrets" / "tavily.key"
            missing_env = root / "missing.env"
            with (
                patch(
                    "qwopus_agent.integrations.tavily_credentials.DEFAULT_TAVILY_KEY_PATH",
                    path,
                ),
                patch(
                    "qwopus_agent.integrations.tavily_credentials.DEFAULT_LEGACY_ENV_PATH",
                    missing_env,
                ),
                patch.dict(os.environ, {"TAVILY_API_KEY": ""}),
            ):
                store = TavilyCredentialStore()
                store.save("tvly-first-key-123456")
                first = resolve_tavily_api_key()
                store.save("tvly-rotated-key-654321")
                second = resolve_tavily_api_key()

            # 原因：联网 Tool 运行在长寿命进程中，缓存启动时的 Key 会让 UI 轮换失效。
            # 作用：锁定每次调用都重新解析托管文件，无需重启 FastAPI 或 Agent worker。
            self.assertEqual(first, "tvly-first-key-123456")
            self.assertEqual(second, "tvly-rotated-key-654321")


if __name__ == "__main__":
    unittest.main()
