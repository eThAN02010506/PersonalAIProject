import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qwopus_agent.api.model_runtime import (
    ModelRuntimeError,
    RuntimeModelController,
    _find_mlx_server,
    _normalize_base_url,
    _validate_model_path,
)
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.llm import ModelCapabilities


class ModelRuntimeTests(unittest.TestCase):
    def test_remote_url_is_normalized_to_openai_v1_root(self) -> None:
        self.assertEqual(
            _normalize_base_url("http://192.168.1.97:8001"),
            "http://192.168.1.97:8001/v1",
        )
        with self.assertRaises(ModelRuntimeError):
            _normalize_base_url("192.168.1.97:8001")

    @patch(
        "qwopus_agent.api.model_runtime.probe_model_settings",
        side_effect=lambda settings: (settings, False, "host is down"),
    )
    def test_require_online_settings_rejects_offline_endpoint(
        self,
        _check: MagicMock,
    ) -> None:
        controller = RuntimeModelController(
            SmolagentsModelSettings(
                model_id="offline-model",
                base_url="http://offline.example/v1",
            )
        )

        # 原因：离线状态不能继续创建后台 Agent，否则用户只能在超时后看到 provider 异常。
        # 作用：锁定 fail-fast 合同，并确认错误保留实际连接诊断。
        with self.assertRaisesRegex(ModelRuntimeError, "host is down"):
            controller.require_online_settings()

    @patch(
        "qwopus_agent.api.model_runtime.probe_model_settings",
        side_effect=lambda settings: (settings, False, "offline"),
    )
    def test_status_reuses_short_lived_probe(
        self,
        check: MagicMock,
    ) -> None:
        controller = RuntimeModelController(
            SmolagentsModelSettings(
                model_id="offline-model",
                base_url="http://offline.example/v1",
            )
        )

        first = controller.status()
        second = controller.status()

        self.assertIs(first, second)
        check.assert_called_once()

    def test_model_parent_venv_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            executable = root / ".venv/bin/mlx_lm.server"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)

            self.assertEqual(_find_mlx_server(model), executable.resolve())

    def test_local_model_requires_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ModelRuntimeError):
                _validate_model_path(str(path))
            (path / "model.safetensors").write_bytes(b"weights")
            self.assertEqual(_validate_model_path(str(path)), path.resolve())

    @patch(
        "qwopus_agent.api.model_runtime.probe_model_settings",
        side_effect=lambda value: (value, True, "online"),
    )
    def test_remote_settings_replace_only_runtime_address(
        self,
        probe: MagicMock,
    ) -> None:
        settings = SmolagentsModelSettings(
            model_id="current",
            base_url="http://old.example/v1",
        )
        controller = RuntimeModelController(settings)

        status = controller.configure_remote(
            "http://new.example:9000",
            ModelCapabilities(context_window_tokens=65536, agent_mode="code"),
        )

        self.assertEqual(status.settings.base_url, "http://new.example:9000/v1")
        self.assertEqual(status.settings.context_window_tokens, 65536)
        self.assertEqual(status.settings.capabilities.agent_mode, "code")
        self.assertEqual(status.mode, "remote")
        probe.assert_called_once()

    @patch(
        "qwopus_agent.api.model_runtime.probe_model_settings",
        side_effect=lambda value: (value, True, "online"),
    )
    @patch("qwopus_agent.api.model_runtime._wait_for_server")
    @patch("qwopus_agent.api.model_runtime._available_port", return_value=18080)
    def test_local_model_starts_discovered_mlx_server_without_shell(
        self,
        _port: MagicMock,
        wait_for_server: MagicMock,
        _probe: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "local-model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            executable = root / ".venv/bin/mlx_lm.server"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            process = MagicMock()
            process.poll.return_value = None
            controller = RuntimeModelController(
                SmolagentsModelSettings(
                    model_id="remote",
                    base_url="http://remote.example/v1",
                )
            )

            with patch.object(controller, "_start_local_process", return_value=process) as start:
                status = controller.configure_local(str(model))

            start.assert_called_once_with(executable.resolve(), model.resolve(), 18080)
            wait_for_server.assert_called_once()
            self.assertEqual(status.mode, "local")
            self.assertEqual(status.settings.base_url, "http://127.0.0.1:18080/v1")
            self.assertEqual(status.local_model_path, str(model.resolve()))


if __name__ == "__main__":
    unittest.main()
