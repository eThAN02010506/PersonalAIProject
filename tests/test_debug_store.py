import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from qwopus_agent.api.runs import ChatRunRegistry
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.services.chat_service import ChatTaskResult
from qwopus_agent.utils.debug_store import append_debug_record, load_debug_records


@dataclass(frozen=True)
class _DebugRun:
    label: str
    steps: tuple[dict[str, str], ...]


class _TraceEvent(BaseModel):
    phase: str
    status: str


class DebugStoreTests(unittest.TestCase):
    def test_round_trip_preserves_nested_raw_agent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            # 原因：Console 必须读取 FastAPI 进程写出的 dataclass 与 Pydantic 混合结果。
            # 作用：锁定 Prompt、Observation 和编排事件经过磁盘后仍保持结构化字段。
            path = append_debug_record(
                source="chat",
                status="completed",
                result="final answer",
                trace=(_TraceEvent(phase="tool_call", status="completed"),),
                debug_runs=(
                    _DebugRun(label="chat", steps=({"observations": "raw evidence"},)),
                ),
                run_id="run-1",
                directory=directory,
            )

            self.assertIsNotNone(path)
            records = load_debug_records(directory=directory)
            self.assertEqual(records[0]["run_id"], "run-1")
            self.assertEqual(records[0]["trace"][0]["phase"], "tool_call")
            self.assertEqual(
                records[0]["debug_runs"][0]["steps"][0]["observations"],
                "raw evidence",
            )

    def test_loader_ignores_partial_or_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "new.json").write_text(
                json.dumps({"id": "valid"}), encoding="utf-8"
            )
            (directory / "old.json").write_text("{", encoding="utf-8")
            (directory / ".pending.tmp").write_text("partial", encoding="utf-8")

            records = load_debug_records(directory=directory)

            self.assertEqual(records, [{"id": "valid"}])

    def test_completed_chat_run_is_written_for_the_console(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = MagicMock()
            repository.list_messages.return_value = []
            task = MagicMock()
            task.refresh_phase.return_value = "completed"
            task.poll_result.return_value = ChatTaskResult(
                status="completed",
                content="final answer",
                trace=({"phase": "tool_call", "status": "completed"},),
                debug_runs=({"label": "chat", "steps": [{"observations": "raw"}]},),
            )
            registry = ChatRunRegistry(
                repository,
                debug_directory=Path(temporary_directory),
            )

            # 原因：正式聊天在 worker 子进程完成，只有 API poll 回调能拿到完整 raw runs。
            # 作用：锁定该回调会写入 Console 目录，同时公开 RunView 仍不含 debug_runs。
            with patch("qwopus_agent.api.runs.start_chat_task", return_value=task):
                run_id = registry.start(
                    "conversation-1",
                    "question",
                    SmolagentsModelSettings(model_id="test", base_url="http://local/v1"),
                    enable_web_search=False,
                    enable_local_knowledge=False,
                )
            view = registry.poll(run_id)

            self.assertIsNotNone(view)
            self.assertNotIn("debug_runs", view.model_dump())
            records = load_debug_records(directory=Path(temporary_directory))
            self.assertEqual(records[0]["run_id"], run_id)
            self.assertEqual(
                records[0]["debug_runs"][0]["steps"][0]["observations"],
                "raw",
            )

    def test_writer_prunes_only_old_complete_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for index in range(201):
                (directory / f"{index:03}.json").write_text("{}", encoding="utf-8")
            pending = directory / ".active.tmp"
            pending.write_text("partial", encoding="utf-8")

            append_debug_record(
                source="chat",
                status="completed",
                trace=(),
                debug_runs=(),
                directory=directory,
            )

            self.assertEqual(len(list(directory.glob("*.json"))), 200)
            self.assertTrue(pending.exists())


if __name__ == "__main__":
    unittest.main()
