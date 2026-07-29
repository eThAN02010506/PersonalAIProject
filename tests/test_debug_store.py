import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from qwopus_agent.api.debug_access import debug_host_is_allowed
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.runs import ChatRunRegistry
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.services.chat_service import ChatTaskResult
from qwopus_agent.services.skill_growth_service import (
    SkillGrowthPolicy,
    SkillGrowthService,
)
from qwopus_agent.skills import SkillCatalog, SkillRegistry
from qwopus_agent.utils.debug_store import (
    _prune_debug_records,
    append_debug_record,
    load_debug_records,
)


@dataclass(frozen=True)
class _DebugRun:
    label: str
    steps: tuple[dict[str, str], ...]


class _TraceEvent(BaseModel):
    phase: str
    status: str


class DebugStoreTests(unittest.TestCase):
    def test_debug_network_scope_is_always_host_only(self) -> None:
        # 原因：Debug 内容包含所有账号的 Prompt 和 Observation，管理员身份仍不足以允许远程读取。
        # 作用：锁定 IPv4/IPv6 回环可用，局域网与公网地址始终拒绝。
        self.assertTrue(debug_host_is_allowed("127.0.0.1"))
        self.assertTrue(debug_host_is_allowed("::1"))
        self.assertFalse(debug_host_is_allowed("192.168.1.42"))
        self.assertFalse(debug_host_is_allowed("fd00::42"))
        self.assertFalse(debug_host_is_allowed("8.8.8.8"))

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
            task.close.assert_called_once_with()

    def test_completed_chat_run_persists_resolved_task_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = ConversationRepository(
                root / "qwopus.db",
                import_legacy=False,
            )
            repository.initialize()
            conversation = repository.create_conversation()
            task = MagicMock()
            task.refresh_phase.return_value = "completed"
            task.poll_result.return_value = ChatTaskResult(
                status="completed",
                content="context analysis",
            )
            registry = ChatRunRegistry(repository, debug_directory=root / "debug")

            with patch("qwopus_agent.api.runs.start_chat_task", return_value=task):
                run_id = registry.start(
                    conversation.id,
                    "分析当前上下文方案",
                    SmolagentsModelSettings(
                        model_id="test",
                        base_url="http://local/v1",
                    ),
                    enable_web_search=False,
                    enable_local_knowledge=False,
                    response_detail="detailed",
                )
            registry.poll(run_id)
            memory = repository.get_memory(conversation.id)

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertEqual(memory.task_state.last_task_type, "analyze")
        self.assertEqual(
            memory.task_state.last_successful_objective,
            "分析当前上下文方案",
        )
        self.assertEqual(
            memory.task_state.last_answer_contract.response_detail,
            "detailed",
        )

    def test_completed_chat_tool_trace_creates_a_manual_skill_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = MagicMock()
            repository.list_messages.return_value = []
            task = MagicMock()
            task.refresh_phase.return_value = "completed"
            task.poll_result.return_value = ChatTaskResult(
                status="completed",
                content="A complete current web research answer.",
                trace=(
                    {
                        "phase": "tool_call",
                        "status": "completed",
                        "tool": "tavily_search",
                    },
                ),
            )
            skill_registry = SkillRegistry.discover(
                catalog=SkillCatalog(root / "catalog.json"),
                workflow_root=root / "workflows",
            )
            catalog = SkillCatalog(root / "catalog.json")
            growth = SkillGrowthService(
                registry=skill_registry,
                catalog=catalog,
                workflow_root=root / "workflows",
                history_path=root / "history.json",
                policy=SkillGrowthPolicy(
                    min_successes=1,
                    min_output_chars=1,
                    auto_promote=False,
                ),
            )
            registry = ChatRunRegistry(
                repository,
                debug_directory=root / "debug",
                skill_catalog=catalog,
                skill_growth=growth,
            )

            with patch("qwopus_agent.api.runs.start_chat_task", return_value=task):
                run_id = registry.start(
                    "conversation-1",
                    "research the current topic",
                    SmolagentsModelSettings(
                        model_id="test",
                        base_url="http://local/v1",
                    ),
                    enable_web_search=True,
                    enable_local_knowledge=False,
                )
            view = registry.poll(run_id)
            candidate = catalog.latest("learned_web_search")
            skill_names = skill_registry.list_names()

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.status, "candidate")
        self.assertNotIn("learned_web_search", skill_names)
        assert view is not None
        self.assertEqual(view.trace[-1]["phase"], "skill_growth")

    def test_debug_count_reaps_abandoned_terminal_run(self) -> None:
        repository = MagicMock()
        repository.list_messages.return_value = []
        task = MagicMock()
        task.refresh_phase.return_value = "completed"
        task.poll_result.return_value = ChatTaskResult(
            status="completed",
            content="reaped answer",
        )
        registry = ChatRunRegistry(repository)

        with patch("qwopus_agent.api.runs.start_chat_task", return_value=task):
            registry.start(
                "conversation-1",
                "question",
                SmolagentsModelSettings(model_id="test", base_url="http://local/v1"),
                enable_web_search=False,
                enable_local_knowledge=False,
            )

        # 原因：浏览器刷新后可能永远不会再次 poll 原 run_id。
        # 作用：Debug/维护调用仍收割终态、保存答案并释放 worker 资源。
        self.assertEqual(registry.debug_counts(), (0, 1))
        repository.add_message.assert_called_with(
            "conversation-1",
            "assistant",
            "reaped answer",
        )
        task.close.assert_called_once_with()

    def test_completed_run_cache_is_bounded(self) -> None:
        repository = MagicMock()
        repository.list_messages.return_value = []
        registry = ChatRunRegistry(repository, max_completed_runs=1)

        for answer in ("first", "second"):
            task = MagicMock()
            task.refresh_phase.return_value = "completed"
            task.poll_result.return_value = ChatTaskResult(
                status="completed",
                content=answer,
            )
            with patch("qwopus_agent.api.runs.start_chat_task", return_value=task):
                run_id = registry.start(
                    "conversation-1",
                    answer,
                    SmolagentsModelSettings(
                        model_id="test",
                        base_url="http://local/v1",
                    ),
                    enable_web_search=False,
                    enable_local_knowledge=False,
                )
            registry.poll(run_id)

        # 原因：对话答案已进入 SQLite，旧 RunView 不应无限常驻进程内存。
        # 作用：锁定容量裁剪只保留最近完成的轮询结果。
        self.assertEqual(registry.debug_counts(), (0, 1))

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

    def test_pruner_bounds_total_record_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for index in range(3):
                (directory / f"{index}.json").write_text("x" * 10, encoding="utf-8")

            # 原因：少数超大 Agent Trace 可以在 200 条数量限制内耗尽磁盘。
            # 作用：锁定清理器会保留较新的记录，并按总字节数淘汰最旧记录。
            _prune_debug_records(directory, keep=10, max_bytes=20)

            self.assertEqual(
                sorted(path.name for path in directory.glob("*.json")),
                ["1.json", "2.json"],
            )

    def test_pruner_removes_expired_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            expired = directory / "expired.json"
            recent = directory / "recent.json"
            expired.write_text("{}", encoding="utf-8")
            recent.write_text("{}", encoding="utf-8")
            old_timestamp = (
                datetime.now(UTC) - timedelta(days=2)
            ).timestamp()
            os.utime(expired, (old_timestamp, old_timestamp))

            # 原因：低频使用时记录数和字节数都可能未超限，但敏感调试内容仍会永久留存。
            # 作用：锁定超过保留期限的完整记录会被清理，近期记录不受影响。
            _prune_debug_records(
                directory,
                keep=10,
                max_bytes=1024,
                max_age=timedelta(days=1),
            )

            self.assertFalse(expired.exists())
            self.assertTrue(recent.exists())


if __name__ == "__main__":
    unittest.main()
