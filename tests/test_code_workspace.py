import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qwopus_agent.code_workspace.models import (
    CodeChatMessage,
    CodeWorkspaceAgentRun,
)
from qwopus_agent.code_workspace.repository import CodeChangeRepository
from qwopus_agent.code_workspace.security import (
    CodeWorkspaceError,
    read_code_file,
    scan_code_workspace,
    search_code_workspace,
)
from qwopus_agent.integrations.smolagents_code_workspace import (
    run_smolagents_code_workspace_chat,
)
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.llm import BaseLLM, ChatMessage, LLMResponse
from qwopus_agent.services.code_workspace_service import CodeWorkspaceService


class ProposalLLM(BaseLLM):
    """Return one deterministic exact-replacement proposal."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.messages = messages
        return LLMResponse(
            content=(
                '{"summary":"Improve greeting","reason":"Use a clearer greeting.",'
                '"verification_plan":["Run Python tests"],'
                '"changes":[{"path":"src/example.py","replacements":['
                '{"old_text":"return \\"hello\\"","new_text":"return \\"hello world\\""}]}]}'
            ),
            model="proposal-test-model",
        )


class CodeConversationRunner:
    """Return one grounded result from a simulated read-only Agent loop."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str], list[str]]] = []

    def __call__(
        self,
        root: str,
        transcript: str,
        eligible_paths: list[str],
        selected_files: list[str],
    ) -> CodeWorkspaceAgentRun:
        self.calls.append((root, transcript, eligible_paths, selected_files))
        return CodeWorkspaceAgentRun(
            content=(
                '{"mode":"ready",'
                '"message":"I inspected src/example.py: greet returns a fixed value. '
                'I can clarify it while preserving the function signature and verify it '
                'with the existing Python tests.",'
                '"objective":"Clarify the greeting without changing its public signature.",'
                '"selected_files":["src/example.py"]}'
            ),
            inspected_files=["src/example.py"],
            tool_calls=["code_search", "code_read"],
            state="success",
        )


class CodeWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "src").mkdir()
        self.source = self.root / "src" / "example.py"
        self.source.write_text(
            'def greet() -> str:\n    return "hello"\n',
            encoding="utf-8",
        )
        (self.root / "tests").mkdir()
        self.context_source = self.root / "tests" / "test_example.py"
        self.context_source.write_text(
            'def test_greet() -> None:\n    assert greet() == "hello world"\n',
            encoding="utf-8",
        )
        (self.root / ".env").write_text("SECRET=hidden", encoding="utf-8")
        (self.root / "secrets.toml").write_text('token = "hidden"', encoding="utf-8")
        self.repository = CodeChangeRepository(self.root / ".changes")
        self.service = CodeWorkspaceService(
            self.repository,
            llm_factory=ProposalLLM,
            code_chat_runner=CodeConversationRunner(),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_scan_read_and_search_hide_sensitive_files(self) -> None:
        tree = scan_code_workspace(self.root)
        serialized = tree.model_dump_json()
        view = read_code_file(self.root, "src/example.py")
        matches = search_code_workspace(self.root, "return")

        self.assertEqual(tree.file_count, 2)
        self.assertNotIn(".env", serialized)
        self.assertNotIn("secrets.toml", serialized)
        self.assertIn('return "hello"', view.content)
        self.assertEqual(matches[0].path, "src/example.py")
        with self.assertRaises(CodeWorkspaceError):
            read_code_file(self.root, "../outside.py")
        with self.assertRaises(CodeWorkspaceError):
            read_code_file(self.root, ".env")
        with self.assertRaises(CodeWorkspaceError):
            read_code_file(self.root, "secrets.toml")

    def test_proposal_requires_approval_and_can_be_rolled_back(self) -> None:
        proposal = self.service.propose(
            root=str(self.root),
            objective="Make the greeting clearer.",
            selected_files=["src/example.py"],
            owner_user_id="admin",
        )

        self.assertEqual(proposal.status, "proposed")
        self.assertIn('-    return "hello"', proposal.unified_diff)
        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            'def greet() -> str:\n    return "hello"\n',
        )

        applied = self.service.apply(proposal.id, "admin")
        self.assertEqual(applied.status, "applied")
        self.assertIn("hello world", self.source.read_text(encoding="utf-8"))

        rolled_back = self.service.rollback(proposal.id, "admin")
        self.assertEqual(rolled_back.status, "rolled_back")
        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            'def greet() -> str:\n    return "hello"\n',
        )

    def test_abstract_request_is_grounded_through_code_conversation(self) -> None:
        runner = CodeConversationRunner()
        service = CodeWorkspaceService(
            self.repository,
            llm_factory=ProposalLLM,
            code_chat_runner=runner,
        )

        reply = service.chat(
            root=str(self.root),
            message="Make the greeting nicer without changing its public API.",
            history=[
                CodeChatMessage(
                    role="assistant",
                    content="I will inspect the relevant implementation first.",
                )
            ],
            selected_files=[],
        )

        self.assertEqual(reply.mode, "ready")
        self.assertEqual(reply.selected_files, ["src/example.py"])
        self.assertEqual(reply.inspected_files, ["src/example.py"])
        self.assertIn("src/example.py", runner.calls[0][2])
        self.assertIn("Make the greeting nicer", runner.calls[0][1])
        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            'def greet() -> str:\n    return "hello"\n',
        )

    def test_code_conversation_downgrades_uninspected_ready_answer(self) -> None:
        def ungrounded_runner(
            _root: str,
            _transcript: str,
            _eligible_paths: list[str],
            _selected_files: list[str],
        ) -> CodeWorkspaceAgentRun:
            return CodeWorkspaceAgentRun(
                content=(
                    '{"mode":"ready","message":"This answer names a file but the Agent '
                    'did not actually read it, so it must not enter the proposal stage.",'
                    '"objective":"Change the greeting.",'
                    '"selected_files":["src/example.py"]}'
                ),
                inspected_files=[],
            )

        service = CodeWorkspaceService(
            self.repository,
            llm_factory=ProposalLLM,
            code_chat_runner=ungrounded_runner,
        )

        reply = service.chat(
            root=str(self.root),
            message="Improve the greeting while preserving compatibility.",
            history=[],
            selected_files=[],
        )

        self.assertEqual(reply.mode, "answer")
        self.assertIsNone(reply.objective)
        self.assertEqual(reply.selected_files, [])

    def test_code_conversation_uses_grounded_message_when_objective_is_missing(
        self,
    ) -> None:
        def missing_objective_runner(
            _root: str,
            _transcript: str,
            _eligible_paths: list[str],
            _selected_files: list[str],
        ) -> CodeWorkspaceAgentRun:
            return CodeWorkspaceAgentRun(
                content=(
                    '{"mode":"ready","message":"The inspected greeting function returns '
                    'a fixed string. Update only that return value while preserving the '
                    'public signature and verify it with the existing tests.",'
                    '"objective":null,"selected_files":["src/example.py"]}'
                ),
                inspected_files=["src/example.py"],
            )

        service = CodeWorkspaceService(
            self.repository,
            llm_factory=ProposalLLM,
            code_chat_runner=missing_objective_runner,
        )

        reply = service.chat(
            root=str(self.root),
            message="Improve the greeting without changing its public API.",
            history=[],
            selected_files=[],
        )

        self.assertEqual(reply.mode, "ready")
        self.assertEqual(reply.objective, reply.message)

    def test_smolagents_code_runner_uses_registered_search_and_read_skills(self) -> None:
        captured: dict[str, object] = {}

        class InspectingAgent:
            def __init__(self, tools: list[object]) -> None:
                self.tools = {tool.name: tool for tool in tools}  # type: ignore[attr-defined]

            def run(self, prompt: str, **kwargs: object) -> SimpleNamespace:
                captured["prompt"] = prompt
                captured["run_kwargs"] = kwargs
                search_output = self.tools["code_search"].forward("greet")
                read_output = self.tools["code_read"].forward("src/example.py", 1)
                captured["search_output"] = search_output
                captured["read_output"] = read_output
                try:
                    self.tools["code_search"].forward("  GREET  ")
                except CodeWorkspaceError as exc:
                    captured["duplicate_search_error"] = str(exc)
                try:
                    self.tools["code_read"].forward("src/example.py", 2)
                except CodeWorkspaceError as exc:
                    captured["duplicate_read_error"] = str(exc)
                return SimpleNamespace(
                    output={
                        "mode": "ready",
                        "message": (
                            "I inspected src/example.py and found that greet returns a fixed "
                            "string. The change can preserve its signature and be verified "
                            "with the existing tests."
                        ),
                        "objective": "Clarify the returned greeting.",
                        "selected_files": ["src/example.py"],
                    },
                    state="success",
                    steps=[
                        {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "code_search",
                                        "arguments": '{"query":"greet"}',
                                    }
                                }
                            ],
                            "observations": search_output,
                        },
                        {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "code_read",
                                        "arguments": (
                                            '{"path":"src/example.py","start_line":1}'
                                        ),
                                    }
                                }
                            ],
                            "observations": read_output,
                        },
                    ],
                )

        def build_agent(**kwargs: object) -> InspectingAgent:
            captured["agent_kwargs"] = kwargs
            return InspectingAgent(kwargs["tools"])  # type: ignore[arg-type]

        with patch(
            "qwopus_agent.integrations.smolagents_code_workspace."
            "build_smolagents_tool_calling_agent",
            side_effect=build_agent,
        ):
            result = run_smolagents_code_workspace_chat(
                str(self.root),
                "USER: Make the greeting clearer.",
                ["src/example.py"],
                [],
                settings=SmolagentsModelSettings(
                    model_id="test-model",
                    base_url="http://127.0.0.1:9999/v1",
                ),
            )

        self.assertEqual(result.inspected_files, ["src/example.py"])
        self.assertEqual(result.tool_calls, ["code_search", "code_read"])
        self.assertIn('"path": "src/example.py"', str(captured["search_output"]))
        self.assertIn('return "hello"', str(captured["read_output"]))
        self.assertIn("already inspected", str(captured["duplicate_read_error"]))
        self.assertEqual(captured["agent_kwargs"]["planning_interval"], 11)  # type: ignore[index]
        self.assertIn("AVAILABLE SOURCE PATHS", str(captured["prompt"]))
        self.assertIn("shared Planner", str(captured["prompt"]))
        self.assertIn("service and its tests before UI", str(captured["prompt"]))
        self.assertIn("already searched", str(captured["duplicate_search_error"]))

    def test_proposal_reads_context_without_granting_write_access(self) -> None:
        proposal_llm = ProposalLLM()
        service = CodeWorkspaceService(
            self.repository,
            llm_factory=lambda: proposal_llm,
            code_chat_runner=CodeConversationRunner(),
        )

        proposal = service.propose(
            root=str(self.root),
            objective="Match the existing test contract.",
            selected_files=["src/example.py"],
            context_files=["tests/test_example.py"],
            owner_user_id="admin",
        )

        prompt = proposal_llm.messages[1].content
        self.assertEqual(proposal.changed_files, ["src/example.py"])
        self.assertIn("EDITABLE SOURCE FILES", prompt)
        self.assertIn("READ-ONLY CONTEXT FILES", prompt)
        self.assertIn('assert greet() == "hello world"', prompt)
        self.assertEqual(
            self.context_source.read_text(encoding="utf-8"),
            'def test_greet() -> None:\n    assert greet() == "hello world"\n',
        )

    def test_proposal_rejects_changes_to_read_only_context(self) -> None:
        class ContextEditingLLM(BaseLLM):
            def generate(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.2,
                max_tokens: int | None = None,
            ) -> LLMResponse:
                return LLMResponse(
                    content=(
                        '{"summary":"Weaken test","reason":"Invalid context edit.",'
                        '"verification_plan":["Run tests"],'
                        '"changes":[{"path":"tests/test_example.py","replacements":['
                        '{"old_text":"hello world","new_text":"hello"}]}]}'
                    ),
                    model="context-editing-model",
                )

        service = CodeWorkspaceService(
            self.repository,
            llm_factory=ContextEditingLLM,
            code_chat_runner=CodeConversationRunner(),
        )

        with self.assertRaisesRegex(CodeWorkspaceError, "explicitly selected"):
            service.propose(
                root=str(self.root),
                objective="Make the implementation satisfy its test.",
                selected_files=["src/example.py"],
                context_files=["tests/test_example.py"],
                owner_user_id="admin",
            )

    def test_proposal_repairs_one_invalid_model_response(self) -> None:
        class RepairingProposalLLM(ProposalLLM):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def generate(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.2,
                max_tokens: int | None = None,
            ) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content="I will improve the greeting.",
                        model="repairing-proposal-model",
                    )
                return super().generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        proposal_llm = RepairingProposalLLM()
        service = CodeWorkspaceService(
            self.repository,
            llm_factory=lambda: proposal_llm,
            code_chat_runner=CodeConversationRunner(),
        )

        proposal = service.propose(
            root=str(self.root),
            objective="Make the greeting clearer.",
            selected_files=["src/example.py"],
            owner_user_id="admin",
        )

        self.assertEqual(proposal_llm.calls, 2)
        self.assertEqual(proposal.changed_files, ["src/example.py"])

    def test_apply_rejects_changes_made_after_proposal(self) -> None:
        proposal = self.service.propose(
            root=str(self.root),
            objective="Make the greeting clearer.",
            selected_files=["src/example.py"],
            owner_user_id="admin",
        )
        self.source.write_text('def greet() -> str:\n    return "user edit"\n', encoding="utf-8")

        with self.assertRaisesRegex(CodeWorkspaceError, "changed after"):
            self.service.apply(proposal.id, "admin")
        self.assertIn("user edit", self.source.read_text(encoding="utf-8"))

    def test_rollback_rejects_file_changed_after_apply(self) -> None:
        proposal = self.service.propose(
            root=str(self.root),
            objective="Make the greeting clearer.",
            selected_files=["src/example.py"],
            owner_user_id="admin",
        )
        applied = self.service.apply(proposal.id, "admin")
        self.assertEqual(applied.status, "applied")
        self.assertIn("hello world", self.source.read_text(encoding="utf-8"))

        # 应用后、回滚前，文件被外部改动 → 回滚不得覆盖用户的新改动。
        edited = 'def greet() -> str:\n    return "user edit after apply"\n'
        self.source.write_text(edited, encoding="utf-8")

        with self.assertRaisesRegex(CodeWorkspaceError, "changed after"):
            self.service.rollback(proposal.id, "admin")
        # 用户改动必须原样保留。
        self.assertIn("user edit after apply", self.source.read_text(encoding="utf-8"))

    def test_workspace_root_must_be_git_root(self) -> None:
        with self.assertRaisesRegex(CodeWorkspaceError, "repository root"):
            scan_code_workspace(self.root / "src")


if __name__ == "__main__":
    unittest.main()
