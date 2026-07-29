import asyncio
import json
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from qwopus_agent.agents import AgentRouter, Plan, PlanStep, SkillExecutor, SkillPlanner
from qwopus_agent.services.skill_growth_service import (
    SkillGrowthPolicy,
    SkillGrowthService,
    SkillRunTrace,
    SkillTraceStep,
)
from qwopus_agent.skills import (
    BaseSkill,
    SkillCatalog,
    SkillRegistry,
    SkillRequest,
    SkillResponse,
    WorkflowSpec,
)


class RecordingSkill(BaseSkill):
    description = "Records workflow calls for tests."

    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls: list[SkillRequest] = []

    async def run(self, request: SkillRequest) -> SkillResponse:
        self.calls.append(request)
        return SkillResponse(
            success=not self.fail,
            content=f"{self.name} completed {request.query}",
        )


@dataclass
class StaticPlanner:
    steps: list[PlanStep]

    async def plan(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
    ) -> Plan:
        return Plan(objective=objective, steps=self.steps)


@dataclass
class BrokenObserver:
    async def observe(
        self,
        objective: str,
        run: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        raise RuntimeError("growth storage unavailable")


class SkillGrowthServiceTests(unittest.TestCase):
    def _components(
        self,
        root: Path,
        *,
        min_successes: int = 2,
        fail_second: bool = False,
        auto_promote: bool = True,
    ) -> tuple[SkillRegistry, SkillCatalog, SkillGrowthService, RecordingSkill, RecordingSkill]:
        registry = SkillRegistry()
        first = RecordingSkill("alpha")
        second = RecordingSkill("beta", fail=fail_second)
        registry.register(first)
        registry.register(second)
        catalog = SkillCatalog(storage_path=root / "catalog.json")
        service = SkillGrowthService(
            registry=registry,
            catalog=catalog,
            workflow_root=root / "workflows",
            history_path=root / "history.json",
            policy=SkillGrowthPolicy(
                min_successes=min_successes,
                min_output_chars=1,
                auto_promote=auto_promote,
            ),
        )
        return registry, catalog, service, first, second

    def test_router_deploys_repeated_success_as_executable_workflow(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, catalog, service, first, second = self._components(root)
            arguments = {
                "file_path": "/private/original.xlsx",
                "api_key": "must-not-persist",
                "mode": "summary",
            }
            router = AgentRouter(
                planner=StaticPlanner(
                    [
                        PlanStep("alpha", "run", arguments),
                        PlanStep("beta", "run", arguments),
                    ]
                ),
                executor=SkillExecutor(registry),
                observers=(service,),
            )

            first_run = asyncio.run(router.run("analyze workbook"))
            self.assertNotIn("learned_alpha_then_beta", registry.list_names())
            second_run = asyncio.run(router.run("analyze another workbook"))

            self.assertEqual(first_run.observer_errors, ())
            self.assertEqual(second_run.observer_errors, ())
            self.assertIn("learned_alpha_then_beta", registry.list_names())
            active = catalog.active("learned_alpha_then_beta")
            self.assertIsNotNone(active)
            self.assertEqual(active.version, "0.1.0")

            spec_text = Path(active.spec_path).read_text(encoding="utf-8")
            self.assertNotIn("must-not-persist", spec_text)
            self.assertNotIn("/private/original.xlsx", spec_text)
            self.assertIn('"mode": "summary"', spec_text)
            self.assertIn("analyze workbook", spec_text)
            self.assertIn("analyze another workbook", spec_text)

            response = asyncio.run(
                registry.execute(
                    "learned_alpha_then_beta",
                    SkillRequest(
                        query="new workbook",
                        arguments={"file_path": "/tmp/new.xlsx"},
                    ),
                )
            )
            self.assertTrue(response.success)
            self.assertEqual(len(response.data["steps"]), 2)
            self.assertEqual(first.calls[-1].arguments["file_path"], "/tmp/new.xlsx")
            self.assertEqual(second.calls[-1].arguments["file_path"], "/tmp/new.xlsx")

            vague_plan = asyncio.run(
                SkillPlanner(registry).plan(
                    "do that again",
                    context={"previous_objective": "analyze another workbook"},
                )
            )
            self.assertEqual(
                [step.skill_name for step in vague_plan.steps],
                ["learned_alpha_then_beta"],
            )

    def test_candidate_requires_promotion_when_auto_promote_is_disabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, catalog, service, _, _ = self._components(
                root,
                min_successes=1,
                auto_promote=False,
            )
            router = AgentRouter(
                planner=StaticPlanner(
                    [PlanStep("alpha", "run"), PlanStep("beta", "run")]
                ),
                executor=SkillExecutor(registry),
                observers=(service,),
            )

            run = asyncio.run(router.run("prepare the recurring report"))
            candidate = catalog.latest("learned_alpha_then_beta")

            self.assertEqual(run.observer_errors, ())
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.status, "candidate")
            self.assertNotIn("learned_alpha_then_beta", registry.list_names())

            promoted = service.promote(candidate.name, candidate.version)

            # 原因：持久化 candidate 本身不应成为可调用能力。
            # 作用：证明只有显式 promote 完成校验、热加载和 active 切换。
            self.assertEqual(promoted.status, "active")
            self.assertIn("learned_alpha_then_beta", registry.list_names())

    def test_framework_neutral_trace_creates_manual_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, catalog, service, _, _ = self._components(
                root,
                min_successes=1,
                auto_promote=False,
            )

            decision = service.observe_trace(
                "research and summarize the current topic",
                SkillRunTrace(
                    success=True,
                    output="A complete source-grounded result.",
                    steps=(
                        SkillTraceStep("alpha"),
                        SkillTraceStep("beta", {"mode": "summary"}),
                    ),
                ),
                context={"trace_id": "production-run-1"},
            )

            self.assertEqual(decision.status, "candidate")
            self.assertEqual(decision.manifest.source_run_id, "production-run-1")
            self.assertEqual(catalog.latest("learned_alpha_then_beta").status, "candidate")
            self.assertNotIn("learned_alpha_then_beta", registry.list_names())

    def test_growth_policy_never_auto_promotes_by_default(self) -> None:
        self.assertFalse(SkillGrowthPolicy().auto_promote)

    def test_failed_execution_is_never_observed_as_a_skill(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, catalog, service, _, _ = self._components(
                root,
                min_successes=1,
                fail_second=True,
            )
            router = AgentRouter(
                planner=StaticPlanner(
                    [PlanStep("alpha", "run"), PlanStep("beta", "run")]
                ),
                executor=SkillExecutor(registry),
                observers=(service,),
            )

            run = asyncio.run(router.run("failed workflow"))

            self.assertFalse(run.execution.success)
            self.assertEqual(catalog.list(), [])
            self.assertFalse((root / "history.json").exists())

    def test_changed_workflow_signature_creates_patch_version(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, catalog, service, _, _ = self._components(root, min_successes=1)

            async def execute(arguments: dict[str, Any]) -> None:
                router = AgentRouter(
                    planner=StaticPlanner(
                        [
                            PlanStep("alpha", "run", arguments),
                            PlanStep("beta", "run", arguments),
                        ]
                    ),
                    executor=SkillExecutor(registry),
                    observers=(service,),
                )
                await router.run("version workflow")

            asyncio.run(execute({"mode": "summary"}))
            asyncio.run(execute({"format": "markdown"}))

            manifests = [
                item for item in catalog.list() if item.name == "learned_alpha_then_beta"
            ]
            self.assertEqual([item.version for item in manifests], ["0.1.0", "0.1.1"])
            self.assertEqual([item.status for item in manifests], ["archived", "active"])
            active_skill = registry.get("learned_alpha_then_beta")
            self.assertEqual(active_skill.spec.version, "0.1.1")

            rolled_back = service.rollback("learned_alpha_then_beta", "0.1.0")

            self.assertEqual(rolled_back.status, "active")
            self.assertEqual(
                registry.get("learned_alpha_then_beta").spec.version,
                "0.1.0",
            )
            self.assertEqual(
                catalog.latest("learned_alpha_then_beta", status="archived").version,
                "0.1.1",
            )

    def test_rejected_candidate_cannot_be_promoted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, catalog, service, _, _ = self._components(
                root,
                min_successes=1,
                auto_promote=False,
            )
            decision = service.observe_trace(
                "repeat alpha",
                SkillRunTrace(
                    success=True,
                    output="A complete reusable output.",
                    steps=(SkillTraceStep("alpha"),),
                ),
            )
            assert decision.manifest is not None

            rejected = service.reject(
                decision.manifest.name,
                decision.manifest.version,
            )

            self.assertEqual(rejected.status, "rejected")
            self.assertEqual(
                catalog.latest(decision.manifest.name).status,
                "rejected",
            )
            with self.assertRaisesRegex(ValueError, "must be candidate"):
                service.promote(
                    decision.manifest.name,
                    decision.manifest.version,
                )

    def test_promotion_rejects_catalog_spec_path_outside_workflow_root(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, catalog, service, _, _ = self._components(
                root,
                min_successes=1,
                auto_promote=False,
            )
            decision = service.observe_trace(
                "repeat alpha safely",
                SkillRunTrace(
                    success=True,
                    output="A complete reusable output.",
                    steps=(SkillTraceStep("alpha"),),
                ),
            )
            assert decision.manifest is not None
            outside = root / "outside.json"
            outside.write_text(
                Path(decision.manifest.spec_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            catalog.register(
                replace(decision.manifest, spec_path=str(outside))
            )

            with self.assertRaisesRegex(ValueError, "outside"):
                service.promote(
                    decision.manifest.name,
                    decision.manifest.version,
                )

    def test_deployed_workflow_reloads_and_tampered_spec_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, catalog, service, _, _ = self._components(root, min_successes=1)
            router = AgentRouter(
                planner=StaticPlanner(
                    [PlanStep("alpha", "run"), PlanStep("beta", "run")]
                ),
                executor=SkillExecutor(registry),
                observers=(service,),
            )
            asyncio.run(router.run("deploy workflow"))

            reloaded = SkillRegistry()
            reloaded.register(RecordingSkill("alpha"))
            reloaded.register(RecordingSkill("beta"))
            reloaded.load_deployed(catalog, root / "workflows")
            self.assertIn("learned_alpha_then_beta", reloaded.list_names())

            active = catalog.active("learned_alpha_then_beta")
            spec_path = Path(active.spec_path)
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
            payload["description"] = "tampered"
            spec_path.write_text(json.dumps(payload), encoding="utf-8")

            rejected = SkillRegistry()
            rejected.register(RecordingSkill("alpha"))
            rejected.register(RecordingSkill("beta"))
            rejected.load_deployed(catalog, root / "workflows")
            self.assertNotIn("learned_alpha_then_beta", rejected.list_names())

    def test_recursive_workflow_call_is_blocked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry, _, service, _, _ = self._components(root, min_successes=1)
            router = AgentRouter(
                planner=StaticPlanner([PlanStep("alpha", "run")]),
                executor=SkillExecutor(registry),
                observers=(service,),
            )
            asyncio.run(router.run("deploy one step"))

            response = asyncio.run(
                registry.execute(
                    "learned_alpha",
                    SkillRequest(
                        query="recursive",
                        context={"workflow_stack": ["learned_alpha"]},
                    ),
                )
            )

            self.assertFalse(response.success)
            self.assertIn("Recursive workflow call blocked", response.content)

    def test_observer_failure_does_not_replace_agent_result(self) -> None:
        registry = SkillRegistry()
        registry.register(RecordingSkill("alpha"))
        router = AgentRouter(
            planner=StaticPlanner([PlanStep("alpha", "run")]),
            executor=SkillExecutor(registry),
            observers=(BrokenObserver(),),
        )

        run = asyncio.run(router.run("keep result"))

        self.assertTrue(run.execution.success)
        self.assertIn("alpha completed", run.execution.content)
        self.assertEqual(
            run.observer_errors,
            ("RuntimeError: growth storage unavailable",),
        )


class WorkflowSpecTests(unittest.TestCase):
    def test_workflow_checksum_changes_with_content(self) -> None:
        spec = WorkflowSpec(
            name="learned_alpha",
            version="0.1.0",
            description="Alpha workflow.",
            steps=({"skill_name": "alpha"},),
            source_signature="signature",
        ).sealed()

        changed = spec.model_copy(update={"description": "Changed workflow."})

        self.assertTrue(spec.checksum_is_valid())
        self.assertFalse(changed.checksum_is_valid())


if __name__ == "__main__":
    unittest.main()
