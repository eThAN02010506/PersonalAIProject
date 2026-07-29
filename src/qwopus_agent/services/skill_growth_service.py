"""Automatic extraction, validation, versioning, and deployment of workflow skills."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwopus_agent.agents.router import AgentRun
from qwopus_agent.skills import (
    SkillCatalog,
    SkillManifest,
    SkillRegistry,
    WorkflowSkill,
    WorkflowSpec,
    WorkflowStep,
)

DEFAULT_GROWTH_HISTORY_PATH = Path("storage/skills/growth_history.json")
DEFAULT_WORKFLOW_ROOT = Path("storage/skills/workflows")
SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
)
RUNTIME_PATH_KEYS = {
    "file_path",
    "input_path",
    "output_path",
    "path",
    "paths",
    "upload_path",
}


@dataclass(frozen=True)
class SkillGrowthPolicy:
    """Thresholds that prevent one-off or weak runs from becoming skills."""

    min_successes: int = 2
    min_output_chars: int = 40
    name_prefix: str = "learned_"
    auto_promote: bool = False

    def __post_init__(self) -> None:
        if self.min_successes < 1:
            raise ValueError("min_successes must be at least 1.")
        if self.min_output_chars < 0:
            raise ValueError("min_output_chars cannot be negative.")
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", self.name_prefix):
            raise ValueError("name_prefix must be a safe Python-style identifier.")


@dataclass(frozen=True)
class SkillGrowthDecision:
    """Auditable outcome from observing one Agent run."""

    status: str
    reason: str
    signature: str | None = None
    manifest: SkillManifest | None = None


@dataclass(frozen=True)
class SkillTraceStep:
    """One reusable capability call retained from a successful run."""

    skill_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillRunTrace:
    """Framework-neutral execution trace evaluated by the growth service."""

    success: bool
    output: str
    steps: tuple[SkillTraceStep, ...] = ()


@dataclass
class SkillGrowthService:
    """Learn safe declarative workflows from repeated successful Agent runs."""

    registry: SkillRegistry
    catalog: SkillCatalog = field(default_factory=SkillCatalog)
    workflow_root: Path = DEFAULT_WORKFLOW_ROOT
    history_path: Path = DEFAULT_GROWTH_HISTORY_PATH
    policy: SkillGrowthPolicy = field(default_factory=SkillGrowthPolicy)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    async def observe(
        self,
        objective: str,
        run: AgentRun,
        context: dict[str, Any] | None = None,
    ) -> SkillGrowthDecision:
        """Adapt the legacy Planner/Executor result to the shared trace contract."""
        complete_trace = bool(
            run.plan.steps
            and len(run.execution.steps) == len(run.plan.steps)
        )
        if not complete_trace:
            return SkillGrowthDecision("ignored", "Run has no complete execution trace.")
        return self.observe_trace(
            objective,
            SkillRunTrace(
                success=run.execution.success,
                output=run.execution.content,
                steps=tuple(
                    SkillTraceStep(
                        skill_name=step.skill_name,
                        arguments=dict(step.arguments),
                    )
                    for step in run.plan.steps
                ),
            ),
            context=context,
        )

    def observe_trace(
        self,
        objective: str,
        trace: SkillRunTrace,
        context: dict[str, Any] | None = None,
    ) -> SkillGrowthDecision:
        """Create a candidate from one framework-neutral successful execution trace."""
        if not trace.success:
            return SkillGrowthDecision("ignored", "Execution was not successful.")
        if not trace.steps:
            return SkillGrowthDecision("ignored", "Run has no complete execution trace.")
        if len(trace.output.strip()) < self.policy.min_output_chars:
            return SkillGrowthDecision(
                "ignored",
                "Successful output is too short to validate reuse.",
            )
        if any(
            step.skill_name.startswith(self.policy.name_prefix)
            for step in trace.steps
        ):
            return SkillGrowthDecision("ignored", "Learned workflows are not learned recursively.")

        signature = self._signature(trace)
        with self._lock:
            history = self._load_history()
            record = dict(history.get(signature, {}))
            successes = int(record.get("successes", 0)) + 1
            record.update(
                {
                    "successes": successes,
                    "skills": [step.skill_name for step in trace.steps],
                    "last_objective": objective,
                    "intent_examples": _append_intent_example(
                        record.get("intent_examples"),
                        objective,
                    ),
                }
            )
            history[signature] = record
            self._write_history(history)

            if successes < self.policy.min_successes:
                return SkillGrowthDecision(
                    "observed",
                    f"Waiting for {self.policy.min_successes - successes} more successful run(s).",
                    signature,
                )

            skill_name = self._skill_name(trace)
            active = self.catalog.active(skill_name)
            if active is not None and active.source_signature == signature:
                return SkillGrowthDecision(
                    "deployed",
                    "The matching workflow version is already active.",
                    signature,
                    active,
                )

            version = self.catalog.next_patch_version(skill_name)
            spec = self._build_spec(
                skill_name,
                version,
                signature,
                trace,
                intent_examples=tuple(record["intent_examples"]),
            ).sealed()
            self.validate(spec)
            spec_path = self._persist_spec(spec)
            source_run_id = str((context or {}).get("trace_id") or uuid.uuid4().hex)
            candidate = SkillManifest(
                name=spec.name,
                version=spec.version,
                description=spec.description,
                module_path="qwopus_agent.skills.workflow:WorkflowSkill",
                checksum=spec.checksum,
                status="candidate",
                spec_path=str(spec_path.resolve()),
                created_at=datetime.now(UTC).isoformat(),
                source_run_id=source_run_id,
                source_signature=signature,
            )
            # 原因：候选必须先进入 Catalog，激活状态才具备可审计的来源和版本。
            # 作用：运行时热部署成功后再切换 active，失败候选不会在下次启动时加载。
            self.catalog.register(candidate)
            if not self.policy.auto_promote:
                return SkillGrowthDecision(
                    "candidate",
                    "Workflow passed validation and is waiting for promotion.",
                    signature,
                    candidate,
                )
            active_manifest = self.promote(spec.name, spec.version)
            record["deployed_version"] = spec.version
            self._write_history(history)
            return SkillGrowthDecision(
                "deployed",
                "Workflow passed validation and was deployed.",
                signature,
                active_manifest,
            )

    def promote(self, name: str, version: str) -> SkillManifest:
        """Validate and activate one persisted candidate workflow."""
        with self._lock:
            return self._activate_version(
                name,
                version,
                required_status="candidate",
            )

    def reject(self, name: str, version: str) -> SkillManifest:
        """Reject one candidate without deleting its spec or provenance."""
        with self._lock:
            return self.catalog.reject(name, version)

    def rollback(self, name: str, version: str) -> SkillManifest:
        """Reactivate one archived version after repeating every safety check."""
        with self._lock:
            return self._activate_version(
                name,
                version,
                required_status="archived",
            )

    def spec_for(self, manifest: SkillManifest) -> WorkflowSpec | None:
        """Load one valid confined spec for review without activating it."""
        if manifest.spec_path is None:
            return None
        try:
            spec_path = self._confined_spec_path(manifest.spec_path)
            spec = WorkflowSpec.model_validate_json(
                spec_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if (
            spec.name != manifest.name
            or spec.version != manifest.version
            or spec.checksum != manifest.checksum
            or not spec.checksum_is_valid()
        ):
            return None
        return spec

    def _activate_version(
        self,
        name: str,
        version: str,
        *,
        required_status: str,
    ) -> SkillManifest:
        """Load, validate, register, and atomically activate one exact version."""
        manifest = next(
            (
                item
                for item in self.catalog.list()
                if item.name == name and item.version == version
            ),
            None,
        )
        if manifest is None or manifest.spec_path is None:
            raise KeyError(f"Unknown workflow candidate: {name}@{version}")
        if manifest.status != required_status:
            raise ValueError(
                f"Skill {name}@{version} must be {required_status}, "
                f"not {manifest.status}."
            )
        spec = self.spec_for(manifest)
        if spec is None:
            raise ValueError("Workflow spec is missing, outside its root, or invalid.")
        self.validate(spec)
        # 原因：Catalog 的 active 状态不能证明当前进程已经加载了可执行 Skill。
        # 作用：先完成完整性校验和热注册，再原子切换持久化激活版本。
        self.registry.register(WorkflowSkill(spec, self.registry), replace=True)
        return self.catalog.activate(name, version)

    def _confined_spec_path(self, spec_path: str) -> Path:
        allowed_root = self.workflow_root.resolve()
        resolved = Path(spec_path).resolve()
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            # 原因：Catalog 是可编辑本地文件，不能把其中的绝对路径直接当成可信输入。
            # 作用：Promote、Rollback 和管理 API 都只能读取当前 workflow_root 内的 spec。
            raise ValueError("Workflow spec path is outside the workflow root.") from exc
        if not resolved.is_file():
            raise ValueError("Workflow spec file does not exist.")
        return resolved

    def validate(self, spec: WorkflowSpec) -> None:
        """Validate integrity, references, recursion, and persisted arguments."""
        if not spec.checksum_is_valid():
            raise ValueError("Workflow checksum validation failed.")
        if not spec.name.startswith(self.policy.name_prefix):
            raise ValueError("Learned workflow name uses an unexpected prefix.")
        available = set(self.registry.list_names())
        for step in spec.steps:
            if step.skill_name == spec.name:
                raise ValueError("A workflow cannot call itself.")
            if step.skill_name not in available:
                raise ValueError(f"Workflow references unknown skill: {step.skill_name}")
            if "{query}" not in step.query_template:
                raise ValueError("Workflow query templates must contain {query}.")
            if _contains_sensitive_key(step.arguments):
                raise ValueError("Workflow contains sensitive or runtime-specific arguments.")

    def _signature(self, trace: SkillRunTrace) -> str:
        payload = {
            "skills": [step.skill_name for step in trace.steps],
            "argument_keys": [sorted(step.arguments) for step in trace.steps],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _skill_name(self, trace: SkillRunTrace) -> str:
        sequence = "_then_".join(
            _safe_name(step.skill_name) for step in trace.steps
        )
        name = f"{self.policy.name_prefix}{sequence}"
        if len(name) <= 90:
            return name
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
        return f"{name[:79]}_{digest}"

    def _build_spec(
        self,
        name: str,
        version: str,
        signature: str,
        trace: SkillRunTrace,
        *,
        intent_examples: tuple[str, ...],
    ) -> WorkflowSpec:
        steps = tuple(
            WorkflowStep(
                skill_name=step.skill_name,
                query_template="{query}",
                arguments=_sanitize_arguments(step.arguments),
            )
            for step in trace.steps
        )
        sequence = " -> ".join(step.skill_name for step in trace.steps)
        return WorkflowSpec(
            name=name,
            version=version,
            description=f"Learned and validated workflow: {sequence}.",
            intent_examples=intent_examples,
            steps=steps,
            source_signature=signature,
        )

    def _persist_spec(self, spec: WorkflowSpec) -> Path:
        target = self.workflow_root / spec.name / f"{spec.version}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def _load_history(self) -> dict[str, dict[str, Any]]:
        if not self.history_path.exists():
            return {}
        payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        signatures = payload.get("signatures", {})
        if not isinstance(signatures, dict):
            raise ValueError("Skill growth history has an invalid shape.")
        return {str(key): dict(value) for key, value in signatures.items()}

    def _write_history(self, history: dict[str, dict[str, Any]]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(f"{self.history_path.suffix}.tmp")
        temporary.write_text(
            json.dumps({"signatures": history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.history_path)


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return normalized or "skill"


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold()
    return normalized in RUNTIME_PATH_KEYS or any(
        fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS
    )


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _is_sensitive_key(str(key)) or _contains_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _sanitize_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_arguments(item)
            for key, item in value.items()
            if not _is_sensitive_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_arguments(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _append_intent_example(existing: Any, objective: str) -> list[str]:
    """Keep a small, deduplicated set of successful user intents."""
    examples = (
        [str(item).strip()[:500] for item in existing if str(item).strip()]
        if isinstance(existing, list)
        else []
    )
    normalized = " ".join(objective.split())[:500]
    if normalized and normalized not in examples:
        examples.append(normalized)
    # 原因：历史样例会进入持久化 Skill，不能随运行次数无限增长。
    # 作用：保留最近八个已验证意图，足够辅助匹配且不膨胀 Catalog。
    return examples[-8:]
