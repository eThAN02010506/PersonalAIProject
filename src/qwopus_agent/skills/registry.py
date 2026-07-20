"""Automatic discovery and runtime registry for independent skills."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import qwopus_agent.skills as skills_package
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse
from qwopus_agent.skills.catalog import SkillCatalog
from qwopus_agent.skills.workflow import WorkflowSkill, WorkflowSpec

IGNORED_MODULES = {"base", "catalog", "registry", "workflow"}
DEFAULT_WORKFLOW_ROOT = Path("storage/skills/workflows")


@dataclass
class SkillRegistry:
    """Registry for statically discovered and dynamically learned skills."""

    _skills: dict[str, BaseSkill] = field(default_factory=dict)

    def register(self, skill: BaseSkill, *, replace: bool = False) -> None:
        """Register one skill instance, optionally replacing its active version."""
        if skill.name in self._skills and not replace:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill:
        """Resolve a skill by name."""
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    def list_names(self) -> list[str]:
        """Return registered skill names in deterministic order."""
        return sorted(self._skills)

    async def execute(self, name: str, request: SkillRequest) -> SkillResponse:
        """Execute one registered skill through the common typed contract."""
        # 原因：调用方不应先取得具体 Skill 再了解其 run() 调用细节。
        # 作用：把 Skill 查找和异步执行收口到 Registry 的统一入口。
        return await self.get(name).run(request)

    def load_deployed(
        self,
        catalog: SkillCatalog,
        workflow_root: Path = DEFAULT_WORKFLOW_ROOT,
    ) -> None:
        """Load valid active workflow specs from the persistent catalog."""
        allowed_root = workflow_root.resolve()
        for manifest in catalog.deployed():
            if manifest.spec_path is None:
                continue
            spec_path = Path(manifest.spec_path).resolve()
            try:
                spec_path.relative_to(allowed_root)
            except ValueError:
                continue
            if not spec_path.is_file():
                continue
            try:
                spec = WorkflowSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                spec.name != manifest.name
                or spec.version != manifest.version
                or spec.checksum != manifest.checksum
                or not spec.checksum_is_valid()
                or spec.name in {step.skill_name for step in spec.steps}
                or any(step.skill_name not in self._skills for step in spec.steps)
            ):
                continue
            # 原因：Catalog 只是元数据，启动后仍需恢复可调用的 WorkflowSkill 对象。
            # 作用：自动部署所有已激活且完整性校验通过的成长 Skill。
            self.register(WorkflowSkill(spec, self), replace=True)

    @classmethod
    def discover(
        cls,
        overrides: dict[str, BaseSkill] | None = None,
        *,
        catalog: SkillCatalog | None = None,
        workflow_root: Path = DEFAULT_WORKFLOW_ROOT,
    ) -> SkillRegistry:
        """Scan built-in modules, then restore deployed workflow skills."""
        registry = cls()
        overrides = overrides or {}
        for module_info in pkgutil.iter_modules(skills_package.__path__):
            if module_info.name.startswith("_") or module_info.name in IGNORED_MODULES:
                continue
            module = importlib.import_module(f"{skills_package.__name__}.{module_info.name}")
            skill = _create_skill_from_module(module)
            if skill is not None and skill.name in overrides:
                # 原因：自动发现要保留零手动注册，同时生产环境需要注入真实 provider。
                # 作用：允许调用方用同名 Skill 覆盖默认占位实现。
                skill = overrides[skill.name]
            if skill is not None:
                registry.register(skill)
        registry.load_deployed(catalog or SkillCatalog(), workflow_root)
        return registry


def _create_skill_from_module(module: ModuleType) -> BaseSkill | None:
    """Create a skill from a module-level factory when present."""
    factory = getattr(module, "create_skill", None)
    if factory is None:
        return None
    skill = factory()
    if not isinstance(skill, BaseSkill):
        raise TypeError(f"{module.__name__}.create_skill() must return BaseSkill.")
    return skill
