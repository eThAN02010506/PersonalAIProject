"""Automatic Skill Registry.

The registry scans `qwopus_agent.skills`, imports skill modules, and registers any module that exposes
`create_skill()`. This keeps capability registration automatic when a new Skill file is added.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType

import qwopus_agent.skills as skills_package
from qwopus_agent.skills.base import BaseSkill


IGNORED_MODULES = {"base", "registry"}


@dataclass
class SkillRegistry:
    """Registry for dynamically discovered skills."""

    # Reason: Executor should depend on this abstraction, not concrete skill modules.
    _skills: dict[str, BaseSkill] = field(default_factory=dict)

    def register(self, skill: BaseSkill) -> None:
        """Register one skill instance."""
        if skill.name in self._skills:
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

    @classmethod
    def discover(cls) -> SkillRegistry:
        """Build a registry by scanning and importing the skills package."""
        registry = cls()
        for module_info in pkgutil.iter_modules(skills_package.__path__):
            if module_info.name.startswith("_") or module_info.name in IGNORED_MODULES:
                continue

            # Reason: importing modules here lets new files self-register through `create_skill`.
            module = importlib.import_module(f"{skills_package.__name__}.{module_info.name}")
            skill = _create_skill_from_module(module)
            if skill is not None:
                registry.register(skill)
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
