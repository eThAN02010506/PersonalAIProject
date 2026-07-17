"""Reusable skill system interfaces."""

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse
from qwopus_agent.skills.catalog import SkillCatalog, SkillManifest
from qwopus_agent.skills.registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "SkillCatalog",
    "SkillManifest",
    "SkillRegistry",
    "SkillRequest",
    "SkillResponse",
]
