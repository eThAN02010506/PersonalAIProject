"""Reusable skill system interfaces."""

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse
from qwopus_agent.skills.registry import SkillRegistry

__all__ = ["BaseSkill", "SkillRegistry", "SkillRequest", "SkillResponse"]
