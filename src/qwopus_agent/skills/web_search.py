"""Unified web search skill."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class WebSearchSkill(BaseSkill):
    """Provide one search(query) capability for the Planner."""

    # Reason: Planner should choose web search without caring about the provider.
    name: str = "web_search"

    # Role: Unified external search capability; provider wiring comes later.
    description: str = "Search the web through a single provider-independent interface."

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Return a safe placeholder until a concrete search provider is injected."""
        return SkillResponse(
            success=False,
            content="web_search provider is not configured yet.",
            data={"query": request.query},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return WebSearchSkill()
