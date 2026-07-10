import unittest

from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class DemoSkill(BaseSkill):
    name = "demo"
    description = "Demo skill for contract tests."

    async def run(self, request: SkillRequest) -> SkillResponse:
        return SkillResponse(success=True, content=request.query)


class SkillRegistryTests(unittest.TestCase):
    def test_registry_registers_and_resolves_skills(self) -> None:
        registry = SkillRegistry()
        skill = DemoSkill()

        registry.register(skill)

        self.assertIs(registry.get("demo"), skill)
        self.assertEqual(registry.list_names(), ["demo"])

    def test_registry_rejects_duplicate_names(self) -> None:
        registry = SkillRegistry()
        registry.register(DemoSkill())

        with self.assertRaisesRegex(ValueError, "Skill already registered"):
            registry.register(DemoSkill())

    def test_registry_discovers_package_without_manual_registration(self) -> None:
        registry = SkillRegistry.discover()

        self.assertIsInstance(registry.list_names(), list)
