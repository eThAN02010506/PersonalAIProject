import asyncio
import unittest

from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class DemoSkill(BaseSkill):
    name = "demo"
    description = "Demo skill for contract tests."

    async def run(self, request: SkillRequest) -> SkillResponse:
        return SkillResponse(success=True, content=request.query)


class FailingSkill(BaseSkill):
    name = "failing"
    description = "Skill that exposes execution errors for contract tests."

    async def run(self, request: SkillRequest) -> SkillResponse:
        raise RuntimeError("skill failed")


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

    def test_registry_executes_skill_through_typed_contract(self) -> None:
        registry = SkillRegistry()
        registry.register(DemoSkill())

        # 原因：Registry 必须成为调用 Skill 的唯一公共入口。
        # 作用：验证名称解析、异步执行和 SkillResponse 返回形成完整链路。
        response = asyncio.run(
            registry.execute("demo", SkillRequest(query="hello"))
        )

        self.assertTrue(response.success)
        self.assertEqual(response.content, "hello")

    def test_registry_execute_rejects_unknown_skill(self) -> None:
        registry = SkillRegistry()

        # 原因：未知 Skill 属于配置错误，不能伪装成正常执行结果。
        # 作用：确保调用方能得到清晰且可测试的 Unknown skill 错误。
        with self.assertRaisesRegex(KeyError, "Unknown skill: missing"):
            asyncio.run(
                registry.execute("missing", SkillRequest(query="hello"))
            )

    def test_registry_execute_preserves_skill_error(self) -> None:
        registry = SkillRegistry()
        registry.register(FailingSkill())

        # 原因：Registry 不负责解释具体 Skill 的运行异常。
        # 作用：保留原始错误，让上层 Executor 决定日志和失败策略。
        with self.assertRaisesRegex(RuntimeError, "skill failed"):
            asyncio.run(
                registry.execute("failing", SkillRequest(query="hello"))
            )
