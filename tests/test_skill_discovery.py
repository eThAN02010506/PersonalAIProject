import types
import unittest
from unittest.mock import patch

from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class ExtraSkill(BaseSkill):
    name = "extra_skill"
    description = "A skill module that appears without any registration."

    async def run(self, request: SkillRequest) -> SkillResponse:
        return SkillResponse(success=True, content="extra")


class SkillDiscoveryTests(unittest.TestCase):
    def test_registry_auto_discovers_builtin_skills(self) -> None:
        registry = SkillRegistry.discover()

        self.assertEqual(
            registry.list_names(),
            [
                "browser",
                "code_patch",
                "code_read",
                "code_search",
                "code_test",
                "code_tree",
                "document_parser",
                "excel_analysis",
                "excel_modeling",
                "excel_schema",
                "excel_statistics",
                "graph_search",
                "rag_search",
                "web_search",
            ],
        )

    def test_registry_picks_up_new_skill_module_without_registration(self) -> None:
        extra_module = types.ModuleType("qwopus_agent.skills.extra_discovered")
        extra_module.create_skill = lambda: ExtraSkill()

        def fake_import_module(name: str):
            if name == "qwopus_agent.skills.extra_discovered":
                return extra_module
            import importlib

            return importlib.import_module(name)

        with (
            patch("qwopus_agent.skills.registry.pkgutil.iter_modules") as iter_modules,
            patch("qwopus_agent.skills.registry.importlib.import_module", fake_import_module),
        ):
            iter_modules.return_value = pkgutil_module_infos() + [ModuleInfo("extra_discovered")]
            registry = SkillRegistry.discover()

        # 原因：内置 Skill 的加入不应依赖中央注册列表。
        # 作用：锁定一个带 create_skill() 的新模块被自动发现并注册。
        self.assertIn("extra_skill", registry.list_names())
        self.assertIsInstance(registry.get("extra_skill"), ExtraSkill)


def pkgutil_module_infos():
    import pkgutil

    import qwopus_agent.skills as skills_package

    return [
        info
        for info in pkgutil.iter_modules(skills_package.__path__)
        if not info.name.startswith("_")
        and info.name not in {"base", "catalog", "registry", "workflow"}
    ]


class ModuleInfo:
    def __init__(self, name: str) -> None:
        self.name = name


if __name__ == "__main__":
    unittest.main()
