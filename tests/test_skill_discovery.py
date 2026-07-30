import unittest

from qwopus_agent.skills import SkillRegistry


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
                "excel_schema",
                "graph_search",
                "rag_search",
                "web_search",
            ],
        )
