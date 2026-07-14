import unittest

from qwopus_agent.memory.minirag import MiniRAG


class MiniRAGTests(unittest.TestCase):
    def test_insert_and_search_expose_only_simple_knowledge_api(self) -> None:
        memory = MiniRAG()

        memory.insert("# Revenue\nQ1 revenue increased.")

        self.assertEqual(memory.search("revenue"), ["# Revenue\nQ1 revenue increased."])

    def test_insert_rejects_empty_documents(self) -> None:
        memory = MiniRAG()

        with self.assertRaisesRegex(ValueError, "document must not be empty"):
            memory.insert(" ")

    def test_search_supports_chinese_queries_without_spaces(self) -> None:
        memory = MiniRAG()

        memory.insert("# 销售分析\n收入增长，缺失值较少。")

        self.assertEqual(memory.search("分析收入"), ["# 销售分析\n收入增长，缺失值较少。"])
