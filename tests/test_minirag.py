import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.memory.minirag import MiniRAG


class MiniRAGTests(unittest.TestCase):
    def test_insert_and_search_expose_only_simple_knowledge_api(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")

            memory.insert("# Revenue\nQ1 revenue increased.")

            self.assertEqual(memory.search("revenue"), ["# Revenue\nQ1 revenue increased."])

    def test_insert_rejects_empty_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")

            with self.assertRaisesRegex(ValueError, "document must not be empty"):
                memory.insert(" ")

    def test_search_supports_chinese_queries_without_spaces(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")

            memory.insert("# 销售分析\n收入增长，缺失值较少。")

            self.assertEqual(memory.search("分析收入"), ["# 销售分析\n收入增长，缺失值较少。"])

    def test_insert_persists_documents_for_next_instance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = MiniRAG(storage_path=storage_path)

            memory.insert("# Persisted\nLong-term knowledge.")

            reloaded = MiniRAG(storage_path=storage_path)
            self.assertEqual(
                reloaded.search("knowledge"),
                ["# Persisted\nLong-term knowledge."],
            )

    def test_insert_deduplicates_exact_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = MiniRAG(storage_path=storage_path)

            memory.insert("# Same")
            memory.insert("# Same")

            self.assertEqual(memory.search("same"), ["# Same"])
            self.assertEqual(len(storage_path.read_text(encoding="utf-8").splitlines()), 1)
