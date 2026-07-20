import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.memory.minirag import MiniRAG
from qwopus_agent.services.knowledge_maintenance_service import KnowledgeMaintenanceService
from tests.minirag_fakes import TestEmbeddingBackend, make_test_minirag


class MiniRAGTests(unittest.TestCase):
    def test_insert_and_search_expose_only_simple_knowledge_api(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")

            memory.insert("# Revenue\nQ1 revenue increased.")

            self.assertEqual(memory.search("revenue"), ["# Revenue\nQ1 revenue increased."])

    def test_insert_rejects_empty_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")

            with self.assertRaisesRegex(ValueError, "document must not be empty"):
                memory.insert(" ")

    def test_search_supports_chinese_queries_without_spaces(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")

            memory.insert("# 销售分析\n收入增长，缺失值较少。")

            self.assertEqual(memory.search("分析收入"), ["# 销售分析\n收入增长，缺失值较少。"])

    def test_insert_persists_documents_for_next_instance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = make_test_minirag(storage_path)

            memory.insert("# Persisted\nLong-term knowledge.")

            reloaded = make_test_minirag(storage_path)
            self.assertEqual(
                reloaded.search("knowledge"),
                ["# Persisted\nLong-term knowledge."],
            )

    def test_search_ranks_documents_with_vector_similarity(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")

            memory.insert("# Cooking\nPasta sauce and tomato recipe.")
            memory.insert("# Agent\nPlanner Executor SkillRegistry routing architecture.")

            # 原因：MiniRAG 已从关键词过滤升级为内部向量排序。
            # 作用：验证 search(query) 外部接口不变，但更相关文档排在前面。
            results = memory.search("executor routing")

            self.assertEqual(
                results[0],
                "# Agent\nPlanner Executor SkillRegistry routing architecture.",
            )

    def test_insert_deduplicates_exact_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = make_test_minirag(storage_path)

            memory.insert("# Same")
            memory.insert("# Same")

            self.assertEqual(memory.search("same"), ["# Same"])
            self.assertEqual(len(storage_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_search_uses_semantic_embeddings_instead_of_exact_keywords(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert("The automobile maintenance schedule is due next week.")
            memory.insert("This recipe explains how to prepare tomato pasta.")

            results = memory.search("car maintenance")

            self.assertIn("automobile maintenance", results[0])

    def test_search_returns_file_and_page_citations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert("# File: budget.pdf\n\n## Page 3\n\nAnnual revenue reached 2 million.")

            results = memory.search("revenue")

            self.assertEqual(
                results,
                ["[Source: budget.pdf | Page: 3]\nAnnual revenue reached 2 million."],
            )

    def test_restart_reuses_persisted_vector_index(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            first_backend = TestEmbeddingBackend()
            memory = MiniRAG(storage_path=storage_path, embedding_backend=first_backend)
            memory.insert("Persistent semantic knowledge about project planning.")
            self.assertGreater(first_backend.encode_calls, 0)

            second_backend = TestEmbeddingBackend()
            reloaded = MiniRAG(storage_path=storage_path, embedding_backend=second_backend)

            # 原因：启动时应直接加载已有向量，而不是重复计算全部 embedding。
            # 作用：证明持久化索引有效，并让大量文档的应用重启保持快速。
            self.assertEqual(second_backend.encode_calls, 0)
            self.assertTrue(reloaded.search("project planning"))
            self.assertTrue((Path(tmpdir) / "documents_index/vdb_qwopus_chunks.json").exists())

    def test_search_keeps_a_small_document_beside_many_large_document_chunks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            large_content = "\n\n".join(
                f"Revenue table section {index} contains annual finance totals."
                for index in range(80)
            )
            memory.insert(f"# File: large.xlsx\n\n{large_content}")
            memory.insert("# File: policy.txt\n\nAnnual revenue policy requires an audit.")

            results = memory.search("annual revenue")

            # 原因：一个大型工作簿可能产生几十个高相似 chunk。
            # 作用：锁定跨文档多样化，确保小文件仍能进入 Agent 的检索上下文。
            self.assertTrue(any("Source: policy.txt" in result for result in results))

    def test_search_prioritizes_persistent_cross_document_graph_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = make_test_minirag(storage_path)
            memory.insert(
                "# File: ownership.pdf\n\n"
                "[[Aurora Holdings|Organization]] -[owns]-> "
                "[[Blue Harbor Ltd|Organization]]"
            )
            memory.insert(
                "# File: project.docx\n\n"
                "[[Blue Harbor Ltd|Organization]] -[participates_in]-> "
                "[[Project Lantern|Project]]"
            )
            memory.insert(
                "# File: budget.xlsx\n\n"
                "[[Project Lantern|Project]] -[has_budget]-> "
                "[[USD 6 million|Amount]]"
            )

            results = memory.search(
                "How is Aurora Holdings related to USD 6 million?"
            )

            self.assertIn("[Knowledge Graph Path]", results[0])
            self.assertIn("Aurora Holdings -[owns]-> Blue Harbor Ltd", results[0])
            self.assertIn("Project Lantern -[has_budget]-> USD 6 million", results[0])
            self.assertIn("Source: ownership.pdf", results[0])
            self.assertIn("Source: budget.xlsx", results[0])

            reloaded = make_test_minirag(storage_path)
            self.assertIn(
                "[Knowledge Graph Path]",
                reloaded.search("Aurora Holdings and USD 6 million")[0],
            )
            self.assertTrue((Path(tmpdir) / "documents_graph.json").exists())

    def test_same_source_update_replaces_old_vector_and_graph_facts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = make_test_minirag(storage_path)
            memory.insert(
                "# File: company.txt\n\n"
                "[[Company A|Organization]] -[owns]-> [[Company B|Organization]]"
            )

            memory.insert(
                "# File: company.txt\n\n"
                "[[Company A|Organization]] -[partners_with]-> [[Company C|Organization]]"
            )

            old_fact_results = "\n".join(memory.search("Company A and Company B"))
            self.assertNotIn("Company B", old_fact_results)
            self.assertNotIn("-[owns]->", old_fact_results)
            self.assertIn(
                "Company A -[partners_with]-> Company C",
                memory.search("Company A and Company C")[0],
            )
            stored = storage_path.read_text(encoding="utf-8")
            self.assertNotIn("Company B", stored)
            self.assertEqual(len(stored.splitlines()), 1)

    def test_delete_preserves_relation_supported_by_another_document(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            relation = "[[Company A|Organization]] -[owns]-> [[Company B|Organization]]"
            memory.insert(f"# File: first.txt\n\n{relation}")
            memory.insert(f"# File: second.txt\n\n{relation}")
            maintenance = KnowledgeMaintenanceService(memory)

            self.assertEqual(maintenance.delete_source("first.txt"), 1)
            remaining = memory.search("Company A and Company B")[0]
            self.assertIn("[Knowledge Graph Path]", remaining)
            self.assertNotIn("Source: first.txt", remaining)
            self.assertIn("Source: second.txt", remaining)

            self.assertEqual(maintenance.delete_source("second.txt"), 1)
            self.assertNotIn(
                "[Knowledge Graph Path]",
                "\n".join(memory.search("Company A and Company B")),
            )

    def test_rebuild_restores_graph_and_vector_indexes_from_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = make_test_minirag(storage_path)
            memory.insert(
                "# File: facts.txt\n\n"
                "[[Company A|Organization]] -[owns]-> [[Company B|Organization]]"
            )
            original_documents = storage_path.read_text(encoding="utf-8")

            KnowledgeMaintenanceService(memory).rebuild_indexes()

            self.assertEqual(storage_path.read_text(encoding="utf-8"), original_documents)
            self.assertIn(
                "[Knowledge Graph Path]",
                memory.search("Company A and Company B")[0],
            )
            self.assertTrue(memory.search("owns"))
