import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.llm import BaseLLM, ChatMessage, LLMResponse
from qwopus_agent.memory.graph_extraction import (
    CompositeGraphExtractor,
    LLMGraphExtractor,
    RuleBasedGraphExtractor,
)
from qwopus_agent.memory.minirag import (
    MiniRAG,
    _diverse_chunks,
    _KnowledgeChunk,
)
from qwopus_agent.services.knowledge_maintenance_service import KnowledgeMaintenanceService
from tests.minirag_fakes import TestEmbeddingBackend, make_test_minirag


class _NaturalRelationLLM(BaseLLM):
    """Return one evidence-backed relationship for the integration test."""

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens
        chunks = json.loads(messages[-1].content)
        chunk = chunks[0]
        evidence = "Aurora Holdings owns Blue Harbor Ltd."
        return LLMResponse(
            content=json.dumps(
                {
                    "entities": [
                        {
                            "name": "Aurora Holdings",
                            "entity_type": "Organization",
                            "aliases": [],
                            "description": "",
                            "chunk_id": chunk["chunk_id"],
                            "evidence": evidence,
                        },
                        {
                            "name": "Blue Harbor Ltd",
                            "entity_type": "Organization",
                            "aliases": [],
                            "description": "",
                            "chunk_id": chunk["chunk_id"],
                            "evidence": evidence,
                        },
                    ],
                    "relations": [
                        {
                            "source": "Aurora Holdings",
                            "relation": "owns",
                            "target": "Blue Harbor Ltd",
                            "confidence": 0.98,
                            "chunk_id": chunk["chunk_id"],
                            "evidence": evidence,
                        }
                    ],
                }
            ),
            model="graph-test-model",
        )


class MiniRAGTests(unittest.TestCase):
    def test_diverse_chunks_remove_near_duplicate_evidence(self) -> None:
        def chunk(identifier: str, content: str) -> _KnowledgeChunk:
            return _KnowledgeChunk(
                id=identifier,
                document_id=identifier,
                source=f"{identifier}.md",
                page=None,
                page_end=None,
                section_id="section",
                section_path=("Section",),
                content=content,
                position=0,
            )

        repeated = "Revenue increased by ten percent during the first quarter. " * 8
        ranked = [
            chunk("first", repeated),
            chunk("duplicate", repeated + " "),
            chunk("different", "Customer retention improved after the support redesign."),
        ]

        # 原因：重叠切片可能拥有近乎相同的高向量分数并反复进入模型上下文。
        # 作用：保留首个高排名证据和不同证据，同时剔除近重复 source。
        self.assertEqual(
            [item.id for item in _diverse_chunks(ranked)],
            ["first", "different"],
        )

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

    def test_natural_document_builds_evidence_bound_graph_with_llm_extractor(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = MiniRAG(
                storage_path=Path(tmpdir) / "documents.jsonl",
                embedding_backend=TestEmbeddingBackend(),
                graph_extractor=CompositeGraphExtractor(
                    extractors=(
                        RuleBasedGraphExtractor(),
                        LLMGraphExtractor(lambda: _NaturalRelationLLM()),
                    )
                ),
            )
            memory.insert(
                "# File: ownership.pdf\n\n"
                "Aurora Holdings owns Blue Harbor Ltd."
            )

            paths = memory.graph_index.paths_between(
                "Aurora Holdings",
                "Blue Harbor Ltd",
            )

            # 原因：独立 extractor 单测不能证明自然文本真正进入持久 MiniRAG 图谱。
            # 作用：锁定普通文档 → LLM 证据校验 → 实体/关系持久化 → 图路径查询全链路。
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].relations[0].relation, "owns")
            self.assertEqual(paths[0].evidence[0].source, "ownership.pdf")

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

    def test_existing_instance_refreshes_after_another_instance_writes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            first = make_test_minirag(storage_path)
            second = make_test_minirag(storage_path)

            second.insert("# File: shared.md\n\nCross-process knowledge is now available.")

            # 原因：FastAPI worker 与 Agent 子进程可能长期持有同一路径的独立内存快照。
            # 作用：证明旧实例会在搜索前检测事实库变化，并加载另一个实例刚写入的文档。
            self.assertIn(
                "Cross-process knowledge",
                "\n".join(first.search("cross process knowledge")),
            )
            self.assertEqual(first.list_sources(), ["shared.md"])

    def test_reload_preserves_documents_with_unicode_line_separators(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "documents.jsonl"
            memory = make_test_minirag(storage_path)
            memory.insert(
                "# File: lesson-27.md\n\n"
                "Life is offered with joy.\u2028This is still the same JSONL record."
            )

            reloaded = make_test_minirag(storage_path)

            # 原因：真实第 27 课含 U+2028，splitlines() 会把合法 JSON 字符串拆坏并静默漏源。
            # 作用：锁定 JSONL 只按物理换行读取，Unicode 排版字符不会删除整份文档。
            self.assertEqual(reloaded.list_sources(), ["lesson-27.md"])
            self.assertTrue(reloaded.search("offered with joy"))

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

    def test_search_excludes_sources_below_request_relevance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert("Revenue planning notes also mention staffing and office logistics.")

            # 原因：滑块必须改变实际检索结果，不能只在前端隐藏 Source。
            # 作用：锁定低阈值保留弱相关内容，高阈值在 Agent 推理前剔除它。
            self.assertTrue(memory.search("revenue", min_relevance=0.25))
            self.assertEqual(memory.search("revenue", min_relevance=0.95), [])

    def test_search_rejects_invalid_relevance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")

            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                memory.search("revenue", min_relevance=1.1)

    def test_search_returns_file_and_page_citations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert("# File: budget.pdf\n\n## Page 3\n\nAnnual revenue reached 2 million.")

            results = memory.search("revenue")

            self.assertEqual(
                results,
                ["[Source: budget.pdf | Page: 3]\nAnnual revenue reached 2 million."],
            )

    def test_search_returns_section_path_and_filters_current_document(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            north_id = memory.insert(
                "# File: north.md\n\n# Finance\n\n## Revenue\n\nNorth revenue increased."
            )
            memory.insert(
                "# File: south.md\n\n# Finance\n\n## Revenue\n\nSouth revenue decreased."
            )

            results = memory.search(
                "revenue",
                document_ids=(north_id,),
            )

            self.assertTrue(results)
            self.assertTrue(all("Source: north.md" in result for result in results))
            self.assertIn("Section: Finance / Revenue", results[0])
            self.assertNotIn("South revenue", "\n".join(results))

    def test_section_filter_trims_mixed_graph_evidence(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert(
                "# File: scoped.md\n\n"
                "# Allowed\n\n"
                "[[Company A|Organization]] -[owns]-> [[Company B|Organization]]\n\n"
                "# Blocked\n\n"
                "[[Company A]] -[owns]-> [[Company B]]"
            )
            allowed_section_id = next(
                chunk.section_id
                for chunk in memory._records[0].chunks
                if chunk.section_path == ("Allowed",)
            )

            results = memory.search(
                "Company A and Company B",
                section_ids=(allowed_section_id,),
            )

            self.assertTrue(results)
            self.assertIn("[Knowledge Graph Path]", results[0])
            self.assertEqual(results[0].count("- [Source: scoped.md]"), 1)
            self.assertIn(
                "[[Company A|Organization]] -[owns]-> [[Company B|Organization]]",
                results[0],
            )
            self.assertNotIn(
                "[[Company A]] -[owns]-> [[Company B]]",
                "\n".join(results),
            )

    def test_section_filter_drops_graph_path_with_an_unscoped_hop(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert(
                "# File: scoped.md\n\n"
                "# Allowed\n\n"
                "[[Company A|Organization]] -[owns]-> [[Company B|Organization]]\n\n"
                "# Blocked\n\n"
                "[[Company B|Organization]] -[funds]-> [[Project C|Project]]"
            )
            allowed_section_id = next(
                chunk.section_id
                for chunk in memory._records[0].chunks
                if chunk.section_path == ("Allowed",)
            )

            results = memory.search(
                "Company A and Project C",
                section_ids=(allowed_section_id,),
            )

            self.assertNotIn("[Knowledge Graph Path]", "\n".join(results))
            self.assertNotIn("-[funds]->", "\n".join(results))

    def test_search_can_limit_results_by_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert("# File: alpha.txt\n\nShared project schedule.")
            memory.insert("# File: beta.txt\n\nShared project budget.")

            results = memory.search("shared project", sources=("beta.txt",))

            self.assertTrue(results)
            self.assertTrue(all("Source: beta.txt" in result for result in results))

    def test_search_can_match_a_document_by_its_source_filename(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert(
                "# File: lesson-zeta-21.md\n\n"
                "This body deliberately contains unrelated general notes."
            )

            results = memory.search("lesson zeta 21")

            # 原因：文件名过去只存在于 metadata，没有进入 embedding，按课次查找会漏文档。
            # 作用：锁定来源名称参与语义召回，同时结果仍返回可审计的 Source。
            self.assertTrue(results)
            self.assertIn("Source: lesson-zeta-21.md", results[0])

    def test_search_can_return_evidence_from_eleven_matching_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            unique_topics = [
                "revenue forecasting",
                "customer retention",
                "supply planning",
                "product quality",
                "team hiring",
                "security controls",
                "market research",
                "delivery schedule",
                "budget review",
                "risk register",
                "support workflow",
            ]
            for index, topic in enumerate(unique_topics, start=21):
                memory.insert(
                    f"# File: lesson-{index}.md\n\n"
                    f"Portfolio comparison evidence discusses {topic}."
                )

            results = memory.search("portfolio comparison evidence")
            sources = {
                result.split("Source: ", 1)[1].split(" |", 1)[0].split("]", 1)[0]
                for result in results
                if "Source: " in result
            }

            # 原因：旧的 top_k=5 和 Tool max_results=3 不可能覆盖 21-31 共 11 份文档。
            # 作用：证明检索层至少能把每份匹配文档的最佳 chunk 交给上层 Agent。
            self.assertEqual(len(sources), 11)

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
