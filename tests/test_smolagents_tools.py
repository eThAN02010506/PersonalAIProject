import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import Workbook

from qwopus_agent.documents import (
    build_document_structure,
    chunk_document_structure,
    summarize_document,
)
from qwopus_agent.integrations.smolagents_tools import (
    TavilySearchConfig,
    build_document_collection_summary_tool,
    build_document_outline_tool,
    build_document_search_tool,
    build_document_section_tool,
    build_document_summary_tool,
    build_excel_analysis_tool,
    build_excel_modeling_tool,
    build_excel_schema_tool,
    build_excel_statistics_tool,
    build_graph_search_tool,
    build_minirag_search_tool,
    build_tavily_search_tool,
)
from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphChunk
from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex
from qwopus_agent.utils.token_budget import TokenBudgetManager, estimate_tokens
from tests.minirag_fakes import make_test_minirag


class FakeTool:
    def __init__(self, *args, **kwargs) -> None:
        pass


class SmolagentsToolsTests(unittest.TestCase):
    def test_minirag_tool_preserves_user_named_source_hint(self) -> None:
        queries: list[str] = []

        class RecordingKnowledgeStore:
            def insert(self, document: str, *, document_id: str | None = None) -> str:
                return document_id or "unused"

            def search(self, query: str, min_relevance: float = 0.25) -> list[str]:
                queries.append(query)
                return ["[Source: README.md]\nReact and FastAPI."]

        with patch.dict(sys.modules, {"smolagents": types.SimpleNamespace(Tool=FakeTool)}):
            tool = build_minirag_search_tool(
                RecordingKnowledgeStore(),
                source_hints=("README.md",),
            )
            result = tool.forward("frontend framework")

        # 原因：真实模型把用户写出的 README.md 从 Tool 查询参数中删掉后发生了零召回。
        # 作用：锁定工具适配层会保留用户点名的已授权来源，不需要放宽全局相关性阈值。
        self.assertIn("User-named sources: README.md", queries[0])
        self.assertIn("React and FastAPI", result)

    def test_tavily_tool_formats_search_results(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        payload = {
            "answer": "Rinse rice, simmer it, then rest it.",
            "results": [
                {
                    "title": "How to Cook Rice",
                    "url": "https://example.com/rice",
                    "content": "Use water and rice, then cook until tender.",
                }
            ],
        }

        with (
            patch.dict(sys.modules, {"smolagents": fake_module}),
            patch("qwopus_agent.integrations.tavily.urllib.request.urlopen") as urlopen,
        ):
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                payload
            ).encode("utf-8")
            tool = build_tavily_search_tool(
                TavilySearchConfig(api_key="test-key", max_results=2, timeout_seconds=1)
            )
            result = tool.forward("how to cook rice")
            duplicate_result = tool.forward("  HOW   TO COOK RICE ")

        # 原因：Tavily Tool 是 smolagents 驱动联网的受控入口。
        # 作用：验证 Tool 返回给 Agent 的是可读证据，而不是原始 JSON。
        self.assertIn("Rinse rice", result)
        self.assertIn("How to Cook Rice", result)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.tavily.com/search")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        # 原因：模型可能只改变大小写或空格后重复搜索相同问题。
        # 作用：证明 Tool 层只消耗一次 Tavily 请求，并引导 Agent 使用已有证据收尾。
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("already completed", duplicate_result)

    def test_tavily_tool_requires_api_key(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)

        with (
            patch.dict(sys.modules, {"smolagents": fake_module}),
            patch(
                "qwopus_agent.integrations.tavily.resolve_tavily_api_key",
                return_value="",
            ),
        ):
            # 原因：开发机可能存在 .env.local，缺密钥分支必须与本机凭据隔离。
            # 作用：稳定验证 Tool 在任何部署环境下都会拒绝无凭据请求。
            tool = build_tavily_search_tool(TavilySearchConfig(api_key=""))
            with self.assertRaisesRegex(RuntimeError, "TAVILY_API_KEY"):
                tool.forward("OpenAI")

    def test_document_tools_use_outline_search_and_section_without_prefix_truncation(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown = (
                "# Introduction\n\nOpening content.\n\n"
                "# Final Findings\n\nThe decisive answer is at the end."
            )
            structure = chunk_document_structure(
                build_document_structure(markdown, source="notes.txt")
            )
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            minirag.insert(
                f"# File: notes.txt\n\n{markdown}",
                document_id=structure.document_id,
            )

            with patch.dict(sys.modules, {"smolagents": fake_module}):
                outline = build_document_outline_tool({"notes.txt": structure})
                search = build_document_search_tool(
                    minirag,
                    {"notes.txt": structure},
                    min_relevance=0.25,
                )
                section = build_document_section_tool({"notes.txt": structure})
                summary = build_document_summary_tool(
                    {"notes.txt": summarize_document(structure)}
                )

            final_section = structure.sections[-1]
            self.assertIn("Final Findings", outline.forward("notes.txt"))
            self.assertIn("decisive answer", search.forward("notes.txt", "decisive answer"))
            section_result = section.forward("notes.txt", final_section.id)
            summary_result = summary.forward("notes.txt")

        self.assertIn("decisive answer", section_result)
        self.assertIn("decisive answer", summary_result)
        self.assertIn("remaining_chunks=0", section_result)
        with self.assertRaisesRegex(ValueError, "Unknown file_name"):
            outline.forward("../secret.txt")

    def test_document_search_falls_back_to_exact_text_in_selected_file(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        structure = chunk_document_structure(
            build_document_structure(
                "# Finance\n\nThe approved delivery budget is USD 4.2 million.",
                source="beta.md",
            )
        )

        class EmptyKnowledgeStore:
            def insert(self, document: str, *, document_id: str | None = None) -> str:
                return document_id or "unused"

            def search(self, query: str, **kwargs) -> list[str]:
                return []

        with patch.dict(sys.modules, {"smolagents": fake_module}):
            tool = build_document_search_tool(
                EmptyKnowledgeStore(),
                {"beta.md": structure},
            )
            result = tool.forward("beta.md", "approved budget")

        # 原因：真实多文件测试中 embedding 阈值漏掉了包含完全相同词组的短文档。
        # 作用：证明回退只读取指定文件的匹配 chunk，并保留可追踪的本地来源。
        self.assertIn("USD 4.2 million", result)
        self.assertIn("[beta.md / Finance]", result)

    def test_collection_evidence_keeps_first_and_last_source_with_verified_manifest(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        documents = {}
        summaries = {}
        for index in range(30):
            source = f"lesson-{index + 1:02d}.md"
            structure = chunk_document_structure(
                build_document_structure(
                    f"# Lesson {index + 1}\n\n"
                    + (f"Distinct evidence for lesson {index + 1}. " * 80),
                    source=source,
                )
            )
            documents[source] = structure
            summaries[source] = summarize_document(structure)

        with patch.dict(sys.modules, {"smolagents": fake_module}):
            tool = build_document_collection_summary_tool(
                summaries,
                documents=documents,
                query="Compare every lesson and use specific evidence.",
            )
        observation = tool.forward()

        # 原因：旧实现把所有 per-source 预算用完后再做 prefix truncate，尾部来源必然消失。
        # 作用：manifest、首尾 source 和逐来源 chunk 引用同时存在，证明没有伪全覆盖。
        marker = observation.splitlines()[0]
        self.assertTrue(marker.startswith("QWOPUS_SOURCE_COVERAGE="))
        self.assertEqual(len(json.loads(marker.split("=", 1)[1])), 30)
        self.assertIn("# File: lesson-01.md", observation)
        self.assertIn("# File: lesson-30.md", observation)
        self.assertIn("chunk_id=", observation)

    def test_collection_evidence_reserves_each_lessons_exact_topic_and_scripture(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        lesson_facts = {
            29: (
                "题目：我只在乎谁？",
                "--- 不只求自己的事，更求耶稣的事（中）",
                "经文：腓立比书2章21节",
            ),
            30: (
                "题目：我只在乎谁",
                "--- 一个靠谱的人，是怎样炼成的（下）",
                "经文：腓立比书2章22-24节",
            ),
            31: (
                "题目：成熟的人，如何活在多重关系里（上）",
                "",
                "经文：腓立比书2章25节",
            ),
        }
        documents = {}
        summaries = {}
        for lesson in range(21, 34):
            source = f"腓立比书查经第{lesson}课.docx"
            topic, continuation, scripture = lesson_facts.get(
                lesson,
                (
                    f"题目：第{lesson}课的唯一主题",
                    "",
                    f"经文：测试书2章{lesson}节",
                ),
            )
            metadata_lines = "\n".join(
                line
                for line in (
                    f"腓立比书查经第{lesson}课",
                    topic,
                    continuation,
                    scripture,
                )
                if line
            )
            structure = chunk_document_structure(
                build_document_structure(
                    metadata_lines
                    + "\n\n"
                    + (f"这是第{lesson}课独有的正文证据。 " * 240),
                    source=source,
                )
            )
            documents[source] = structure
            summaries[source] = summarize_document(structure)

        budget = TokenBudgetManager(
            context_window=8192,
            output_reserve=2048,
            system_reserve=1024,
            history_reserve=512,
            safety_reserve=1024,
        )
        with patch.dict(sys.modules, {"smolagents": fake_module}):
            tool = build_document_collection_summary_tool(
                summaries,
                documents=documents,
                query="逐课比较全部文件的题目、经文和正文主题。",
                budget_manager=budget,
            )
        observation = tool.forward()

        def source_block(source: str) -> str:
            start = observation.index(f"# File: {source}")
            end = observation.find("\n\n# File:", start + 1)
            return observation[start:] if end == -1 else observation[start:end]

        # 原因：29～31课共享相邻段落和相似系列名，泛化摘要很容易把三课复制成同一课。
        # 作用：即使小窗口迫使 overview 大幅压缩，每课原始题目与经文仍被固定保留，
        # 且事实锚点先于可截断的正文 evidence/overview。
        for lesson, (topic, continuation, scripture) in lesson_facts.items():
            source = f"腓立比书查经第{lesson}课.docx"
            block = source_block(source)
            self.assertIn(f"topic_line: {topic}", block)
            if continuation:
                self.assertIn(f"topic_continuation: {continuation}", block)
            self.assertIn(f"scripture_line: {scripture}", block)
            self.assertLess(block.index("SOURCE_FACTS"), block.index("QUERY_RELEVANT_EVIDENCE"))

        lesson_29 = source_block("腓立比书查经第29课.docx")
        lesson_30 = source_block("腓立比书查经第30课.docx")
        lesson_31 = source_block("腓立比书查经第31课.docx")
        self.assertNotIn("scripture_line: 经文：腓立比书2章22-24节", lesson_29)
        self.assertNotIn("scripture_line: 经文：腓立比书2章25节", lesson_30)
        self.assertNotIn("一个靠谱的人，是怎样炼成的", lesson_31)
        self.assertIn("Never infer, renumber, reconstruct", observation)
        self.assertIn("Similar adjacent lessons remain distinct", observation)
        self.assertIn("QWOPUS_EXPLICIT_RUBRIC_FOUND=false", observation)
        self.assertLessEqual(estimate_tokens(observation), budget.synthesis_budget)

    def test_excel_schema_tool_bounds_safe_context(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)

        with patch.dict(sys.modules, {"smolagents": fake_module}):
            tool = build_excel_schema_tool(
                {"sales.xlsx": "schema and samples"},
                max_tokens=2,
            )

        self.assertTrue(tool.forward("sales.xlsx").startswith("schema"))
        self.assertIn("truncated", tool.forward("sales.xlsx"))

    def test_excel_analysis_tool_executes_restricted_pandas_locally(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame({"region": ["East", "West", "East"], "revenue": [10, 20, 30]}).to_excel(
                path, index=False
            )

            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_excel_analysis_tool({"sales.xlsx": path})

            # 原因：Excel Tool 必须执行 Agent 提出的 pandas 计算，而不是把整表返回模型。
            # 作用：用真实工作簿证明 Tool 最终只返回本地聚合结果。
            result = tool.forward(
                "sales.xlsx",
                'df = dfs["Sheet1"]\nresult = df.groupby("region")["revenue"].sum().reset_index()',
            )

            with self.assertRaisesRegex(ValueError, "already loaded in dfs"):
                tool.forward(
                    "sales.xlsx",
                    'df = pd.read_excel("sales.xlsx")\nresult = df',
                )

                self.assertIn("East", result)
                self.assertIn("40", result)

    def test_excel_analysis_uses_single_line_names_for_multiline_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiline.xlsx"
            pd.DataFrame(
                [
                    ["Area", "Working households\n(per cent)"],
                    ["North", 60.0],
                    ["South", 40.0],
                ]
            ).to_excel(path, index=False, header=False)

            tool = build_excel_analysis_tool({"multiline.xlsx": path})
            result = tool.forward(
                file_name="multiline.xlsx",
                code=(
                    'df = dfs["Sheet1"]\n'
                    'result = df[["Area", "Working households (per cent)"]]'
                ),
            )

            # 原因：模型只能复制 schema 中可见的单行列名，无法可靠重建隐藏换行。
            # 作用：锁定多行 Excel 表头在 schema 与执行边界使用同一名称。
            self.assertIn("Working households (per cent)", result)
            self.assertIn("60", result)
        self.assertIn('df = dfs["exact sheet or table name"]', tool.description)

    def test_excel_analysis_tool_exposes_secondary_table_regions(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multiple.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Data"
            sheet.append(["region", "revenue"])
            sheet.append(["East", 10])
            sheet.append([])
            sheet.append(["team", "tickets"])
            sheet.append(["Alpha", 7])
            workbook.save(path)

            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_excel_analysis_tool({"multiple.xlsx": path})

            # 原因：现实工作簿常在同一工作表纵向放置多张表，主表之外的数据也必须可计算。
            # 作用：锁定 schema 所展示的 Sheet::table_N 名称在 pandas 沙箱中确实可访问。
            result = tool.forward(
                "multiple.xlsx",
                'df = dfs["Data::table_2"]\nresult = df["tickets"].sum()',
            )

        self.assertIn("| result |", result)
        self.assertIn("| 7 |", result)

    def test_excel_statistics_tool_uses_only_approved_workbook_names(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.xlsx"
            pd.DataFrame(
                {
                    "student": ["A", "B", "C", "D", "E"],
                    "score": [10, 11, 12, 13, 100],
                }
            ).to_excel(path, index=False)

            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_excel_statistics_tool({"scores.xlsx": path})

            result = tool.forward(
                file_name="scores.xlsx",
                table_name="Sheet1",
                method="iqr_outliers",
                value_columns=["score"],
                label_columns=["student"],
                group_column=None,
                scope_table_name=None,
                scope_data_key=None,
                scope_lookup_key=None,
                scope_required_columns=None,
                top_n=20,
                threshold=1.5,
            )

            # 原因：统计 Skill 复用本地路径时不能扩大 Agent 的文件访问范围。
            # 作用：确认模型只提交获准文件名，适配器负责注入真实路径。
            self.assertIn("| E |", result)
            self.assertIn("1.5 x IQR", result)
            with self.assertRaisesRegex(ValueError, "Unknown file_name"):
                tool.forward(
                    file_name="../secret.xlsx",
                    table_name="Sheet1",
                    method="describe",
                    value_columns=["score"],
                    label_columns=[],
                    group_column=None,
                    scope_table_name=None,
                    scope_data_key=None,
                    scope_lookup_key=None,
                    scope_required_columns=None,
                    top_n=20,
                    threshold=1.5,
                )

    def test_excel_modeling_tool_runs_reviewed_regression_locally(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regression.xlsx"
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6],
                    "y": [3.1, 4.9, 7.2, 8.8, 11.1, 12.9],
                }
            ).to_excel(path, index=False)
            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_excel_modeling_tool({"regression.xlsx": path})

            result = tool.forward(
                file_name="regression.xlsx",
                table_name="Sheet1",
                method="linear_regression",
                outcome_column="y",
                predictor_columns=["x"],
                group_column=None,
                confidence_level=0.95,
                include_posthoc=None,
            )

        # 原因：自动发现 Skill 不代表文件 Agent 已获得受控 Tool 入口。
        # 作用：验证获准文件名可执行 OLS，并把系数与模型指标表返回 smolagents。
        self.assertIn("Model summary", result)
        self.assertIn("Coefficients", result)
        self.assertIn("| x |", result)

    def test_minirag_tool_returns_bounded_search_results(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            minirag.insert("Qwopus stores local project knowledge.")

            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_minirag_search_tool(minirag)

            result = tool.forward("Qwopus knowledge")

        self.assertIn("MiniRAG Result 1", result)
        self.assertIn("local project knowledge", result)

    def test_minirag_tool_can_expose_a_distinct_global_scope_name(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = make_test_minirag(Path(tmpdir) / "global.jsonl")
            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_minirag_search_tool(
                    minirag,
                    tool_name="global_rag_search",
                    description="Search explicitly authorized global knowledge.",
                )

        # 原因：私有与全局 Tool 共用名称会造成 smolagents 注册冲突和不可审计调用。
        # 作用：确保适配器可以保留同一 Skill 实现，同时对 Agent 暴露不同权限范围。
        self.assertEqual(tool.name, "global_rag_search")
        self.assertIn("authorized global", tool.description)

    def test_graph_tool_returns_directed_path_and_source_evidence(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with tempfile.TemporaryDirectory() as tmpdir:
            index = KnowledgeGraphIndex(
                graph=PersistentKnowledgeGraph(Path(tmpdir) / "knowledge_graph.json"),
                extractor=RuleBasedGraphExtractor(),
            )
            index.insert(
                (
                    GraphChunk(
                        id="chunk-1",
                        document_id="doc-1",
                        source="ownership.pdf",
                        page="4",
                        content=(
                            "[[Company A|Organization]] -[owns]-> "
                            "[[Company B|Organization]]"
                        ),
                    ),
                )
            )
            phases: list[str] = []
            with patch.dict(sys.modules, {"smolagents": fake_module}):
                tool = build_graph_search_tool(index, progress_callback=phases.append)

            result = tool.forward("How is Company A related to Company B?")

        # 原因：聊天中的关系问题必须通过真实图遍历，而不是依赖向量结果碰巧包含整条路径。
        # 作用：锁定有向边、文件页码证据及 UI 检索进度三个可观察结果。
        self.assertIn("Company A -[owns]-> Company B", result)
        self.assertIn("ownership.pdf, page 4", result)
        self.assertEqual(phases, ["retrieving", "generating"])


if __name__ == "__main__":
    unittest.main()
