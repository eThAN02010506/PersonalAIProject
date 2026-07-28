import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qwopus_agent.documents import (
    build_document_structure,
    chunk_document_structure,
    summarize_document,
)
from qwopus_agent.integrations.smolagents_tools import (
    TavilySearchConfig,
    build_document_outline_tool,
    build_document_search_tool,
    build_document_section_tool,
    build_document_summary_tool,
    build_excel_analysis_tool,
    build_excel_schema_tool,
    build_graph_search_tool,
    build_minirag_search_tool,
    build_tavily_search_tool,
)
from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphChunk
from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex
from tests.minirag_fakes import make_test_minirag


class FakeTool:
    def __init__(self, *args, **kwargs) -> None:
        pass


class SmolagentsToolsTests(unittest.TestCase):
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

        self.assertIn("East", result)
        self.assertIn("40", result)

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
