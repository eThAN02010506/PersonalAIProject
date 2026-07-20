import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphChunk
from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.graph_search import GraphSearchSkill


class GraphSearchSkillTests(unittest.TestCase):
    def test_skill_returns_structured_path_and_citations(self) -> None:
        with TemporaryDirectory() as tmpdir:
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
            skill = GraphSearchSkill(index=index)

            response = asyncio.run(
                skill.run(
                    SkillRequest(
                        query="How is Company A related to Company B?",
                        arguments={"max_hops": 99, "limit": 1},
                    )
                )
            )

            self.assertTrue(response.success)
            self.assertEqual(len(response.data["paths"]), 1)
            self.assertIn("Company A -[owns]-> Company B", response.content)
            self.assertIn("ownership.pdf, page 4", response.content)

    def test_skill_returns_successful_empty_result(self) -> None:
        with TemporaryDirectory() as tmpdir:
            index = KnowledgeGraphIndex(
                graph=PersistentKnowledgeGraph(Path(tmpdir) / "knowledge_graph.json"),
                extractor=RuleBasedGraphExtractor(),
            )

            response = asyncio.run(
                GraphSearchSkill(index=index).run(SkillRequest(query="missing entity"))
            )

            self.assertTrue(response.success)
            self.assertEqual(response.data["paths"], [])


if __name__ == "__main__":
    unittest.main()
