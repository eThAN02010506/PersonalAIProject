import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphChunk
from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex


def _chunk(document_id: str, source: str, content: str) -> GraphChunk:
    return GraphChunk(
        id=f"chunk-{document_id}",
        document_id=document_id,
        source=source,
        content=content,
    )


class KnowledgeGraphIndexTests(unittest.TestCase):
    def test_cross_document_three_hop_query_survives_restart(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "knowledge_graph.json"
            index = KnowledgeGraphIndex(
                PersistentKnowledgeGraph(storage_path),
                RuleBasedGraphExtractor(),
            )
            chunks = (
                _chunk(
                    "doc-1",
                    "ownership.pdf",
                    "[[Aurora Holdings|Organization]] -[owns]-> "
                    "[[Blue Harbor Ltd|Organization]]",
                ),
                _chunk(
                    "doc-2",
                    "project.docx",
                    "[[Blue Harbor Ltd|Organization]] -[participates_in]-> "
                    "[[Project Lantern|Project]]",
                ),
                _chunk(
                    "doc-3",
                    "budget.xlsx",
                    "[[Project Lantern|Project]] -[has_budget]-> "
                    "[[USD 6 million|Amount]]",
                ),
            )
            for chunk in chunks:
                index.insert((chunk,))

            paths = index.search(
                "How is Aurora Holdings related to USD 6 million?",
                max_hops=3,
            )

            self.assertEqual(len(paths), 1)
            self.assertEqual(
                tuple(relation.relation for relation in paths[0].relations),
                ("owns", "participates_in", "has_budget"),
            )
            self.assertEqual(
                {evidence.source for evidence in paths[0].evidence},
                {"ownership.pdf", "project.docx", "budget.xlsx"},
            )

            reloaded = KnowledgeGraphIndex(
                PersistentKnowledgeGraph(storage_path),
                RuleBasedGraphExtractor(),
            )
            reloaded_paths = reloaded.paths_between(
                "Aurora Holdings",
                "USD 6 million",
                max_hops=3,
            )
            self.assertEqual(len(reloaded_paths), 1)

    def test_one_mentioned_entity_returns_bounded_neighborhood(self) -> None:
        with TemporaryDirectory() as tmpdir:
            index = KnowledgeGraphIndex(
                PersistentKnowledgeGraph(Path(tmpdir) / "knowledge_graph.json"),
                RuleBasedGraphExtractor(),
            )
            index.insert(
                (
                    _chunk(
                        "doc-1",
                        "facts.txt",
                        "[[Company A|Organization]] -[owns]-> "
                        "[[Company B|Organization]]",
                    ),
                )
            )

            paths = index.search("Tell me about Company A", max_hops=1, limit=1)

            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].entity_names, ("Company A", "Company B"))


if __name__ == "__main__":
    unittest.main()
