import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from qwopus_agent.memory.entity_resolver import EntityResolver
from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_models import EntityCandidate, GraphEvidence


def _candidate(
    name: str,
    entity_type: str,
    document_id: str,
    *,
    aliases: tuple[str, ...] = (),
) -> EntityCandidate:
    return EntityCandidate(
        name=name,
        entity_type=entity_type,
        aliases=aliases,
        evidence=GraphEvidence(
            document_id=document_id,
            source=f"{document_id}.txt",
            chunk_id=f"chunk-{document_id}",
            text=name,
        ),
    )


class _SemanticEmbedding:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if text in {"International Business Machines", "IBM"}:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class EntityResolverTests(unittest.TestCase):
    def test_aliases_merge_cross_document_evidence_and_survive_restart(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knowledge_graph.json"
            graph = PersistentKnowledgeGraph(path)
            resolver = EntityResolver(graph)

            first = resolver.resolve(_candidate("OpenAI", "Organization", "doc-1"))
            second = resolver.resolve(
                _candidate(
                    "Open AI",
                    "Organization",
                    "doc-2",
                    aliases=("开放人工智能",),
                )
            )

            self.assertEqual(first.id, second.id)
            self.assertEqual(graph.entity_count, 1)
            self.assertEqual(len(second.evidence), 2)
            self.assertIn("开放人工智能", second.aliases)
            self.assertEqual(PersistentKnowledgeGraph(path).entity_count, 1)

    def test_same_name_with_incompatible_types_does_not_merge(self) -> None:
        with TemporaryDirectory() as tmpdir:
            graph = PersistentKnowledgeGraph(Path(tmpdir) / "knowledge_graph.json")
            resolver = EntityResolver(graph)

            organization = resolver.resolve(_candidate("Apple", "Organization", "doc-1"))
            food = resolver.resolve(_candidate("Apple", "Food", "doc-2"))

            self.assertNotEqual(organization.id, food.id)
            self.assertEqual(graph.entity_count, 2)

    def test_semantic_backend_merges_high_confidence_name_variant(self) -> None:
        with TemporaryDirectory() as tmpdir:
            graph = PersistentKnowledgeGraph(Path(tmpdir) / "knowledge_graph.json")
            resolver = EntityResolver(graph, embedding_backend=_SemanticEmbedding())

            first = resolver.resolve(
                _candidate("International Business Machines", "Organization", "doc-1")
            )
            second = resolver.resolve(_candidate("IBM", "Organization", "doc-2"))

            self.assertEqual(first.id, second.id)
            self.assertIn("IBM", second.aliases)


if __name__ == "__main__":
    unittest.main()
