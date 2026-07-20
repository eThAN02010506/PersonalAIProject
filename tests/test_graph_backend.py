import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_models import EntityRecord, GraphEvidence, RelationRecord


def _evidence(document_id: str, text: str) -> GraphEvidence:
    return GraphEvidence(
        document_id=document_id,
        source=f"{document_id}.txt",
        chunk_id=f"chunk-{document_id}",
        text=text,
    )


class PersistentKnowledgeGraphTests(unittest.TestCase):
    def test_graph_persists_entities_relations_and_evidence(self) -> None:
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "knowledge_graph.json"
            graph = PersistentKnowledgeGraph(storage_path)
            graph.upsert_entity(
                EntityRecord(
                    id="company-a",
                    canonical_name="Company A",
                    entity_type="Organization",
                    evidence=(_evidence("doc-a", "Company A owns Company B."),),
                )
            )
            graph.upsert_entity(
                EntityRecord(
                    id="company-b",
                    canonical_name="Company B",
                    entity_type="Organization",
                    evidence=(_evidence("doc-a", "Company A owns Company B."),),
                )
            )
            graph.upsert_relation(
                RelationRecord(
                    id="relation-owns",
                    source_id="company-a",
                    relation="owns",
                    target_id="company-b",
                    confidence=0.98,
                    evidence=(_evidence("doc-a", "Company A owns Company B."),),
                )
            )

            reloaded = PersistentKnowledgeGraph(storage_path)

            self.assertEqual(reloaded.entity_count, 2)
            self.assertEqual(reloaded.relation_count, 1)
            self.assertEqual(reloaded.get_relation("relation-owns").evidence[0].source, "doc-a.txt")

    def test_graph_returns_three_hop_path_with_original_relation_directions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            graph = PersistentKnowledgeGraph(Path(tmpdir) / "knowledge_graph.json")
            for entity_id, name in (
                ("a", "Company A"),
                ("b", "Company B"),
                ("c", "Project C"),
                ("d", "600 million"),
            ):
                graph.upsert_entity(
                    EntityRecord(
                        id=entity_id,
                        canonical_name=name,
                        entity_type="Entity",
                        evidence=(_evidence("path", name),),
                    )
                )
            for relation_id, source, relation, target in (
                ("r1", "a", "owns", "b"),
                ("r2", "b", "participates_in", "c"),
                ("r3", "c", "has_budget", "d"),
            ):
                graph.upsert_relation(
                    RelationRecord(
                        id=relation_id,
                        source_id=source,
                        relation=relation,
                        target_id=target,
                        confidence=1.0,
                        evidence=(_evidence("path", relation),),
                    )
                )

            paths = graph.paths_between(("a",), ("d",), max_hops=3)

            self.assertEqual(len(paths), 1)
            self.assertEqual(
                paths[0].entity_names,
                ("Company A", "Company B", "Project C", "600 million"),
            )
            self.assertEqual(
                tuple(relation.relation for relation in paths[0].relations),
                ("owns", "participates_in", "has_budget"),
            )


if __name__ == "__main__":
    unittest.main()
