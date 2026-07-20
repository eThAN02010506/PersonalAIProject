import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_models import EntityRecord, GraphEvidence, RelationRecord
from qwopus_agent.services.knowledge_graph_service import KnowledgeGraphService


class KnowledgeGraphServiceTests(unittest.TestCase):
    def test_snapshot_dot_and_evidence_rows_are_bounded_and_traceable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knowledge_graph.json"
            graph = PersistentKnowledgeGraph(path)
            evidence = GraphEvidence(
                document_id="doc-1",
                source="facts.pdf",
                page="7",
                chunk_id="chunk-1",
                text='Company "A" owns Company B.',
            )
            graph.upsert_entity(
                EntityRecord(
                    id="a",
                    canonical_name='Company "A"',
                    entity_type="Organization",
                    evidence=(evidence,),
                )
            )
            graph.upsert_entity(
                EntityRecord(
                    id="b",
                    canonical_name="Company B",
                    entity_type="Organization",
                    evidence=(evidence,),
                )
            )
            graph.upsert_relation(
                RelationRecord(
                    id="owns",
                    source_id="a",
                    relation="owns",
                    target_id="b",
                    confidence=1.0,
                    evidence=(evidence,),
                )
            )
            service = KnowledgeGraphService(path)

            snapshot = service.snapshot(entity_type="Organization", max_nodes=10)
            dot = service.to_dot(snapshot)
            rows = service.evidence_rows(snapshot)

            self.assertEqual(service.entity_types(), ["Organization"])
            self.assertEqual(len(snapshot.nodes), 2)
            self.assertIn('"a" -> "b"', dot)
            self.assertIn('Company \\"A\\"', dot)
            self.assertEqual(rows[0]["source"], "facts.pdf")
            self.assertEqual(rows[0]["page"], "7")


if __name__ == "__main__":
    unittest.main()
