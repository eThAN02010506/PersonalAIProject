import json
import unittest

from qwopus_agent.llm.base import BaseLLM, ChatMessage, LLMResponse
from qwopus_agent.memory.graph_extraction import (
    CompositeGraphExtractor,
    LLMGraphExtractor,
    RuleBasedGraphExtractor,
)
from qwopus_agent.memory.graph_models import GraphChunk


class _FakeLLM(BaseLLM):
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del messages, temperature, max_tokens
        return LLMResponse(content=json.dumps(self.payload), model="fake")


class _FailingExtractor:
    def extract(self, chunks):
        del chunks
        raise RuntimeError("offline")


def _chunk(content: str) -> GraphChunk:
    return GraphChunk(
        id="chunk-1",
        document_id="doc-1",
        source="facts.pdf",
        page="2",
        content=content,
    )


class GraphExtractionTests(unittest.TestCase):
    def test_rule_extractor_preserves_source_metadata(self) -> None:
        extraction = RuleBasedGraphExtractor().extract(
            [
                _chunk(
                    "[[Aurora Holdings|Organization]] -[owns]-> "
                    "[[Blue Harbor Ltd|Organization]]"
                )
            ]
        )

        self.assertEqual(len(extraction.entities), 2)
        self.assertEqual(len(extraction.relations), 1)
        self.assertEqual(extraction.relations[0].evidence.source, "facts.pdf")
        self.assertEqual(extraction.relations[0].evidence.page, "2")

    def test_llm_extractor_drops_relation_without_supporting_quote(self) -> None:
        content = "Aurora Holdings owns Blue Harbor Ltd."
        payload = {
            "entities": [],
            "relations": [
                {
                    "source": "Aurora Holdings",
                    "relation": "owns",
                    "target": "Blue Harbor Ltd",
                    "confidence": 0.98,
                    "chunk_id": "chunk-1",
                    "evidence": content,
                },
                {
                    "source": "Aurora Holdings",
                    "relation": "founded",
                    "target": "Project Lantern",
                    "confidence": 0.99,
                    "chunk_id": "chunk-1",
                    "evidence": "Aurora Holdings founded Project Lantern.",
                },
            ],
        }

        extraction = LLMGraphExtractor(lambda: _FakeLLM(payload)).extract([_chunk(content)])

        self.assertEqual(len(extraction.relations), 1)
        self.assertEqual(extraction.relations[0].target, "Blue Harbor Ltd")

    def test_llm_extractor_rejects_fabricated_chunk_id(self) -> None:
        content = "Company A owns Company B."
        payload = {
            "entities": [
                {
                    "name": "Company A",
                    "entity_type": "Organization",
                    "aliases": [],
                    "description": "",
                    "chunk_id": "missing-chunk",
                    "evidence": content,
                }
            ],
            "relations": [],
        }

        extraction = LLMGraphExtractor(lambda: _FakeLLM(payload)).extract([_chunk(content)])

        self.assertEqual(extraction.entities, ())

    def test_composite_extractor_keeps_rule_results_when_llm_fails(self) -> None:
        extractor = CompositeGraphExtractor(
            extractors=(_FailingExtractor(), RuleBasedGraphExtractor())
        )

        extraction = extractor.extract(
            [_chunk("[[Company A|Organization]] -[owns]-> [[Company B|Organization]]")]
        )

        self.assertEqual(len(extraction.relations), 1)


if __name__ == "__main__":
    unittest.main()
