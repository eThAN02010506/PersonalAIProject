"""Render and diversify MiniRAG retrieval results."""

from __future__ import annotations

import re
from collections.abc import Sequence

from qwopus_agent.memory.graph_models import GraphPath
from qwopus_agent.memory.minirag_records import KnowledgeChunk

SEARCH_TOP_K = 12


def render_search_result(chunk: KnowledgeChunk) -> str:
    """Render one semantic chunk with a traceable source citation."""
    if chunk.source == "unknown" and chunk.page is None:
        if chunk.section_path != ("Document content",):
            if chunk.content == chunk.section_path[-1]:
                return f"# {chunk.content}"
            return f"# {chunk.section_path[-1]}\n{chunk.content}"
        return chunk.content
    citation = f"Source: {chunk.source}"
    if chunk.page is not None:
        citation += (
            f" | Pages: {chunk.page}-{chunk.page_end}"
            if chunk.page_end and chunk.page_end != chunk.page
            else f" | Page: {chunk.page}"
        )
    if chunk.section_path != ("Document content",):
        citation += f" | Section: {' / '.join(chunk.section_path)}"
    return f"[{citation}]\n{chunk.content}"


def embedding_content(chunk: KnowledgeChunk) -> str:
    """Include section identity in text sent to the embedding backend."""
    if chunk.section_path == ("Document content",):
        return chunk.content
    return f"Section: {' / '.join(chunk.section_path)}\n\n{chunk.content}"


def source_matches_query(query: str, source: str) -> bool:
    """Match a query against persisted source metadata without changing body vectors."""
    if source == "unknown":
        return False
    normalized_query = re.sub(r"\W+", "", query.casefold())
    normalized_source = re.sub(r"\W+", "", source.casefold())
    if normalized_query and normalized_query in normalized_source:
        return True

    query_tokens = _source_tokens(query)
    source_tokens = _source_tokens(source)
    overlap = query_tokens & source_tokens
    # 原因：文件名进入正文向量会稀释短事实词的余弦分数，造成新的正文漏召回。
    # 作用：用 metadata 补充文件名命中；两个短 token 或一个长 token 即可识别课次/文件名。
    return len(overlap) >= 2 or any(len(token) >= 6 for token in overlap)


def render_graph_search_result(path: GraphPath) -> str:
    """Render one graph path with relation and evidence provenance."""
    names_by_id = dict(zip(path.entity_ids, path.entity_names, strict=False))
    relation_lines = [
        (
            f"- {names_by_id[relation.source_id]} -[{relation.relation}]-> "
            f"{names_by_id[relation.target_id]}"
        )
        for relation in path.relations
    ]
    evidence_lines: list[str] = []
    for evidence in path.evidence:
        citation = f"Source: {evidence.source}"
        if evidence.page is not None:
            citation += f" | Page: {evidence.page}"
        evidence_lines.append(f"- [{citation}] {evidence.text}")
    return (
        "[Knowledge Graph Path]\n"
        + "\n".join(relation_lines)
        + "\nEvidence:\n"
        + "\n".join(evidence_lines)
    )


def diverse_chunks(ranked_chunks: Sequence[KnowledgeChunk]) -> list[KnowledgeChunk]:
    """Keep vector order while preventing duplicate evidence from dominating."""
    primary: list[KnowledgeChunk] = []
    overflow: list[KnowledgeChunk] = []
    seen_documents: set[str] = set()
    accepted_fingerprints: list[set[str]] = []
    for chunk in ranked_chunks:
        fingerprint = _content_fingerprint(chunk.content)
        if any(
            _jaccard_similarity(fingerprint, accepted) >= 0.82
            for accepted in accepted_fingerprints
        ):
            continue
        accepted_fingerprints.append(fingerprint)
        if chunk.document_id in seen_documents:
            overflow.append(chunk)
            continue
        seen_documents.add(chunk.document_id)
        primary.append(chunk)

    # 原因：大型文件会产生很多相似 chunk，可能挤掉其他文档的候选片段。
    # 作用：优先保留每份文档的最佳命中，再按原向量排名补足上下文数量。
    return (primary + overflow)[:SEARCH_TOP_K]


def _content_fingerprint(content: str) -> set[str]:
    normalized = re.sub(r"\W+", "", content.casefold())
    if len(normalized) < 5:
        return {normalized} if normalized else set()
    return {normalized[index : index + 5] for index in range(len(normalized) - 4)}


def _source_tokens(value: str) -> set[str]:
    lowered = value.casefold()
    latin_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", lowered))
    chinese_bigrams = {
        chinese[index : index + 2]
        for index in range(max(len(chinese) - 1, 0))
    }
    return latin_tokens | chinese_bigrams


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
