"""跨语种检索。

检索只使用已激活的对齐版本：先在该版本内解析语种间的对齐路径（可能需要
跨多个语种对组合映射），再把查询向量整体映射到目标空间后排序，保证一次
查询的所有比较都处在同一对齐版本和同一索引版本约定下。没有可靠连接的
语种对会带原因拒配，而不是回退到无意义的直接余弦比较。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .alignment import _finite_vector
from .product import cosine


class CrossLingualRejected(ValueError):
    def __init__(self, reason, code):
        super().__init__(reason)
        self.reason = reason
        self.code = code


def _pair_index(active):
    pairs = {}
    for row in active["plan"]["pairs"]:
        pairs[(row["source_language"], row["target_language"])] = row
    return pairs


def _resolve_path(pairs, source_language, target_language):
    """在已接受的语种对上做 BFS；无可靠连接时显式拒配。"""
    if source_language == target_language:
        return [source_language]
    accepted = {(left, right) for (left, right), row in pairs.items() if row["status"] == "accepted"}
    queue = deque([[source_language]])
    seen = {source_language}
    while queue:
        path = queue.popleft()
        for left, right in sorted(accepted):
            if left != path[-1]:
                continue
            if right == target_language:
                return path + [right]
            if right not in seen:
                seen.add(right)
                queue.append(path + [right])
    reach = active_reachability_note(accepted, source_language)
    raise CrossLingualRejected(
        f"语种 {source_language} 与 {target_language} 之间没有可靠对齐连接，不能跨越；{reach}",
        code="unreachable-language",
    )


def active_reachability_note(accepted_edges, language):
    outgoing = sorted(right for (left, right) in accepted_edges if left == language)
    if not outgoing:
        return f"{language} 没有任何已接受的出向对齐"
    return f"{language} 当前仅有到 {', '.join(outgoing)} 的直接连接"


def _load_pair(pairs, left, right):
    row = pairs.get((left, right))
    if row is None or row["status"] != "accepted":
        raise CrossLingualRejected(
            f"语种对 {left}->{right} 未建立可靠对齐", code="unreachable-language"
        )
    return row


def mapped_query(active, source_language, target_language, vector):
    """返回映射后的查询向量及它依赖的全部语种对与可信范围。"""
    pairs = _pair_index(active)
    path = _resolve_path(pairs, source_language, target_language)
    query = _finite_vector(vector)
    hops = []
    bands = []
    for left, right in zip(path, path[1:]):
        pair = _load_pair(pairs, left, right)
        if query.size != pair["source_dimension"]:
            raise CrossLingualRejected(
                f"查询向量维数 {query.size} 与对齐 {left}->{right} 要求的 {pair['source_dimension']} 不一致",
                code="dimension-mismatch",
            )
        matrix = np.asarray(pair["matrix"], dtype=float)
        query = query @ matrix
        hops.append(
            {
                "pair": f"{left}->{right}",
                "candidate": pair["selected_candidate"],
                "source_index_version": pair["source_index_version"],
                "target_index_version": pair["target_index_version"],
                "pair_fingerprint": pair["pair_fingerprint"],
            }
        )
        bands.append(pair["trust_band"])
    # 多跳路径的可信范围取各跳的保守包络，并标注跳数带来的额外不确定。
    trust_band = {
        "low_90": min(band["low_90"] for band in bands),
        "median": bands[0]["median"] if len(bands) == 1 else None,
        "high_90": max(band["high_90"] for band in bands),
        "credible_threshold": min(band["low_90"] for band in bands),
        "hops": len(bands),
        "basis": "单跳为留家族验证的正确匹配余弦分布；多跳取各跳保守包络，不做独立假设",
    }
    return query, hops, trust_band, path


def cross_language_search(snapshot, active, document_id, target_language, limit=5, blocked_items=None):
    """跨语种素材检索，沿用单语检索的语种权限与 collection 权限约定。

    ``blocked_items`` 是当前消费约束中的禁用资料，检索前一并剔除；查询资料
    本身被禁用时按无权限处理。
    """
    blocked = set(blocked_items or ())
    documents = {row["id"]: row for row in snapshot["tables"]["documents"]}
    vectors = {row["id"]: row for row in snapshot["tables"]["vectors"]}
    if document_id not in documents or document_id not in vectors:
        raise CrossLingualRejected("检索资料不存在", code="unknown-document")
    query_doc = documents[document_id]
    permission = snapshot["tables"]["permissions"][0]
    allowed_languages = set(permission["allowed_languages"])
    allowed_collections = set(permission["allowed_collections"])
    if (
        query_doc["language"] not in allowed_languages
        or query_doc["collection"] not in allowed_collections
        or document_id in blocked
    ):
        raise CrossLingualRejected("没有访问该资料的权限", code="forbidden")
    if target_language not in allowed_languages:
        raise CrossLingualRejected(
            f"权限未覆盖目标语种 {target_language}", code="forbidden-language"
        )
    if target_language == query_doc["language"]:
        raise CrossLingualRejected("目标语种与查询语种相同，请使用单语检索", code="same-language")

    source_vector_row = vectors[document_id]
    mapped, hops, trust_band, path = mapped_query(
        active, query_doc["language"], target_language, source_vector_row["values"]
    )

    results = []
    for identifier, doc in documents.items():
        if doc["language"] != target_language or doc["collection"] not in allowed_collections:
            continue
        if identifier in blocked:
            continue
        vector_row = vectors[identifier]
        # 索引版本约定：只和对齐声明过的目标索引版本比较。
        last_hop = hops[-1]
        if vector_row["index_version"] != last_hop["target_index_version"]:
            continue
        score = cosine(mapped.tolist(), vector_row["values"])
        if score is None:
            continue
        # 可信下限之下表示该候选落在留验正确匹配分布之外，需人工复核。
        credible = score >= trust_band["credible_threshold"]
        results.append(
            {
                "document": identifier,
                "language": doc["language"],
                "score": round(score, 9),
                "credible": credible,
                "index_version": vector_row["index_version"],
                "alignment_id": active["alignment_id"],
                "alignment_version": active["alignment_version"],
                "alignment_path": path,
                "trust_band": trust_band,
            }
        )
    return {
        "query": document_id,
        "source_language": query_doc["language"],
        "target_language": target_language,
        "alignment_id": active["alignment_id"],
        "alignment_version": active["alignment_version"],
        "alignment_path": path,
        "alignment_hops": hops,
        "trust_band": trust_band,
        "results": sorted(results, key=lambda row: (-row["score"], row["document"]))[:limit],
    }
