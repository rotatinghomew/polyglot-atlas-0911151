"""跨语种向量空间对齐。

合作方交付的向量各自处于独立坐标空间，本模块只依据编辑确认的词义对应
（anchors 中 ``confirmed=true`` 的记录）估计线性映射；未确认或存在争议的
对应仅作为复核材料保留，绝不参与拟合，也不会被当成等价真值。

每个候选方案都要经过按文章家族分组的交叉验证（同一文章家族整体留一，
杜绝翻译对泄漏），并报告可信区间与塌缩/热门向量诊断。证据不足、无法
连通或诊断不过的语种对会明确拒绝，并给出拒配原因，而不是勉强给出映射。
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .data import fingerprint

METHOD_VERSION = "cross-lingual-alignment-v1"

# 允许进入拟合的最少确认对应数；低于该数量属于稀疏到不可信的证据。
MIN_CONFIRMED_ANCHORS = 6
# 分组交叉验证的最少折数；可见文章家族太少时不给结论。
MIN_CV_FOLDS = 3
# 每一留验折至少保留的训练对应数；全量证据仍受 MIN_CONFIRMED_ANCHORS 约束。
MIN_CV_TRAIN_ANCHORS = 3
BOOTSTRAP_SAMPLES = 400
BOOTSTRAP_SEED = 20260115

# 诊断阈值：只在高频词上看似对齐的方案必须在这里被拦下。
MIN_EFFECTIVE_RANK_RATIO = 0.75
MIN_TARGET_COVERAGE = 0.60
MIN_HUB_ENTROPY_RATIO = 0.75
# 留一家族验证时，正确家族成员进入前 k 的最低可接受水平。
MIN_RECALL_AT_1 = 0.75


class AlignmentRejected(ValueError):
    """语种对无法建立可靠对齐，原因记录在 ``reason`` 中。"""

    def __init__(self, code, reason, **details):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.details = details


def _finite_vector(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("向量必须是非空有限向量")
    return array


def _cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return None
    return float(np.dot(left, right) / denominator)


def _effective_rank(singular_values):
    values = np.asarray(singular_values, dtype=float)
    squared = values * values
    total = squared.sum()
    if total <= 0:
        return 0.0
    return float(total * total / (squared ** 2).sum())


def _normalized(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise AlignmentRejected("zero-vector", reason="存在零向量，无法比较方向")
    return matrix / norms


def _procrustes(source, target, weights):
    """带权正交 Procrustes：W 为正交阵，保持角度结构（WᵀW=I）。"""
    scale = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    matrix = (source * scale).T @ (target * scale)
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def _least_squares(source, target, weights):
    """带权最小二乘线性映射，允许旋转加缩放；秩不足时拒绝。"""
    scale = np.asarray(weights, dtype=float)
    weighted = source * np.sqrt(scale)[:, None]
    rhs = target * np.sqrt(scale)[:, None]
    gram = weighted.T @ weighted
    rank = int(np.linalg.matrix_rank(gram))
    if rank != gram.shape[0]:
        raise AlignmentRejected("rank-deficient", reason="加权设计矩阵秩不足，最小二乘映射不唯一")
    return np.linalg.solve(gram, weighted.T @ rhs)


CANDIDATE_BUILDERS = {
    "orthogonal-procrustes": _procrustes,
    "weighted-least-squares": _least_squares,
}


def _index_layout(snapshot):
    vectors = {}
    dimensions = {}
    index_versions = {}
    for row in snapshot["tables"]["vectors"]:
        values = _finite_vector(row["values"])
        vectors[row["id"]] = values
        dimensions.setdefault(row["language"], set()).add(values.size)
        index_versions.setdefault(row["language"], set()).add(row["index_version"])
    documents = {row["id"]: row for row in snapshot["tables"]["documents"]}
    for language, sizes in dimensions.items():
        if len(sizes) > 1 or len(index_versions[language]) > 1:
            raise AlignmentRejected(
                "inconsistent-index",
                reason=f"语种 {language} 内部索引版本或维数不一致，沿用单语索引约定先拒绝",
            )
    return vectors, documents, {lang: next(iter(sizes)) for lang, sizes in dimensions.items()}, {
        lang: next(iter(versions)) for lang, versions in index_versions.items()
    }


def _anchor_evidence(snapshot, documents):
    """把 anchors 拆成可拟合证据与复核材料，并记录每个对应为何这样处理。"""
    incoming = defaultdict(list)
    for row in snapshot["tables"]["anchors"]:
        incoming[row["right"]].append(row["id"])

    evidence = []
    review = []
    for row in snapshot["tables"]["anchors"]:
        left_doc = documents.get(row["left"])
        right_doc = documents.get(row["right"])
        entry = {
            "anchor_id": row["id"],
            "left": row["left"],
            "right": row["right"],
            "sense": row["sense"],
            "confidence": float(row["confidence"]),
            "confirmed": bool(row["confirmed"]),
            "reasons": [],
        }
        if left_doc is None or right_doc is None:
            entry["reasons"].append("一端资料在当前快照不可见")
        if not (0.0 < float(row["confidence"]) <= 1.0):
            entry["reasons"].append("置信度超出 (0,1] 范围")
        if not row["confirmed"]:
            # 未确认记录可能是一词多义或争议确认，只能进复核清单。
            entry["reasons"].append("编辑尚未确认，不能作为等价真值参与拟合")
        if entry["reasons"]:
            review.append(entry)
        else:
            # 同一目标被多个源指向（一词多义/多对一）时降低每条的权重，
            # 避免高频目标支配拟合；置信度仍保留为编辑先验。
            multiplicity = len(incoming[row["right"]])
            weight = float(row["confidence"]) / multiplicity
            evidence.append({**entry, "weight": weight, "multiplicity": multiplicity})
    return evidence, review


def _family_groups(snapshot, documents):
    groups = defaultdict(dict)
    for row in snapshot["tables"]["documents"]:
        groups[(row["family"], row["collection"])][row["language"]] = row["id"]
    return groups


def _pair_families(groups, source_language, target_language):
    families = defaultdict(list)
    for (family, _collection), members in groups.items():
        if source_language in members and target_language in members:
            families[family].append((members[source_language], members[target_language]))
    return families


def _evaluate(matrix, pairs, vectors, target_ids, target_matrix):
    """在给定（留家族）样本上计算相似度、排名指标。"""
    source_vectors = np.stack([vectors[left] for left, _ in pairs])
    target_vectors = np.stack([vectors[right] for _, right in pairs])
    mapped = source_vectors @ matrix
    pair_cosines = np.array(
        [cos for cos in (_cosine(a, b) for a, b in zip(mapped, target_vectors)) if cos is not None]
    )

    target_norm = _normalized(target_matrix)
    mapped_norm = _normalized(mapped)
    similarity = mapped_norm @ target_norm.T
    # 正确目标在全量目标语料中的排名。
    target_index = {identifier: position for position, identifier in enumerate(target_ids)}
    ranks = []
    for (_, right), scores in zip(pairs, similarity):
        order = np.argsort(-scores, kind="stable")
        ranks.append(int(np.where(order == target_index[right])[0][0]) + 1)
    ranks = np.array(ranks)
    metrics = {
        "pair_count": len(pairs),
        "mean_pair_cosine": round(float(pair_cosines.mean()), 6),
        "min_pair_cosine": round(float(pair_cosines.min()), 6),
        "recall_at_1": round(float(np.mean(ranks == 1)), 6),
        "recall_at_5": round(float(np.mean(ranks <= 5)), 6),
        "mean_rank": round(float(ranks.mean()), 6),
    }
    return metrics, pair_cosines


def _collapse_diagnostics(matrix, source_corpus_ids, vectors, target_matrix):
    """在全量源语料上映射后检查塌缩，而不是只看少量锚点。

    只在高频词上看似对齐的方案，其映射后有效秩、目标覆盖率和热门熵都会
    明显偏离健康空间，这些指标必须基于整个语料才能暴露问题。
    """
    mapped_corpus = np.stack([vectors[identifier] for identifier in source_corpus_ids]) @ matrix
    target_norm = _normalized(target_matrix)
    mapped_norm = _normalized(mapped_corpus)
    similarity = mapped_norm @ target_norm.T
    top_k = min(5, target_matrix.shape[0])
    top_columns = np.argpartition(-similarity, min(top_k, similarity.shape[1] - 1), axis=1)[:, :top_k]
    counts = np.bincount(top_columns.ravel(), minlength=target_matrix.shape[0])
    coverage = float(np.count_nonzero(counts) / target_matrix.shape[0])
    shares = counts / counts.sum() if counts.sum() else counts
    positive = shares[shares > 0]
    hub_entropy = float(-(positive * np.log(positive)).sum())
    entropy_ratio = float(hub_entropy / math.log(target_matrix.shape[0])) if target_matrix.shape[0] > 1 else 0.0
    source_singular = np.linalg.svd(mapped_norm, compute_uv=False)
    target_singular = np.linalg.svd(target_norm, compute_uv=False)
    rank_ratio = _effective_rank(source_singular) / max(_effective_rank(target_singular), 1e-12)
    return {
        "effective_rank_mapped": round(_effective_rank(source_singular), 6),
        "effective_rank_target": round(_effective_rank(target_singular), 6),
        "effective_rank_ratio": round(rank_ratio, 6),
        "target_coverage": round(coverage, 6),
        "hub_entropy_ratio": round(entropy_ratio, 6),
        "hub_max_share": round(float(counts.max() / max(counts.sum(), 1)), 6),
    }


def _quantile(samples, q):
    return round(float(np.quantile(np.asarray(samples, dtype=float), q)), 6)


def _family_cross_validation(builder, evidence, family_of, family_pairs, vectors, target_ids, target_matrix):
    """按文章家族分组的 K 折交叉验证，同一文章家族整体留一。"""
    evidence_families = sorted({family_of.get(item["left"]) for item in evidence if family_of.get(item["left"])})
    if len(evidence_families) < MIN_CV_FOLDS:
        raise AlignmentRejected("too-few-families", reason="可验证的文章家族不足，无法做分组交叉验证")
    fold_count = min(MIN_CV_FOLDS, len(evidence_families))
    folds = [set() for _ in range(fold_count)]
    for position, family in enumerate(evidence_families):
        folds[position % fold_count].add(family)

    held_cosines = []
    fold_records = []
    for held_families in folds:
        train = [item for item in evidence if family_of.get(item["left"]) not in held_families]
        # 留验折允许证据量低于全量门槛，但至少要保留有意义的拟合样本；
        # 达不到下限的折跳过，而不是放松几何或塌缩要求。
        if len(train) < MIN_CV_TRAIN_ANCHORS:
            continue
        matrix = builder(
            np.stack([vectors[item["left"]] for item in train]),
            np.stack([vectors[item["right"]] for item in train]),
            [item["weight"] for item in train],
        )
        held_pairs = [pair for family in held_families for pair in family_pairs.get(family, [])]
        if not held_pairs:
            continue
        metrics, cosines = _evaluate(matrix, held_pairs, vectors, target_ids, target_matrix)
        held_cosines.extend(cosines.tolist())
        fold_records.append({"held_families": sorted(held_families), **metrics})

    if len(fold_records) < MIN_CV_FOLDS:
        raise AlignmentRejected(
            "cv-evidence-too-sparse", reason="留一家族后确认对应低于最小证据量，验证不具代表性"
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base = np.asarray(held_cosines, dtype=float)
    bootstrap_means = np.array([rng.choice(base, size=base.size, replace=True).mean() for _ in range(BOOTSTRAP_SAMPLES)])
    validation = {
        "fold_count": len(fold_records),
        "folds": fold_records,
        "held_pair_count": int(base.size),
        "recall_at_1": round(float(np.mean([f["recall_at_1"] for f in fold_records])), 6),
        "recall_at_5": round(float(np.mean([f["recall_at_5"] for f in fold_records])), 6),
        "mean_held_cosine": round(float(base.mean()), 6),
        "min_held_cosine": round(float(base.min()), 6),
        "cosine_band_90": {"low": _quantile(base, 0.05), "high": _quantile(base, 0.95)},
        "mean_cosine_ci_90": {
            "low": _quantile(bootstrap_means, 0.05),
            "high": _quantile(bootstrap_means, 0.95),
        },
    }
    return validation, held_cosines


def _collapse_diagnosis(diagnostics, validation):
    failures = []
    if diagnostics["effective_rank_ratio"] < MIN_EFFECTIVE_RANK_RATIO:
        failures.append(
            f"映射后有效秩比 {diagnostics['effective_rank_ratio']} 低于 {MIN_EFFECTIVE_RANK_RATIO}，输出塌缩到少数方向"
        )
    if diagnostics["target_coverage"] < MIN_TARGET_COVERAGE:
        failures.append(
            f"目标语料覆盖率 {diagnostics['target_coverage']} 低于 {MIN_TARGET_COVERAGE}，结果集中在少数热门向量"
        )
    if diagnostics["hub_entropy_ratio"] < MIN_HUB_ENTROPY_RATIO:
        failures.append(
            f"热门向量熵比 {diagnostics['hub_entropy_ratio']} 低于 {MIN_HUB_ENTROPY_RATIO}，存在枢纽塌缩"
        )
    if validation["recall_at_1"] < MIN_RECALL_AT_1:
        failures.append(
            f"留家族验证 recall@1={validation['recall_at_1']} 低于 {MIN_RECALL_AT_1}，只在高频词上看似对齐"
        )
    return failures


def _score_trust_band(held_cosines):
    """编辑可见的可信分数区间：来自正确家族匹配的留验余弦分布。"""
    values = np.asarray(held_cosines, dtype=float)
    return {
        "low_90": _quantile(values, 0.05),
        "median": _quantile(values, 0.5),
        "high_90": _quantile(values, 0.95),
        "credible_threshold": _quantile(values, 0.05),
        "basis": "留文章家族交叉验证中正确家族匹配的余弦分布，不含训练用对应",
    }


def build_pair_alignment(snapshot, source_language, target_language):
    """建立单个语种对的对齐；证据或诊断不过时返回拒配记录而不是抛异常。"""
    vectors, documents, dimensions, index_versions = _index_layout(snapshot)
    if source_language not in dimensions or target_language not in dimensions:
        return {
            "status": "rejected",
            "source_language": source_language,
            "target_language": target_language,
            "rejection_reasons": [f"语种 {source_language if source_language not in dimensions else target_language} 在快照中没有向量"],
        }
    evidence, review = _anchor_evidence(snapshot, documents)
    pair_evidence = [
        item
        for item in evidence
        if documents[item["left"]]["language"] == source_language
        and documents[item["right"]]["language"] == target_language
    ]
    pair_review = [
        item
        for item in review
        if documents.get(item["left"], {}).get("language") == source_language
        and documents.get(item["right"], {}).get("language") == target_language
    ]

    reasons = []
    if len(pair_evidence) < MIN_CONFIRMED_ANCHORS:
        reasons.append(
            f"确认对应仅 {len(pair_evidence)} 条，少于最小证据量 {MIN_CONFIRMED_ANCHORS}，稀疏对应不足以估计线性映射"
        )

    groups = _family_groups(snapshot, documents)
    family_pairs = _pair_families(groups, source_language, target_language)
    family_of = {identifier: family for family, pairs in family_pairs.items() for identifier, _ in pairs}
    target_ids = sorted(identifier for identifier, doc in documents.items() if doc["language"] == target_language)
    target_matrix = np.stack([vectors[identifier] for identifier in target_ids])
    source_corpus_ids = sorted(identifier for identifier, doc in documents.items() if doc["language"] == source_language)

    candidate_reports = []
    chosen = None
    if not reasons:
        source_matrix = np.stack([vectors[item["left"]] for item in pair_evidence])
        target_vectors = np.stack([vectors[item["right"]] for item in pair_evidence])
        weights = [item["weight"] for item in pair_evidence]
        full_pairs = [(item["left"], item["right"]) for item in pair_evidence]
        for name, builder in CANDIDATE_BUILDERS.items():
            try:
                matrix = builder(source_matrix, target_vectors, weights)
                validation, held_cosines = _family_cross_validation(
                    builder, pair_evidence, family_of, family_pairs, vectors, target_ids, target_matrix
                )
                fit_metrics, _ = _evaluate(matrix, full_pairs, vectors, target_ids, target_matrix)
                diagnostics = _collapse_diagnostics(
                    matrix, source_corpus_ids, vectors, target_matrix
                )
                failures = _collapse_diagnosis(diagnostics, validation)
                report = {
                    "candidate": name,
                    "status": "accepted" if not failures else "rejected",
                    "fit_metrics": fit_metrics,
                    "validation": validation,
                    "diagnostics": diagnostics,
                    "rejection_reasons": failures,
                }
            except AlignmentRejected as error:
                matrix = None
                held_cosines = []
                report = {
                    "candidate": name,
                    "status": "rejected",
                    "rejection_reasons": [error.reason],
                }
            candidate_reports.append(report)
            if report["status"] == "accepted" and (
                chosen is None
                or (report["validation"]["recall_at_1"], report["validation"]["mean_held_cosine"])
                > (chosen["report"]["validation"]["recall_at_1"], chosen["report"]["validation"]["mean_held_cosine"])
            ):
                chosen = {"matrix": matrix, "held_cosines": held_cosines, "report": report}

    if chosen is None:
        if not reasons:
            reasons.append("全部候选方案均未通过留家族验证或塌缩诊断")
        return {
            "status": "rejected",
            "source_language": source_language,
            "target_language": target_language,
            "source_index_version": index_versions.get(source_language),
            "target_index_version": index_versions.get(target_language),
            "rejection_reasons": reasons,
            "candidates": candidate_reports,
            "anchor_review": _annotate_review(pair_review, vectors),
            "evidence_count": len(pair_evidence),
        }

    matrix = chosen["matrix"]
    # 复核材料：把未确认/争议对应放到选中映射下观察其偏离程度。
    annotated_review = _annotate_review(
        pair_review, vectors, target_ids=target_ids, target_matrix=target_matrix, matrix=matrix
    )
    result = {
        "status": "accepted",
        "source_language": source_language,
        "target_language": target_language,
        "source_index_version": index_versions[source_language],
        "target_index_version": index_versions[target_language],
        "source_dimension": dimensions[source_language],
        "target_dimension": dimensions[target_language],
        "selected_candidate": chosen["report"]["candidate"],
        "candidates": candidate_reports,
        "matrix": np.round(matrix, 9).tolist(),
        "evidence_anchor_ids": [item["anchor_id"] for item in pair_evidence],
        "evidence_count": len(pair_evidence),
        "trust_band": _score_trust_band(chosen["held_cosines"]),
        "validation": chosen["report"]["validation"],
        "diagnostics": chosen["report"]["diagnostics"],
        "anchor_review": annotated_review,
    }
    result["pair_fingerprint"] = fingerprint(
        {key: value for key, value in result.items() if key not in ("pair_fingerprint",)}
    )
    return result


def _annotate_review(review_entries, vectors, target_ids=None, target_matrix=None, matrix=None):
    """为不能当真值的对应补充几何证据，方便编辑判断一词多义或争议。"""
    target_norm = None
    if target_matrix is not None:
        target_norm = _normalized(target_matrix)
    annotated = []
    for entry in review_entries:
        record = dict(entry)
        if matrix is not None and entry["left"] in vectors and entry["right"] in vectors:
            mapped = vectors[entry["left"]] @ matrix
            record["observed_cosine_to_stated_right"] = _cosine(mapped, vectors[entry["right"]])
            # 该源点在目标空间实际最接近的文档，帮助识别“看起来高频但对应错误”。
            if target_norm is not None:
                direction = mapped / np.linalg.norm(mapped)
                scores = target_norm @ direction
                position = int(np.argmax(scores))
                record["nearest_target_document"] = target_ids[position]
                record["nearest_target_cosine"] = round(float(scores[position]), 6)
        annotated.append(record)
    return annotated


def build_alignment_plan(snapshot, pairs=None):
    """构建完整对齐方案：所有可见语种对、连通性与拒配原因都显式记录。"""
    languages = sorted({row["language"] for row in snapshot["tables"]["documents"]})
    if pairs is None:
        pairs = [(left, right) for left in languages for right in languages if left != right]
    pair_results = [build_pair_alignment(snapshot, left, right) for left, right in pairs]
    accepted = {(row["source_language"], row["target_language"]) for row in pair_results if row["status"] == "accepted"}
    reachability = _reachability(languages, accepted)
    plan = {
        "method": METHOD_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "languages": languages,
        "pairs": pair_results,
        "reachability": reachability,
        "parameters": {
            "min_confirmed_anchors": MIN_CONFIRMED_ANCHORS,
            "min_cv_folds": MIN_CV_FOLDS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "min_effective_rank_ratio": MIN_EFFECTIVE_RANK_RATIO,
            "min_target_coverage": MIN_TARGET_COVERAGE,
            "min_hub_entropy_ratio": MIN_HUB_ENTROPY_RATIO,
            "min_recall_at_1": MIN_RECALL_AT_1,
        },
    }
    plan["alignment_fingerprint"] = fingerprint(
        {key: value for key, value in plan.items() if key != "alignment_fingerprint"}
    )
    return plan


def _reachability(languages, accepted_pairs):
    """按有向连通分量给出可达性；无可靠连接的语种明确标注不可跨越。"""
    graph = defaultdict(set)
    for left, right in accepted_pairs:
        graph[left].add(right)
    result = {}
    for language in languages:
        seen = set()
        stack = [language]
        while stack:
            current = stack.pop()
            for neighbor in graph.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        result[language] = {
            "reachable": sorted(seen),
            "unreachable": sorted(other for other in languages if other != language and other not in seen),
        }
    return result
