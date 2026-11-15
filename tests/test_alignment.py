import json
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from polyglot_atlas.alignment import (
    METHOD_VERSION,
    MIN_CONFIRMED_ANCHORS,
    build_alignment_plan,
    build_pair_alignment,
)
from polyglot_atlas.crosslingual import CrossLingualRejected, cross_language_search
from polyglot_atlas.data import load_bundle, visible_snapshot
from polyglot_atlas.registry import AlignmentRegistry, RegistryConflict


ROOT = Path(__file__).resolve().parents[1]


class AlignmentFixtureTests(unittest.TestCase):
    def setUp(self):
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        self.snapshot = visible_snapshot(bundle, "2026-01-01")
        self.plan = build_alignment_plan(self.snapshot)
        self.registry = AlignmentRegistry(":memory:")
        self.registry.save(self.plan)
        self.active = self.registry.active()
        self.documents = {row["id"]: row for row in self.snapshot["tables"]["documents"]}

    def test_method_version_and_finite_payload(self):
        self.assertEqual(self.plan["method"], METHOD_VERSION)
        json.dumps(self.plan, allow_nan=False)
        self.assertEqual(len(self.plan["alignment_fingerprint"]), 64)

    def test_supported_pairs_accepted_and_fr_explicitly_unreachable(self):
        statuses = {(p["source_language"], p["target_language"]): p["status"] for p in self.plan["pairs"]}
        self.assertEqual(statuses[("zh", "en")], "accepted")
        self.assertEqual(statuses[("en", "ar")], "accepted")
        # fr 与任何语种都没有可靠连接，必须显式拒配而不是静默比较。
        for (left, right), status in statuses.items():
            if "fr" in (left, right):
                self.assertEqual(status, "rejected")
        reach = self.plan["reachability"]
        self.assertEqual(reach["zh"]["reachable"], ["ar", "en"])
        self.assertEqual(reach["zh"]["unreachable"], ["fr"])
        self.assertEqual(reach["fr"]["unreachable"], ["ar", "en", "zh"])

    def test_rejected_pair_carries_reason(self):
        rejected = next(p for p in self.plan["pairs"] if (p["source_language"], p["target_language"]) == ("zh", "fr"))
        self.assertTrue(rejected["rejection_reasons"])
        self.assertIn("少于最小证据量", rejected["rejection_reasons"][0])

    def test_unconfirmed_anchors_never_fit_but_appear_in_review(self):
        pair = next(p for p in self.plan["pairs"] if (p["source_language"], p["target_language"]) == ("zh", "en"))
        fitted = set(pair["evidence_anchor_ids"])
        self.assertNotIn("anchor-0", fitted)
        self.assertNotIn("anchor-7", fitted)
        self.assertNotIn("anchor-14", fitted)
        review = {row["anchor_id"]: row for row in pair["anchor_review"]}
        self.assertEqual(set(review), {"anchor-0", "anchor-7", "anchor-14"})
        for row in review.values():
            self.assertTrue(any("尚未确认" in reason for reason in row["reasons"]))
            self.assertIsNotNone(row["observed_cosine_to_stated_right"])
            # 复核清单还要指出几何上最接近的真实目标，供编辑辨析一词多义。
            self.assertIn("nearest_target_document", row)
        # anchor-7 跨了文章家族，陈述目标与几何最近目标明显不一致。
        self.assertNotEqual(review["anchor-7"]["nearest_target_document"], "en-8")
        self.assertGreater(review["anchor-7"]["nearest_target_cosine"], 0.99)
        # 争议对应在已确认映射下的几何相似度应明显低于已确认对应。
        self.assertLess(review["anchor-7"]["observed_cosine_to_stated_right"], 0.7)
        self.assertLess(review["anchor-14"]["observed_cosine_to_stated_right"], 0.7)

    def test_validation_holds_out_whole_families(self):
        pair = next(p for p in self.plan["pairs"] if (p["source_language"], p["target_language"]) == ("zh", "en"))
        validation = pair["validation"]
        self.assertGreaterEqual(validation["fold_count"], 3)
        # 每折整体留一整个文章家族，训练锚点家族与留验家族不相交。
        anchor_families = {
            self.documents[left]["family"]
            for anchor in self.snapshot["tables"]["anchors"]
            if anchor["confirmed"]
            for left in (anchor["left"],)
            if self.documents[left]["language"] == "zh"
        }
        held_families = {family for fold in validation["folds"] for family in fold["held_families"]}
        self.assertTrue(held_families.issubset(anchor_families))
        self.assertEqual(validation["recall_at_1"], 1.0)
        band = validation["cosine_band_90"]
        self.assertLess(band["low"], band["high"])
        ci = validation["mean_cosine_ci_90"]
        self.assertLess(ci["low"], ci["high"])

    def test_healthy_plan_passes_collapse_diagnostics(self):
        pair = next(p for p in self.plan["pairs"] if (p["source_language"], p["target_language"]) == ("zh", "en"))
        diagnostics = pair["diagnostics"]
        self.assertGreater(diagnostics["effective_rank_ratio"], 0.75)
        self.assertEqual(diagnostics["target_coverage"], 1.0)
        self.assertGreater(diagnostics["hub_entropy_ratio"], 0.75)

    def test_direct_and_two_hop_search_return_family_members(self):
        direct = cross_language_search(self.snapshot, self.active, "zh-2", "en")
        self.assertEqual(direct["alignment_path"], ["zh", "en"])
        self.assertEqual(direct["results"][0]["document"], "en-2")
        self.assertEqual(direct["results"][0]["alignment_version"], self.active["alignment_version"])
        self.assertTrue(direct["results"][0]["credible"])

        hop2 = cross_language_search(self.snapshot, self.active, "zh-2", "ar")
        self.assertEqual(hop2["alignment_path"], ["zh", "en", "ar"])
        self.assertEqual(len(hop2["alignment_hops"]), 2)
        self.assertEqual(hop2["results"][0]["document"], "ar-2")
        self.assertEqual(hop2["trust_band"]["hops"], 2)

    def test_search_results_carry_alignment_version_and_trust_band(self):
        result = cross_language_search(self.snapshot, self.active, "zh-10", "en", limit=3)
        for row in result["results"]:
            self.assertEqual(row["alignment_id"], self.active["alignment_id"])
            self.assertIn("credible_threshold", row["trust_band"])
            self.assertEqual(row["language"], "en")

    def test_unreachable_language_is_refused_with_reason(self):
        # 临时放开 fr 权限，证明拒配来自对齐缺失而不是权限。
        snapshot = deepcopy(self.snapshot)
        snapshot["tables"]["permissions"][0]["allowed_languages"].append("fr")
        with self.assertRaises(CrossLingualRejected) as error:
            cross_language_search(snapshot, self.active, "zh-2", "fr")
        self.assertEqual(error.exception.code, "unreachable-language")
        self.assertIn("没有可靠对齐连接", error.exception.reason)

    def test_permissions_and_index_version_conventions_still_apply(self):
        # collection-3 不在权限内，查询资料本身被拒。
        with self.assertRaises(CrossLingualRejected) as error:
            cross_language_search(self.snapshot, self.active, "zh-3", "en")
        self.assertEqual(error.exception.code, "forbidden")
        # 目标语种权限。
        with self.assertRaises(CrossLingualRejected) as error:
            cross_language_search(self.snapshot, self.active, "zh-2", "fr")
        self.assertEqual(error.exception.code, "forbidden-language")
        # 结果里不会出现 collection-3 的目标文档。
        result = cross_language_search(self.snapshot, self.active, "zh-0", "en", limit=100)
        self.assertTrue(all(self.documents[r["document"]]["collection"] != "collection-3" for r in result["results"]))

    def test_index_version_mismatch_blocks_comparison(self):
        snapshot = deepcopy(self.snapshot)
        # 目标语料切换到不同索引版本后，对齐声明的版本不再覆盖它们。
        for row in snapshot["tables"]["vectors"]:
            if row["language"] == "en":
                row["index_version"] = "index-2"
        result = cross_language_search(snapshot, self.active, "zh-2", "en")
        self.assertEqual(result["results"], [])

    def test_blocked_items_are_excluded(self):
        result = cross_language_search(
            self.snapshot, self.active, "zh-8", "ar", limit=60, blocked_items=["ar-4"]
        )
        self.assertNotIn("ar-4", [row["document"] for row in result["results"]])
        with self.assertRaises(CrossLingualRejected) as error:
            cross_language_search(
                self.snapshot, self.active, "zh-2", "en", blocked_items=["zh-2"]
            )
        self.assertEqual(error.exception.code, "forbidden")

    def test_query_never_mixes_alignment_versions(self):        # 构建第二个（等价）版本并切换，查询必须整体跟随新版本。
        receipt = self.registry.save(deepcopy(self.plan))
        self.assertEqual(receipt["alignment_version"], 1)  # 同内容幂等
        altered = deepcopy(self.plan)
        altered["parameters"]["min_recall_at_1"] = 0.5
        receipt2 = self.registry.save(altered)
        self.assertEqual(receipt2["alignment_version"], 2)
        result = cross_language_search(self.snapshot, self.registry.active(), "zh-2", "en")
        self.assertEqual(result["alignment_version"], 2)
        for row in result["results"]:
            self.assertEqual(row["alignment_version"], 2)


class CollapseDetectionTests(unittest.TestCase):
    """合成场景：只在高频词上看似对齐、其余内容塌缩时必须被识别。"""

    def setUp(self):
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        self.snapshot = visible_snapshot(bundle, "2026-01-01")
        self.vectors = {row["id"]: np.asarray(row["values"], dtype=float) for row in self.snapshot["tables"]["vectors"]}
        self.docs = {row["id"]: row for row in self.snapshot["tables"]["documents"]}

    def _rank_one_matrix(self):
        # 两个“高频”锚点指向同一热门目标，拟合出秩不足的退化映射。
        source = np.stack([self.vectors["zh-0"], self.vectors["zh-1"]])
        target = np.stack([self.vectors["en-1"], self.vectors["en-1"]])
        matrix, *_ = np.linalg.lstsq(source, target, rcond=None)
        return matrix

    def test_rank_collapsed_map_fails_diagnostics(self):
        from polyglot_atlas.alignment import _collapse_diagnostics

        target_ids = sorted(i for i, d in self.docs.items() if d["language"] == "en")
        target_matrix = np.stack([self.vectors[i] for i in target_ids])
        source_ids = sorted(i for i, d in self.docs.items() if d["language"] == "zh")
        diagnostics = _collapse_diagnostics(
            self._rank_one_matrix(), source_ids, self.vectors, target_matrix
        )
        self.assertLess(diagnostics["effective_rank_ratio"], 0.75)
        self.assertLess(diagnostics["target_coverage"], 0.6)
        self.assertLess(diagnostics["hub_entropy_ratio"], 0.75)

    def test_collapse_is_visible_at_search_time(self):
        # 退化映射即便被写入方案，可信阈值也应让绝大多数候选落入不可信。
        from polyglot_atlas.registry import AlignmentRegistry

        plan = build_alignment_plan(self.snapshot)
        pair = next(p for p in plan["pairs"] if (p["source_language"], p["target_language"]) == ("zh", "en"))
        pair["matrix"] = self._rank_one_matrix().tolist()
        pair["pair_fingerprint"] = "degenerate"
        registry = AlignmentRegistry(":memory:")
        registry.save(plan)
        result = cross_language_search(self.snapshot, registry.active(), "zh-20", "en", limit=20)
        self.assertTrue(all(not row["credible"] for row in result["results"]))

    def test_sparse_anchor_pair_is_rejected(self):
        # 删除大部分确认锚点后，证据量低于门槛必须拒配。
        snapshot = deepcopy(self.snapshot)
        keep = {"anchor-1", "anchor-2", "anchor-3"}
        snapshot["tables"]["anchors"] = [
            row
            for row in snapshot["tables"]["anchors"]
            if not (row["confirmed"] and row["left"].startswith("zh")) or row["id"] in keep
        ]
        remaining = [
            row for row in snapshot["tables"]["anchors"]
            if row["confirmed"] and row["left"].startswith("zh")
        ]
        self.assertLess(len(remaining), MIN_CONFIRMED_ANCHORS)
        pair = build_pair_alignment(snapshot, "zh", "en")
        self.assertEqual(pair["status"], "rejected")
        self.assertTrue(any("少于最小证据量" in reason for reason in pair["rejection_reasons"]))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        self.snapshot = visible_snapshot(bundle, "2026-01-01")
        self.registry = AlignmentRegistry(":memory:")

    def test_atomic_activation_and_optimistic_switch(self):
        plan = build_alignment_plan(self.snapshot)
        first = self.registry.save(plan, activate=False)
        self.assertIsNone(self.registry.active_version()[0])
        self.registry.activate(first["alignment_version"])
        self.assertEqual(self.registry.active_version()[0], first["alignment_version"])
        altered = deepcopy(plan)
        altered["parameters"] = dict(plan["parameters"])
        altered["parameters"]["min_recall_at_1"] = 0.5
        second = self.registry.save(altered, activate=False)
        with self.assertRaises(RegistryConflict):
            self.registry.activate(second["alignment_version"], expected_active_version=99)
        self.registry.activate(second["alignment_version"], expected_active_version=first["alignment_version"])
        self.assertEqual(self.registry.active_version()[0], second["alignment_version"])
        # 旧版本仍不可变可读。
        self.assertEqual(self.registry.get(first["alignment_version"])["plan"], plan)

    def test_idempotent_save_replays_same_version(self):
        plan = build_alignment_plan(self.snapshot)
        first = self.registry.save(plan)
        second = self.registry.save(plan)
        self.assertEqual(first["alignment_version"], second["alignment_version"])
        self.assertTrue(second["replayed"])


if __name__ == "__main__":
    unittest.main()
