from pathlib import Path
import tempfile
import unittest

from polyglot_atlas.client import ProductClient, RemoteError
from polyglot_atlas.data import load_bundle
from polyglot_atlas.registry import AlignmentRegistry
from polyglot_atlas.service import running_service
from polyglot_atlas.store import ArtifactStore
from polyglot_atlas.workflow import cross_lingual_material_search, publish_cross_lingual_alignment


ROOT = Path(__file__).resolve().parents[1]


class AlignmentHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        cls.store = ArtifactStore(str(Path(cls.directory.name) / "state.sqlite3"), bundle.get("policy"))
        cls.registry_path = str(Path(cls.directory.name) / "alignments.sqlite3")
        cls.registry = AlignmentRegistry(cls.registry_path)
        cls.running = running_service(bundle, cls.store, cls.registry)
        cls.client = ProductClient(cls.running.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls.running.__exit__(None, None, None)
        cls.store.close()
        cls.registry.close()
        cls.directory.cleanup()

    def test_no_active_alignment_refuses_search(self):
        if self.client.alignment_versions()["versions"]:
            self.skipTest("对齐已由其他用例激活")
        with self.assertRaises(RemoteError) as error:
            self.client.cross_lingual_search("zh-2", "en", "2026-01-01")
        self.assertEqual(error.exception.status, 409)
        self.assertEqual(error.exception.body["code"], "no-active-alignment")

    def test_build_and_persist_alignment(self):
        receipt = publish_cross_lingual_alignment(self.client, "2026-01-01")
        self.assertEqual(receipt["alignment_version"], 1)
        active = self.client.active_alignment()
        self.assertEqual(active["alignment_version"], 1)
        self.assertEqual(active["plan"]["method"], "cross-lingual-alignment-v1")
        # 摘要里拒配语种对带有原因。
        pair = next(p for p in receipt["plan_summary"]["pairs"] if p["pair"] == "zh->fr")
        self.assertEqual(pair["status"], "rejected")
        self.assertTrue(pair["rejection_reasons"])

    def test_cross_lingual_search_roundtrip(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        result = cross_lingual_material_search(self.client, "zh-2", "en", "2026-01-01")
        self.assertEqual(result["results"][0]["document"], "en-2")
        self.assertEqual(result["alignment_id"], self.client.active_alignment()["alignment_id"])
        self.assertEqual(result["alignment_path"], ["zh", "en"])
        two_hop = cross_lingual_material_search(self.client, "zh-2", "ar", "2026-01-01")
        self.assertEqual(two_hop["alignment_path"], ["zh", "en", "ar"])
        self.assertEqual(two_hop["results"][0]["document"], "ar-2")

    def test_unreachable_language_returns_typed_error(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        # fr 在权限与对齐两层都不通；先验证 HTTP 状态与中文错误。
        with self.assertRaises(RemoteError) as error:
            self.client.cross_lingual_search("zh-2", "fr", "2026-01-01")
        self.assertIn(error.exception.status, (409, 422))
        self.assertIn("error", error.exception.body)

    def test_versions_endpoint_and_immutable_history(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        listing = self.client.alignment_versions()
        self.assertEqual([row["alignment_version"] for row in listing["versions"]], [1])
        first = self.client.alignment(1)
        self.assertEqual(first["alignment_id"], listing["versions"][0]["alignment_id"])
        with self.assertRaises(RemoteError) as error:
            self.client.alignment(99)
        self.assertEqual(error.exception.status, 404)

    def test_atomic_switch_keeps_query_on_single_version(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        # 同内容为幂等重放，不产生新版本，查询整体落在同一版本。
        receipt = self.client.build_alignment("2026-01-01", activate=False)
        self.assertEqual(receipt["alignment_version"], 1)
        before = self.client.cross_lingual_search("zh-4", "en", "2026-01-01")
        self.assertEqual(before["alignment_version"], 1)
        # 比较交换激活到不存在的版本应冲突。
        with self.assertRaises(RemoteError) as error:
            self.client.activate_alignment(7, expected_active_version=1)
        self.assertEqual(error.exception.status, 409)

    def test_registry_survives_reopen(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        reopened = AlignmentRegistry(self.registry_path)
        try:
            version, identifier = reopened.active_version()
            self.assertEqual(version, 1)
            self.assertTrue(identifier.startswith("alignment-v1"))
        finally:
            reopened.close()

    def test_search_respects_permissions(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        with self.assertRaises(RemoteError) as error:
            self.client.cross_lingual_search("zh-3", "en", "2026-01-01")
        self.assertEqual(error.exception.status, 422)
        self.assertEqual(error.exception.body["code"], "forbidden")

    def test_search_rejects_snapshot_that_alignment_was_not_built_on(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        # 早期快照的指纹与生效方案不同，服务必须拒绝跨空间查询。
        with self.assertRaises(RemoteError) as error:
            self.client.cross_lingual_search("zh-0", "en", "2024-01-01")
        self.assertEqual(error.exception.status, 409)
        self.assertEqual(error.exception.body["code"], "snapshot-mismatch")

    def test_every_search_result_tracks_single_alignment_version(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        result = self.client.cross_lingual_search("zh-10", "ar", "2026-01-01", limit=10)
        active_version = self.client.active_alignment()["alignment_version"]
        self.assertEqual(result["alignment_version"], active_version)
        for row in result["results"]:
            self.assertEqual(row["alignment_version"], active_version)
            self.assertEqual(row["alignment_path"], ["zh", "en", "ar"])

    def test_policy_blocked_items_are_removed_from_results(self):
        publish_cross_lingual_alignment(self.client, "2026-01-01")
        # 初始消费约束禁用 ar-4，任何结果都不应包含它。
        result = self.client.cross_lingual_search("zh-8", "ar", "2026-01-01", limit=60)
        self.assertNotIn("ar-4", [row["document"] for row in result["results"]])


if __name__ == "__main__":
    unittest.main()
