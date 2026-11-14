from copy import deepcopy
from pathlib import Path
import json
import unittest

from polyglot_atlas.data import load_bundle, visible_snapshot
from polyglot_atlas.product import *


class ProductTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.snapshot = visible_snapshot(load_bundle(root / "fixtures/bundle.json"), "2026-01-01")

    def test_overview_is_finite_and_preserves_sources(self):
        original = deepcopy(self.snapshot)
        result = summarize(self.snapshot)
        self.assertTrue(result)
        self.assertIn("interpretation", result)
        json.dumps(result, allow_nan=False)
        self.assertEqual(self.snapshot, original)

    def test_search_remains_in_language_and_collection(self):
        rows = same_language_search(self.snapshot,"zh-0",100)
        documents = {row["id"]:row for row in self.snapshot["tables"]["documents"]}
        self.assertTrue(rows)
        self.assertTrue(all(documents[row["document"]]["language"] == "zh" for row in rows))
        self.assertTrue(all(documents[row["document"]]["collection"] != "collection-3" for row in rows))

    def test_vector_dimension_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            cosine([1,2],[1])
        self.assertIsNone(cosine([0,0],[1,2]))
