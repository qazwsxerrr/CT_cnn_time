import json
import math
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from angle_selection import select_extra8_from_full, select_uniform_condition_best


class AngleSelectionTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.full_json = self.root / "data" / "alpha_search_cache" / "alpha_full_resume.json"
        self.selected16_json = self.root / "data" / "alpha_search_cache" / "alpha_selected16.json"

    def test_reselected_16_matches_existing_alpha_selected16(self):
        full_payload = json.loads(self.full_json.read_text(encoding="utf-8"))
        existing_payload = json.loads(self.selected16_json.read_text(encoding="utf-8"))

        selected = select_uniform_condition_best(full_payload["results"], k=16)
        existing = existing_payload["selected"]

        self.assertEqual(len(selected), 16)
        self.assertEqual(len(existing), 16)
        for generated, reference in zip(selected, existing):
            self.assertAlmostEqual(float(generated["alpha"]), float(reference["alpha"]), places=12)
            self.assertAlmostEqual(float(generated["tau_star"]), float(reference["tau_star"]), places=12)
            self.assertAlmostEqual(float(generated["cond"]), float(reference["cond"]), delta=max(1.0, abs(float(reference["cond"]))) * 1e-12)

    def test_extra8_from_unexcluded_top24_excludes_original16(self):
        result = select_extra8_from_full(
            full_json=self.full_json,
            selected16_json=self.selected16_json,
            top24_k=24,
            extra_k=8,
        )

        self.assertEqual(len(result["selected24"]), 24)
        self.assertEqual(len(result["extra8"]), 8)
        self.assertEqual(result["repeat_count"], 12)

        original_keys = {round(float(item["alpha"]), 12) for item in result["original16"]}
        extra_keys = {round(float(item["alpha"]), 12) for item in result["extra8"]}
        self.assertTrue(original_keys.isdisjoint(extra_keys))

        expected_degrees = [
            16.740,
            28.800,
            67.422,
            99.538,
            113.153,
            131.999,
            142.244,
            148.972,
        ]
        actual_degrees = [float(item["alpha"]) * 180.0 / math.pi for item in result["extra8"]]
        for actual, expected in zip(actual_degrees, expected_degrees):
            self.assertAlmostEqual(actual, expected, places=3)


if __name__ == "__main__":
    unittest.main()
