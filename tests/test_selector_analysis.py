import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("selector_analysis", Path(__file__).parents[1] / "scripts/analyze_selector_pilot.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class SelectorAnalysisTests(unittest.TestCase):
    def rows(self):
        rows = []
        for split, multiplier in (("seen", 1), ("unseen", 2)):
            for seed in (1, 2):
                for image_id, difference in (("one", .1), ("two", .3)):
                    for task in ("pair1", "pair2"):
                        for method in ("conditioned", "image_only"):
                            rows.append(dict(task_split=split, seed=str(seed), image_id=image_id,
                                task_id=task, method=method, correct="1", margin=str(
                                    .2 + (difference*multiplier if method=="conditioned" else 0))))
        return rows

    def test_seed_repeats_do_not_multiply_independent_images(self):
        result = analysis.analyze(self.rows())
        effect = result["task_splits"]["unseen"]["conditional_minus_image_only"]["margin"]
        self.assertEqual(effect["across_seed_image_effect"]["image_count"], 2)
        self.assertAlmostEqual(effect["across_seed_image_effect"]["mean"], .4)
        self.assertEqual(len(effect["per_seed"]), 2)
        self.assertAlmostEqual(result["unseen_minus_seen_conditional_effect"]["margin"]["mean"], .2)

    def test_unmatched_method_rows_cannot_produce_paired_claim(self):
        with self.assertRaises(AssertionError):
            analysis.analyze(self.rows()[:-1])


if __name__ == "__main__":
    unittest.main()
