# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.sampling_design.bucketed_d_opt_selection import (
    bucket_candidates_by_angle,
    bucketed_d_opt_beam_select,
)
from models.sampling_design.d_opt_selection import d_opt_greedy_select
from models.sampling_design.gap_constrained_d_opt_selection import (
    circular_gap_stats,
    gap_constrained_d_opt_select,
)
from models.sampling_design.reduced_operator import (
    load_candidate_records,
    make_random_sketch_basis,
    reduced_information_for_record,
)


class SamplingDesignTests(unittest.TestCase):
    def test_load_candidate_records_prefers_raw_results_over_selected(self) -> None:
        payload = {
            "selected": [
                {
                    "alpha": 0.1,
                    "tau_star": 0.2,
                    "is_valid": True,
                    "cond": 2.0,
                    "log_cond": math.log(2.0),
                }
            ],
            "results": [
                {"alpha": 0.0, "tau_star": None, "is_valid": False, "cond": "inf"},
                {
                    "alpha": 0.3,
                    "tau_star": 0.4,
                    "is_valid": True,
                    "cond": 3.0,
                    "log_cond": math.log(3.0),
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            records = load_candidate_records(path)

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(float(records[0]["alpha"]), 0.3)
        self.assertAlmostEqual(float(records[0]["tau_star"]), 0.4)

    def test_make_random_sketch_basis_is_orthonormal_and_reproducible(self) -> None:
        basis_a = make_random_sketch_basis(n=12, rank=4, seed=7)
        basis_b = make_random_sketch_basis(n=12, rank=4, seed=7)

        self.assertEqual(basis_a.shape, (12, 4))
        np.testing.assert_allclose(basis_a.T @ basis_a, np.eye(4), atol=1.0e-12)
        np.testing.assert_allclose(basis_a, basis_b, atol=0.0)

    def test_reduced_information_for_record_returns_symmetric_psd_matrix(self) -> None:
        z_basis = make_random_sketch_basis(n=9, rank=3, seed=1)
        record = {"alpha": 0.37, "tau_star": 0.25, "is_valid": True, "cond": 4.0}

        enriched = reduced_information_for_record(
            record,
            z_basis=z_basis,
            image_size=3,
            injective_tol=1.0e-12,
            value_tol=1.0e-15,
        )

        g = np.asarray(enriched["reduced_info"], dtype=np.float64)
        self.assertEqual(g.shape, (3, 3))
        np.testing.assert_allclose(g, g.T, atol=1.0e-12)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(g))), -1.0e-10)
        self.assertGreater(float(enriched["reduced_info_trace"]), 0.0)

    def test_d_opt_greedy_select_uses_logdet_gain(self) -> None:
        candidates = [
            {"name": "x", "alpha": 0.1, "tau_star": 0.0, "reduced_info": np.diag([3.0, 0.0])},
            {"name": "y", "alpha": 1.1, "tau_star": 0.0, "reduced_info": np.diag([0.0, 2.0])},
            {"name": "weak-y", "alpha": 1.2, "tau_star": 0.0, "reduced_info": np.diag([0.0, 0.5])},
        ]

        selected, trace = d_opt_greedy_select(
            candidates,
            k=2,
            sketch_rank=2,
            lambda_info=1.0,
            gamma_uniform=0.0,
        )

        self.assertEqual([item["name"] for item in selected], ["x", "y"])
        self.assertEqual([item["step"] for item in trace], [1, 2])
        self.assertGreater(trace[0]["gain"], trace[1]["gain"])

    def test_bucket_candidates_by_angle_keeps_best_records_per_bucket(self) -> None:
        candidates = [
            {"name": "b0_worse", "alpha": 0.10, "tau_star": 0.0, "log_cond": 5.0},
            {"name": "b0_best", "alpha": 0.20, "tau_star": 0.0, "log_cond": 1.0},
            {"name": "b1_best", "alpha": 2.00, "tau_star": 0.0, "log_cond": 2.0},
            {"name": "b1_worse", "alpha": 2.20, "tau_star": 0.0, "log_cond": 4.0},
        ]

        buckets = bucket_candidates_by_angle(candidates, bucket_count=2, per_bucket_keep=1)

        self.assertEqual([[item["name"] for item in bucket] for bucket in buckets], [["b0_best"], ["b1_best"]])

    def test_bucketed_d_opt_beam_select_selects_one_candidate_per_bucket(self) -> None:
        candidates = [
            {"name": "a_strong_x", "alpha": 0.10, "tau_star": 0.0, "log_cond": 5.0, "reduced_info": np.diag([4.0, 0.0])},
            {"name": "a_weak", "alpha": 0.20, "tau_star": 0.0, "log_cond": 1.0, "reduced_info": np.diag([1.0, 0.0])},
            {"name": "b_strong_y", "alpha": 1.80, "tau_star": 0.0, "log_cond": 4.0, "reduced_info": np.diag([0.0, 3.0])},
            {"name": "b_weak", "alpha": 2.20, "tau_star": 0.0, "log_cond": 2.0, "reduced_info": np.diag([0.0, 0.5])},
        ]

        selected, trace = bucketed_d_opt_beam_select(
            candidates,
            k=2,
            sketch_rank=2,
            lambda_info=1.0,
            per_bucket_keep=2,
            beam_size=4,
            uniformity_epsilon=None,
        )

        self.assertEqual([item["name"] for item in selected], ["a_strong_x", "b_strong_y"])
        self.assertEqual([item["bucket"] for item in selected], [0, 1])
        self.assertEqual(len(trace), 2)
        self.assertGreater(trace[-1]["best_logdet"], trace[0]["best_logdet"])

    def test_circular_gap_stats_reports_min_and_max_gaps(self) -> None:
        stats = circular_gap_stats([0.0, math.pi / 3.0])

        self.assertAlmostEqual(stats["min_gap_deg"], 60.0)
        self.assertAlmostEqual(stats["max_gap_deg"], 120.0)

    def test_gap_constrained_d_opt_select_enforces_final_gap_bounds(self) -> None:
        candidates = [
            {"name": "a0", "alpha": 0.0, "tau_star": 0.0, "reduced_info": np.diag([4.0, 0.0, 0.0])},
            {"name": "a_close", "alpha": 0.40, "tau_star": 0.0, "reduced_info": np.diag([9.0, 0.0, 0.0])},
            {"name": "b", "alpha": math.pi / 3.0, "tau_star": 0.0, "reduced_info": np.diag([0.0, 4.0, 0.0])},
            {"name": "c", "alpha": 2.0 * math.pi / 3.0, "tau_star": 0.0, "reduced_info": np.diag([0.0, 0.0, 4.0])},
        ]

        selected, trace = gap_constrained_d_opt_select(
            candidates,
            k=3,
            sketch_rank=3,
            lambda_info=1.0,
            min_gap_deg=50.0,
            max_gap_deg=130.0,
            beam_size=16,
        )

        self.assertEqual([item["name"] for item in selected], ["a0", "b", "c"])
        stats = circular_gap_stats([float(item["alpha"]) for item in selected])
        self.assertGreaterEqual(stats["min_gap_deg"], 50.0)
        self.assertLessEqual(stats["max_gap_deg"], 130.0)
        self.assertGreaterEqual(trace[-1]["feasible_final_count"], 1)


if __name__ == "__main__":
    unittest.main()
