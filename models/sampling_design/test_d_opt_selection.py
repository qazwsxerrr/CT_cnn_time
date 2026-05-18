# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.sampling_design.d_opt_selection import d_opt_greedy_select
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


if __name__ == "__main__":
    unittest.main()
