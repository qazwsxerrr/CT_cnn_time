from __future__ import annotations

import tempfile
import unittest
import os
import io
from pathlib import Path
from contextlib import redirect_stdout
from argparse import Namespace

import torch

from offline_tvinit_data import (
    OfflineBatchProvider,
    _log,
    _resolve_generation_counts,
    OfflineCTDataset,
    apply_alpha8_tvinit_env_defaults,
    default_alpha_json_path,
    generate_mixed_offline_dataset,
    load_offline_tensors,
    save_offline_tensors,
)


class OfflineTVInitDataCompatibilityTest(unittest.TestCase):
    def test_loader_returns_only_network_training_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pt"
            torch.save(
                {
                    "coeff_true": torch.ones(2, 128, 128),
                    "g_observed": torch.arange(12, dtype=torch.float32).view(2, 6),
                    "coeff_initial": torch.zeros(2, 128, 128),
                    "unused_summary": {"should": "be ignored"},
                },
                path,
            )

            tensors = load_offline_tensors(path)
            self.assertEqual(set(tensors.keys()), {"coeff_true", "g_observed", "coeff_initial"})
            self.assertEqual(tuple(tensors["coeff_true"].shape), (2, 1, 128, 128))
            self.assertEqual(tuple(tensors["g_observed"].shape), (2, 6))
            self.assertEqual(tuple(tensors["coeff_initial"].shape), (2, 1, 128, 128))

            dataset = OfflineCTDataset(path)
            coeff_true, g_observed, coeff_initial = dataset[1]
            self.assertEqual(tuple(coeff_true.shape), (1, 128, 128))
            self.assertEqual(tuple(g_observed.shape), (6,))
            self.assertEqual(tuple(coeff_initial.shape), (1, 128, 128))
            self.assertEqual(len(dataset), 2)

    def test_save_writes_minimal_tensor_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.pt"
            save_offline_tensors(
                path,
                coeff_true=torch.ones(1, 1, 128, 128),
                g_observed=torch.zeros(1, 8),
                coeff_initial=torch.full((1, 1, 128, 128), 0.5),
            )

            raw = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(raw.keys()), {"coeff_true", "g_observed", "coeff_initial"})
            self.assertEqual(raw["coeff_true"].dtype, torch.float32)
            self.assertEqual(raw["g_observed"].dtype, torch.float32)
            self.assertEqual(raw["coeff_initial"].dtype, torch.float32)

    def test_default_env_matches_alpha8_tvinit_profile(self):
        alpha_json = default_alpha_json_path()
        self.assertTrue(alpha_json.exists(), str(alpha_json))

        old_values = {}
        for key in (
            "ALPHA_CONDITION_JSON_OVERRIDE",
            "INIT_METHOD_OVERRIDE",
            "DATA_FIDELITY_CHANNEL_MODE_OVERRIDE",
            "PHYSICS_RESIDUAL_MODE_OVERRIDE",
            "NOISE_LEVEL_OVERRIDE",
        ):
            old_values[key] = os.environ.pop(key, None)
        try:
            env = apply_alpha8_tvinit_env_defaults()
            self.assertEqual(env["ALPHA_CONDITION_JSON_OVERRIDE"], str(alpha_json))
            self.assertEqual(env["INIT_METHOD_OVERRIDE"], "l2_tv_admm")
            self.assertEqual(env["DATA_FIDELITY_CHANNEL_MODE_OVERRIDE"], "stacked_selected")
            self.assertEqual(env["PHYSICS_RESIDUAL_MODE_OVERRIDE"], "stacked_selected_cg")
            self.assertEqual(env["NOISE_LEVEL_OVERRIDE"], "0.1")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_batch_provider_sequentially_wraps_across_dataset_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequence.pt"
            values = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
            save_offline_tensors(
                path,
                coeff_true=values,
                g_observed=torch.arange(10, dtype=torch.float32).view(5, 2),
                coeff_initial=values + 10.0,
            )

            provider = OfflineBatchProvider(path, shuffle=False)
            batch1 = provider.generate_batch(3)
            batch2 = provider.generate_batch(3)

            self.assertEqual(batch1[0].view(-1).tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(batch2[0].view(-1).tolist(), [3.0, 4.0, 0.0])
            self.assertEqual(batch2[3].view(-1).tolist(), [13.0, 14.0, 10.0])

    def test_batch_provider_selects_explicit_indices_without_advancing_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indexed.pt"
            values = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
            save_offline_tensors(
                path,
                coeff_true=values,
                g_observed=torch.arange(12, dtype=torch.float32).view(6, 2),
                coeff_initial=values + 10.0,
            )

            provider = OfflineBatchProvider(path, shuffle=False)
            indexed = provider.generate_batch_by_indices([4, 1, 4])
            next_sequential = provider.generate_batch(2)

            self.assertEqual(indexed[0].view(-1).tolist(), [4.0, 1.0, 4.0])
            self.assertEqual(indexed[3].view(-1).tolist(), [14.0, 11.0, 14.0])
            self.assertEqual(next_sequential[0].view(-1).tolist(), [0.0, 1.0])

    def test_batch_provider_random_batch_is_seeded_and_does_not_advance_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "random.pt"
            values = torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1)
            save_offline_tensors(
                path,
                coeff_true=values,
                g_observed=torch.arange(20, dtype=torch.float32).view(10, 2),
                coeff_initial=values + 10.0,
            )

            provider = OfflineBatchProvider(path, shuffle=False)
            random_a = provider.generate_random_batch(4, random_seed=123)
            random_b = provider.generate_random_batch(4, random_seed=123)
            random_c = provider.generate_random_batch(4, random_seed=124)
            next_sequential = provider.generate_batch(2)

            self.assertEqual(random_a[0].view(-1).tolist(), random_b[0].view(-1).tolist())
            self.assertNotEqual(random_a[0].view(-1).tolist(), random_c[0].view(-1).tolist())
            self.assertEqual(len(set(random_a[0].view(-1).tolist())), 4)
            self.assertEqual(next_sequential[0].view(-1).tolist(), [0.0, 1.0])

    def test_mixed_generation_keeps_random_then_shepp_order_and_minimal_payload(self):
        class FakeGenerator:
            def __init__(self, data_source):
                self.data_source = data_source

            def generate_batch(self, batch_size, random_seed=None):
                del random_seed
                base = 1.0 if self.data_source == "random_ellipses" else 9.0
                coeff_true = torch.full((batch_size, 1, 2, 2), base)
                f_true = coeff_true.clone()
                g_observed = torch.full((batch_size, 3), base + 1.0)
                coeff_initial = torch.full((batch_size, 1, 2, 2), base + 2.0)
                return coeff_true, f_true, g_observed, coeff_initial

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.pt"
            generate_mixed_offline_dataset(
                path,
                random_ellipses_samples=2,
                shepp_logan_samples=1,
                batch_size=2,
                generator_factory=FakeGenerator,
            )

            raw = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(raw.keys()), {"coeff_true", "g_observed", "coeff_initial"})
            self.assertEqual(raw["coeff_true"][:, 0, 0, 0].tolist(), [1.0, 1.0, 9.0])
            self.assertEqual(raw["g_observed"][:, 0].tolist(), [2.0, 2.0, 10.0])
            self.assertEqual(raw["coeff_initial"][:, 0, 0, 0].tolist(), [3.0, 3.0, 11.0])

    def test_log_prefix_contains_timestamp(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _log("hello")
        self.assertRegex(buffer.getvalue(), r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello\n$")

    def test_mixed_cli_counts_default_only_when_no_count_is_provided(self):
        self.assertEqual(
            _resolve_generation_counts(
                Namespace(num_samples=None, random_ellipses_samples=None, shepp_logan_samples=None)
            ),
            (3000, 500),
        )
        self.assertEqual(
            _resolve_generation_counts(
                Namespace(num_samples=None, random_ellipses_samples=None, shepp_logan_samples=500)
            ),
            (0, 500),
        )
        self.assertEqual(
            _resolve_generation_counts(
                Namespace(num_samples=None, random_ellipses_samples=8000, shepp_logan_samples=None)
            ),
            (8000, 0),
        )


if __name__ == "__main__":
    unittest.main()
