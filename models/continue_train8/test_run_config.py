import re
import unittest
from pathlib import Path


class ContinueTrain8RunConfigTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]

    def _read(self, relative_path: str) -> str:
        return (self.root / relative_path).read_text(encoding="utf-8")

    def test_continue8_scripts_keep_first_stage_physics_modules_enabled(self):
        for relative_path in (
            "run_alpha16_continue_extra8.sh",
            "models/continue_train8/run_continue_train8.sh",
        ):
            with self.subTest(path=relative_path):
                text = self._read(relative_path)
                self.assertRegex(text, r"PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE=.*1")
                self.assertRegex(text, r"BASE_LR_OVERRIDE=.*0\.001")
                self.assertIn("checkpoints/alpha16_even8_grad_phys_morozov_direct_noise01", text)
                self.assertNotIn("STAGE1_CHECKPOINT_PATH_OVERRIDE=\"${SCRIPT_DIR}/checkpoints/deep_learn", text)
                self.assertIn("alpha16_plus_extra8_continue_grad_phys_morozov_direct_noise01", text)

    def test_train_and_eval_defaults_keep_first_stage_physics_modules_enabled(self):
        for relative_path in (
            "models/continue_train8/train.py",
            "models/continue_train8/test.py",
        ):
            with self.subTest(path=relative_path):
                text = self._read(relative_path)
                self.assertIn('PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE", "1"', text)
                self.assertIn('BASE_LR_OVERRIDE", "0.001"', text)
                self.assertIn("alpha16_plus_extra8_continue_grad_phys_morozov_direct_noise01", text)

    def test_config_uses_output_tag_as_checkpoint_directory_name(self):
        text = self._read("models/config.py")
        self.assertIn('CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")', text)
        self.assertIn('MODEL_DIR = os.path.join(CHECKPOINT_ROOT, _model_dir_name)', text)
        self.assertNotIn('MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "deep_learn")', text)


if __name__ == "__main__":
    unittest.main()
