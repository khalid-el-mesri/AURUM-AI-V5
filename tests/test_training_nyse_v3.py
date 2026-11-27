"""AURUM-AI NYSE V3 trainer unit tests; entirely synthetic, no real XOM/Test access."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deep_learning.training_v3 import train_nyse_v3 as t


def synthetic_npz(path, horizon, *, train=True, n=8):
    first = 1_700_000_000_000 if train else 1_700_100_000_000
    ends = first + np.arange(n, dtype=np.int64) * 300_000
    targets = ends + horizon * 60_000
    prices = np.full(n, 100.0, dtype=np.float64)
    future = prices * np.exp(np.linspace(-0.003, 0.003, n))
    X = np.ones((n, 20, 6), dtype=np.float32) * 0.25
    X[:, :, 0] = np.linspace(-0.2, 0.2, n)[:, None]
    np.savez_compressed(
        path, X=X, y=np.log(future / prices).astype(np.float32),
        feature_names=np.asarray(t.EXPECTED_FEATURES),
        end_timestamp_ms=ends, target_timestamp_ms=targets,
        signal_available_at_ms=ends + 300_000,
        target_available_at_ms=targets + 300_000,
        current_close=prices, next_close=future,
    )


def make_dataset(root, horizon=5):
    root.mkdir(parents=True, exist_ok=True)
    (root / "normalization_stats.json").write_text(json.dumps({
        "fitted_on": "train 5m windows only",
        "feature_names": list(t.EXPECTED_FEATURES), "window_size": 20,
        "mean": [0] * 6, "std": [1] * 6,
    }), encoding="utf-8")
    synthetic_npz(root / t.filename_for("train", horizon), horizon, train=True)
    synthetic_npz(root / t.filename_for("validation", horizon), horizon, train=False)


class TrainingV3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        self.outputs = Path(self.tmp.name) / "outputs"
        make_dataset(self.data)

    def test_input_names(self):
        self.assertEqual(t.filename_for("train", 5), "train_normalized.npz")
        self.assertEqual(t.filename_for("validation", 60), "validation_60m_normalized.npz")

    def test_forbids_test_split(self):
        with self.assertRaises(ValueError):
            t.filename_for("test", 5)

    def test_model_architectures_and_shapes(self):
        x = torch.zeros((4, 20, 6))
        for name in ("lstm", "gru", "cnn1d"):
            with self.subTest(name=name):
                self.assertEqual(tuple(t.make_model(name)(x).shape), (4,))

    def test_cnn_layers_identical_to_historical_design(self):
        m = t.make_model("cnn1d")
        self.assertIsInstance(m.network[0], torch.nn.Conv1d)
        self.assertEqual(m.network[0].kernel_size, (3,))
        self.assertEqual(m.network[0].padding, (1,))
        self.assertIsInstance(m.network[2], torch.nn.Conv1d)
        self.assertIsInstance(m.network[4], torch.nn.AdaptiveAvgPool1d)

    def test_lstm_and_gru_hidden_sizes(self):
        self.assertEqual(t.make_model("lstm").lstm.hidden_size, 32)
        self.assertEqual(t.make_model("gru").gru.hidden_size, 32)

    def test_load_valid(self):
        train, val = t.load_pair(self.data, 5)
        self.assertEqual(train["X"].shape, (8, 20, 6))
        self.assertEqual(val["X"].shape, (8, 20, 6))

    def test_dry_run_has_no_filesystem_side_effect(self):
        result = t.train_experiment(model_name="cnn1d", horizon=5, data_dir=self.data,
                                    experiments_dir=self.outputs, dry_run=True)
        self.assertEqual(result["train_samples"], 8)
        self.assertFalse(self.outputs.exists())

    def test_missing_key_rejected(self):
        p = self.data / "train_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files if name != "target_available_at_ms"}
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "missing keys"):
            t.load_pair(self.data, 5)

    def test_wrong_horizon_rejected(self):
        p = self.data / "train_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files}
        d["target_timestamp_ms"] = d["target_timestamp_ms"] + 300_000
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "wrong target horizon"):
            t.load_pair(self.data, 5)

    def test_invalid_signal_availability_rejected(self):
        p = self.data / "train_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files}
        d["signal_available_at_ms"] = d["signal_available_at_ms"] - 300_000
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "availability timestamp"):
            t.load_pair(self.data, 5)

    def test_invalid_target_rejected(self):
        p = self.data / "train_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files}
        d["y"] = d["y"] + 0.01
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "log return"):
            t.load_pair(self.data, 5)

    def test_nonfinite_rejected(self):
        p = self.data / "train_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files}
        d["X"][0, 0, 0] = np.nan
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            t.load_pair(self.data, 5)

    def test_feature_order_rejected(self):
        p = self.data / "train_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files}
        d["feature_names"] = d["feature_names"][::-1]
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "feature order"):
            t.load_pair(self.data, 5)

    def test_bad_scaler_rejected(self):
        stats = json.loads((self.data / "normalization_stats.json").read_text())
        stats["fitted_on"] = "train + validation"
        (self.data / "normalization_stats.json").write_text(json.dumps(stats))
        with self.assertRaisesRegex(ValueError, "Train-only"):
            t.load_pair(self.data, 5)

    def test_split_overlap_rejected(self):
        p = self.data / "validation_normalized.npz"
        with np.load(p, allow_pickle=False) as src:
            d = {name: src[name] for name in src.files}
        original = d["end_timestamp_ms"][0]
        replacement = 1_700_000_000_000 + 2 * 300_000
        delta = replacement - original
        for key in ("end_timestamp_ms", "target_timestamp_ms", "signal_available_at_ms", "target_available_at_ms"):
            d[key] = d[key] + delta
        np.savez_compressed(p, **d)
        with self.assertRaisesRegex(ValueError, "overlap"):
            t.load_pair(self.data, 5)

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            t.make_model("unsupported")

    def test_bad_run_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "run ID"):
            t.train_experiment(model_name="lstm", horizon=5, data_dir=self.data,
                               experiments_dir=self.outputs, run_id="../danger", dry_run=True)

    def test_existing_experiment_is_protected(self):
        (self.outputs / "lstm_5m_seed42").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            t.train_experiment(model_name="lstm", horizon=5, data_dir=self.data,
                               experiments_dir=self.outputs, dry_run=True)

    def test_metrics_zero_baseline(self):
        m = t.metrics(np.array([1.0, -1.0]), np.array([0., 0.]))
        self.assertEqual(m["MAE"], 1)
        self.assertEqual(m["RMSE"], 1)

    def test_15m_and_60m_paths_supported(self):
        for horizon in (15, 60):
            make_dataset(self.data, horizon)
            self.assertEqual(len(t.load_pair(self.data, horizon)[0]["y"]), 8)

    def test_tiny_train_lstm_checkpoint_and_report(self):
        info = t.train_experiment(model_name="lstm", horizon=5, data_dir=self.data,
                                  experiments_dir=self.outputs, epochs=1)
        run_dir = self.outputs / "lstm_5m_seed42"
        self.assertEqual(info["status"], "COMPLETED")
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["model_name"], "lstm")
        self.assertEqual(ckpt["forecast_horizon_minutes"], 5)
        model = t.make_model("lstm")
        model.load_state_dict(ckpt["model_state_dict"])
        self.assertTrue((run_dir / "report.json").exists())

    def test_tiny_train_gru_checkpoint_and_report(self):
        info = t.train_experiment(model_name="gru", horizon=5, data_dir=self.data,
                                  experiments_dir=self.outputs, epochs=1)
        run_dir = self.outputs / "gru_5m_seed42"
        self.assertEqual(info["status"], "COMPLETED")
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["model_name"], "gru")
        model = t.make_model("gru")
        model.load_state_dict(ckpt["model_state_dict"])
        self.assertTrue((run_dir / "report.json").exists())

    def test_tiny_train_cnn1d_creates_protected_run(self):
        info = t.train_experiment(model_name="cnn1d", horizon=5, data_dir=self.data,
                                  experiments_dir=self.outputs, epochs=1)
        run_dir = self.outputs / "cnn1d_5m_seed42"
        self.assertEqual(info["status"], "COMPLETED")
        self.assertTrue((run_dir / "best.pt").is_file())
        self.assertTrue((run_dir / "report.json").is_file())
        self.assertTrue((run_dir / "validation_predictions.npz").is_file())
        self.assertFalse((self.outputs / "test_normalized.npz").exists())
        with np.load(run_dir / "validation_predictions.npz", allow_pickle=False) as p:
            self.assertEqual(len(p["y_pred"]), 8)
            self.assertIn("signal_available_at_ms", p.files)
        with self.assertRaises(FileExistsError):
            t.train_experiment(model_name="cnn1d", horizon=5, data_dir=self.data,
                               experiments_dir=self.outputs, epochs=1)


if __name__ == "__main__":
    unittest.main()
