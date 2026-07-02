import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asr_noise_robust.results_export import (
    ResultSpec,
    collect_metric_rows,
    condition_sort_key,
    export_result_artifacts,
)


def _write_metrics(path: Path, model: str, values: dict[str, tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "feature",
                "model",
                "train_condition",
                "test_condition",
                "wer",
                "cer",
                "substitutions",
                "deletions",
                "insertions",
                "notes",
            ],
        )
        writer.writeheader()
        for condition, (wer, cer) in values.items():
            writer.writerow(
                {
                    "experiment": "unit",
                    "feature": "feature",
                    "model": model,
                    "train_condition": "clean",
                    "test_condition": condition,
                    "wer": f"{wer:.6f}",
                    "cer": f"{cer:.6f}",
                    "substitutions": "2",
                    "deletions": "1",
                    "insertions": "0",
                    "notes": "unit",
                }
            )


class ResultsExportTests(unittest.TestCase):
    def test_condition_sort_key_orders_clean_then_decreasing_snr(self):
        ordered = sorted(["snr_5", "clean", "snr_20", "snr_10"], key=condition_sort_key)

        self.assertEqual(ordered, ["clean", "snr_20", "snr_10", "snr_5"])

    def test_collect_metric_rows_adds_system_labels_and_numeric_fields(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metrics_path = root / "metrics.csv"
            _write_metrics(metrics_path, "model-a", {"clean": (0.1, 0.05), "snr_5": (0.8, 0.4)})

            rows = collect_metric_rows(
                [ResultSpec("sys_a", "System A", "group", metrics_path)]
            )

            self.assertEqual([row["system_id"] for row in rows], ["sys_a", "sys_a"])
            self.assertEqual(rows[0]["system_label"], "System A")
            self.assertEqual(rows[0]["condition_label"], "clean")
            self.assertEqual(rows[1]["condition_label"], "5 dB")
            self.assertAlmostEqual(rows[0]["wer"], 0.1)

    def test_export_result_artifacts_writes_tables_and_figures(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metrics_a = root / "a.csv"
            metrics_b = root / "b.csv"
            values_a = {
                "clean": (0.1, 0.04),
                "snr_20": (0.2, 0.08),
                "snr_10": (0.4, 0.2),
                "snr_5": (0.8, 0.5),
            }
            values_b = {
                "clean": (0.2, 0.1),
                "snr_20": (0.3, 0.12),
                "snr_10": (0.5, 0.25),
                "snr_5": (0.9, 0.55),
            }
            _write_metrics(metrics_a, "model-a", values_a)
            _write_metrics(metrics_b, "model-b", values_b)

            outputs = export_result_artifacts(
                specs=[
                    ResultSpec("sys_a", "System A", "baseline", metrics_a),
                    ResultSpec("sys_b", "System B", "ssl", metrics_b),
                ],
                tables_dir=root / "tables",
                figures_dir=root / "figures",
            )

            self.assertTrue(outputs.long_metrics_path.exists())
            self.assertTrue(outputs.summary_table_path.exists())
            self.assertEqual(len(outputs.figure_paths), 4)
            for figure_path in outputs.figure_paths:
                self.assertTrue(figure_path.exists())
                self.assertGreater(figure_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
