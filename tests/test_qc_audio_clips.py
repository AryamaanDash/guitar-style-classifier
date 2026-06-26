from __future__ import annotations

import csv
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qc_audio_clips import QCConfig, analyze_wav_file, run_qc


def write_wav(path: Path, samples: list[int], sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


class AudioClipQCTests(unittest.TestCase):
    def test_analyze_wav_file_records_core_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "tone.wav"
            samples = [
                int(10000 * math.sin(2 * math.pi * 440 * index / 8000))
                for index in range(8000)
            ]
            write_wav(wav_path, samples)

            metrics = analyze_wav_file(wav_path, QCConfig())

            self.assertTrue(metrics["qc_readable"])
            self.assertFalse(metrics["qc_invalid_audio_flag"])
            self.assertAlmostEqual(metrics["qc_duration_seconds"], 1.0)
            self.assertEqual(metrics["qc_sample_rate_hz"], 8000)
            self.assertEqual(metrics["qc_channels"], 1)
            self.assertEqual(metrics["qc_sample_width_bytes"], 2)
            self.assertGreater(metrics["qc_peak_amplitude"], 0.30)
            self.assertGreater(metrics["qc_rms_amplitude"], 0.20)
            self.assertEqual(metrics["qc_clipped_sample_percent"], 0.0)

    def test_run_qc_flags_duplicates_silence_and_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_dir = root / "data" / "processed"
            metadata_csv = root / "data" / "metadata" / "clips.csv"
            output_dir = root / "results" / "qc"

            label_dir = processed_dir / "session_test" / "test_label"
            normal_samples = [1000] * 8000
            duplicate_samples = normal_samples.copy()
            silent_samples = [0] * 8000
            clipped_samples = [32767] * 80 + [0] * 7920

            write_wav(label_dir / "normal.wav", normal_samples)
            write_wav(label_dir / "duplicate.wav", duplicate_samples)
            write_wav(label_dir / "silent.wav", silent_samples)
            write_wav(label_dir / "clipped.wav", clipped_samples)

            metadata_csv.parent.mkdir(parents=True, exist_ok=True)
            with metadata_csv.open("w", newline="", encoding="utf-8") as metadata_file:
                writer = csv.DictWriter(
                    metadata_file,
                    fieldnames=[
                        "clip_filename",
                        "source_filename",
                        "label",
                        "session",
                        "clip_index",
                        "sample_rate_hz",
                        "channels",
                    ],
                )
                writer.writeheader()
                for index, clip_filename in enumerate(
                    ["normal.wav", "duplicate.wav", "silent.wav", "clipped.wav"]
                ):
                    writer.writerow(
                        {
                            "clip_filename": clip_filename,
                            "source_filename": "source.wav",
                            "label": "test_label",
                            "session": "session_test",
                            "clip_index": index,
                            "sample_rate_hz": "8000",
                            "channels": "1",
                        }
                    )

            result = run_qc(
                QCConfig(
                    metadata_csv=metadata_csv,
                    processed_dir=processed_dir,
                    output_dir=output_dir,
                    expected_sample_rate_hz=8000,
                    expected_channels=1,
                    min_label_size_for_outlier=99,
                    skip_spectrograms=True,
                )
            )

            with result.qc_csv_path.open(newline="", encoding="utf-8") as qc_file:
                rows = {
                    row["clip_filename"]: row
                    for row in csv.DictReader(qc_file)
                }

            self.assertEqual(result.row_count, 4)
            self.assertEqual(rows["normal.wav"]["label"], "test_label")
            self.assertEqual(rows["normal.wav"]["qc_duplicate_flag"], "true")
            self.assertEqual(rows["duplicate.wav"]["qc_duplicate_flag"], "true")
            self.assertEqual(rows["silent.wav"]["qc_nearly_silent_flag"], "true")
            self.assertEqual(rows["clipped.wav"]["qc_clipping_flag"], "true")
            self.assertTrue(result.summary_report_path.exists())
            self.assertTrue(result.summary_by_label_path.exists())


if __name__ == "__main__":
    unittest.main()
