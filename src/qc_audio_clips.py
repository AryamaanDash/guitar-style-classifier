from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import statistics
import sys
import warnings as warning_tools
import wave
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "clips.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS_BASE_DIR = PROJECT_ROOT / "results" / "qc"

QC_FIELDNAMES = [
    "qc_metadata_found",
    "qc_file_found",
    "qc_readable",
    "qc_invalid_audio_flag",
    "qc_error",
    "qc_relative_path",
    "qc_file_path",
    "qc_label_for_group",
    "qc_duration_seconds",
    "qc_expected_duration_seconds",
    "qc_sample_rate_hz",
    "qc_expected_sample_rate_hz",
    "qc_channels",
    "qc_expected_channels",
    "qc_sample_width_bytes",
    "qc_frame_count",
    "qc_sample_count",
    "qc_peak_amplitude",
    "qc_rms_amplitude",
    "qc_rms_dbfs",
    "qc_clipped_sample_percent",
    "qc_audio_sha256",
    "qc_duplicate_flag",
    "qc_duplicate_group_id",
    "qc_duplicate_count",
    "qc_duplicate_of",
    "qc_duration_flag",
    "qc_unexpected_sample_rate_flag",
    "qc_unexpected_channels_flag",
    "qc_metadata_mismatch_flag",
    "qc_nearly_silent_flag",
    "qc_clipping_flag",
    "qc_amplitude_outlier_flag",
    "qc_rms_dbfs_label_median",
    "qc_rms_dbfs_modified_z",
    "qc_any_flag",
    "qc_flag_reasons",
    "qc_spectrogram_path",
]

REASON_BY_FIELD = {
    "qc_invalid_audio_flag": "invalid_audio",
    "qc_duration_flag": "duration_off",
    "qc_unexpected_sample_rate_flag": "unexpected_sample_rate",
    "qc_unexpected_channels_flag": "unexpected_channels",
    "qc_metadata_mismatch_flag": "metadata_mismatch",
    "qc_nearly_silent_flag": "nearly_silent",
    "qc_clipping_flag": "clipping",
    "qc_duplicate_flag": "duplicate_audio",
    "qc_amplitude_outlier_flag": "amplitude_outlier",
}


@dataclass
class QCConfig:
    metadata_csv: Path = DEFAULT_METADATA_CSV
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    output_dir: Path = DEFAULT_RESULTS_BASE_DIR
    expected_duration_seconds: float = 1.0
    duration_tolerance_seconds: float = 0.05
    expected_sample_rate_hz: int | None = None
    expected_channels: int | None = None
    near_silence_rms: float = 0.001
    near_silence_peak: float = 0.005
    near_clipping_threshold: float = 0.98
    clipping_percent_threshold: float = 0.01
    amplitude_outlier_modified_z: float = 3.5
    amplitude_outlier_min_db: float = 12.0
    min_label_size_for_outlier: int = 5
    skip_spectrograms: bool = False


@dataclass
class QCResult:
    output_dir: Path
    qc_csv_path: Path
    flagged_csv_path: Path
    summary_report_path: Path
    summary_by_label_path: Path
    spectrogram_dir: Path
    row_count: int
    flagged_count: int
    spectrogram_count: int
    warnings: list[str]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def optional_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.{digits}f}"
    return str(value)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def read_metadata(metadata_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not metadata_csv.exists():
        return [], []

    with metadata_csv.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def discover_wav_files(processed_dir: Path) -> list[Path]:
    if not processed_dir.exists():
        return []

    return sorted(
        path
        for path in processed_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def relative_to_processed(path: Path, processed_dir: Path) -> str:
    try:
        return path.relative_to(processed_dir).as_posix()
    except ValueError:
        return path.as_posix()


def metadata_relative_path(row: dict[str, str]) -> Path | None:
    for path_column in (
        "clip_path",
        "file_path",
        "filepath",
        "path",
        "relative_path",
    ):
        value = row.get(path_column)
        if value:
            return Path(value)

    clip_filename = row.get("clip_filename")
    if not clip_filename:
        return None

    parts = []
    if row.get("session"):
        parts.append(row["session"])
    if row.get("label"):
        parts.append(row["label"])
    parts.append(clip_filename)
    return Path(*parts)


def make_extra_metadata_row(
    path: Path,
    processed_dir: Path,
    metadata_fieldnames: list[str],
) -> dict[str, str]:
    row = {fieldname: "" for fieldname in metadata_fieldnames}
    if "clip_filename" in row:
        row["clip_filename"] = path.name

    try:
        relative_parts = path.relative_to(processed_dir).parts
    except ValueError:
        relative_parts = path.parts

    if "session" in row and len(relative_parts) >= 3:
        row["session"] = relative_parts[-3]
    if "label" in row and len(relative_parts) >= 2:
        row["label"] = relative_parts[-2]
    return row


def label_for_group(
    row: dict[str, str],
    path: Path | None,
    processed_dir: Path,
) -> str:
    if row.get("label"):
        return row["label"]
    if path is not None:
        try:
            relative_parts = path.relative_to(processed_dir).parts
        except ValueError:
            relative_parts = path.parts
        if len(relative_parts) >= 2:
            return relative_parts[-2]
    return "unknown"


def decode_pcm_samples(frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(sample - 128) / 128.0 for sample in frames]

    if sample_width == 2:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        return [sample / 32768.0 for sample in samples]

    if sample_width == 3:
        decoded = []
        for index in range(0, len(frames), 3):
            sample = int.from_bytes(frames[index : index + 3], "little")
            if sample >= 2**23:
                sample -= 2**24
            decoded.append(sample / 8388608.0)
        return decoded

    if sample_width == 4:
        samples = array("i")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        return [sample / 2147483648.0 for sample in samples]

    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def analyze_wav_file(path: Path, config: QCConfig) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "qc_file_found": path.exists(),
        "qc_readable": False,
        "qc_invalid_audio_flag": False,
        "qc_error": "",
        "qc_duration_seconds": None,
        "qc_sample_rate_hz": None,
        "qc_channels": None,
        "qc_sample_width_bytes": None,
        "qc_frame_count": None,
        "qc_sample_count": None,
        "qc_peak_amplitude": None,
        "qc_rms_amplitude": None,
        "qc_rms_dbfs": None,
        "qc_clipped_sample_percent": None,
        "qc_audio_sha256": "",
    }

    if not path.exists():
        metrics["qc_invalid_audio_flag"] = True
        metrics["qc_error"] = "File referenced by metadata was not found"
        return metrics

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (EOFError, wave.Error, OSError) as error:
        metrics["qc_invalid_audio_flag"] = True
        metrics["qc_error"] = str(error)
        return metrics

    metrics["qc_readable"] = True
    metrics["qc_channels"] = channels
    metrics["qc_sample_width_bytes"] = sample_width
    metrics["qc_sample_rate_hz"] = sample_rate
    metrics["qc_frame_count"] = frame_count
    metrics["qc_duration_seconds"] = frame_count / sample_rate if sample_rate else None

    try:
        samples = decode_pcm_samples(frames, sample_width)
    except ValueError as error:
        metrics["qc_invalid_audio_flag"] = True
        metrics["qc_error"] = str(error)
        return metrics

    sample_count = len(samples)
    metrics["qc_sample_count"] = sample_count
    if sample_rate <= 0 or channels <= 0 or frame_count <= 0 or sample_count <= 0:
        metrics["qc_invalid_audio_flag"] = True
        metrics["qc_error"] = "Audio contains no decodable samples"
        return metrics

    peak = max(abs(sample) for sample in samples)
    mean_square = sum(sample * sample for sample in samples) / sample_count
    rms = math.sqrt(mean_square)
    clipped_samples = sum(
        1 for sample in samples if abs(sample) >= config.near_clipping_threshold
    )

    hash_input = (
        f"{channels}|{sample_width}|{sample_rate}|".encode("utf-8") + frames
    )
    metrics["qc_peak_amplitude"] = peak
    metrics["qc_rms_amplitude"] = rms
    metrics["qc_rms_dbfs"] = 20.0 * math.log10(rms) if rms > 0 else float("-inf")
    metrics["qc_clipped_sample_percent"] = 100.0 * clipped_samples / sample_count
    metrics["qc_audio_sha256"] = hashlib.sha256(hash_input).hexdigest()
    return metrics


def infer_expected_values(rows: list[dict[str, Any]], config: QCConfig) -> tuple[int | None, int | None]:
    expected_sample_rate = config.expected_sample_rate_hz
    expected_channels = config.expected_channels

    readable_rows = [
        row
        for row in rows
        if row.get("qc_readable") and not row.get("qc_invalid_audio_flag")
    ]
    if expected_sample_rate is None:
        sample_rates = [
            row["qc_sample_rate_hz"]
            for row in readable_rows
            if row.get("qc_sample_rate_hz") is not None
        ]
        expected_sample_rate = Counter(sample_rates).most_common(1)[0][0] if sample_rates else None

    if expected_channels is None:
        channels = [
            row["qc_channels"]
            for row in readable_rows
            if row.get("qc_channels") is not None
        ]
        expected_channels = Counter(channels).most_common(1)[0][0] if channels else None

    return expected_sample_rate, expected_channels


def add_duplicate_flags(rows: list[dict[str, Any]]) -> None:
    hashes_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        audio_hash = row.get("qc_audio_sha256")
        if audio_hash:
            hashes_to_rows[audio_hash].append(row)

    duplicate_group_index = 0
    for duplicate_rows in hashes_to_rows.values():
        if len(duplicate_rows) < 2:
            for row in duplicate_rows:
                row["qc_duplicate_flag"] = False
                row["qc_duplicate_group_id"] = ""
                row["qc_duplicate_count"] = 1
                row["qc_duplicate_of"] = ""
            continue

        duplicate_group_index += 1
        group_id = f"dup_{duplicate_group_index:04d}"
        canonical_path = duplicate_rows[0].get("qc_relative_path", "")
        for row in duplicate_rows:
            row["qc_duplicate_flag"] = True
            row["qc_duplicate_group_id"] = group_id
            row["qc_duplicate_count"] = len(duplicate_rows)
            row["qc_duplicate_of"] = canonical_path


def add_amplitude_outlier_flags(rows: list[dict[str, Any]], config: QCConfig) -> None:
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rms_dbfs = row.get("qc_rms_dbfs")
        if (
            row.get("qc_readable")
            and not row.get("qc_invalid_audio_flag")
            and isinstance(rms_dbfs, float)
            and math.isfinite(rms_dbfs)
        ):
            rows_by_label[row["qc_label_for_group"]].append(row)

    for row in rows:
        row["qc_amplitude_outlier_flag"] = False
        row["qc_rms_dbfs_label_median"] = None
        row["qc_rms_dbfs_modified_z"] = None

    for label_rows in rows_by_label.values():
        if len(label_rows) < config.min_label_size_for_outlier:
            continue

        values = [row["qc_rms_dbfs"] for row in label_rows]
        label_median = statistics.median(values)
        absolute_deviations = [abs(value - label_median) for value in values]
        mad = statistics.median(absolute_deviations)

        for row in label_rows:
            difference_db = row["qc_rms_dbfs"] - label_median
            row["qc_rms_dbfs_label_median"] = label_median

            if mad > 0:
                modified_z = 0.6745 * difference_db / mad
                row["qc_rms_dbfs_modified_z"] = modified_z
                row["qc_amplitude_outlier_flag"] = (
                    abs(modified_z) >= config.amplitude_outlier_modified_z
                    and abs(difference_db) >= config.amplitude_outlier_min_db
                )
            else:
                row["qc_rms_dbfs_modified_z"] = 0.0
                row["qc_amplitude_outlier_flag"] = (
                    abs(difference_db) >= config.amplitude_outlier_min_db
                )


def add_threshold_flags(
    rows: list[dict[str, Any]],
    config: QCConfig,
    expected_sample_rate: int | None,
    expected_channels: int | None,
) -> None:
    for row in rows:
        row["qc_expected_duration_seconds"] = config.expected_duration_seconds
        row["qc_expected_sample_rate_hz"] = expected_sample_rate
        row["qc_expected_channels"] = expected_channels
        row["qc_duration_flag"] = False
        row["qc_unexpected_sample_rate_flag"] = False
        row["qc_unexpected_channels_flag"] = False
        row["qc_metadata_mismatch_flag"] = False
        row["qc_nearly_silent_flag"] = False
        row["qc_clipping_flag"] = False

        if row.get("qc_invalid_audio_flag"):
            continue

        duration = row.get("qc_duration_seconds")
        if duration is not None:
            row["qc_duration_flag"] = (
                abs(duration - config.expected_duration_seconds)
                > config.duration_tolerance_seconds
            )

        if expected_sample_rate is not None and row.get("qc_sample_rate_hz") is not None:
            row["qc_unexpected_sample_rate_flag"] = (
                row["qc_sample_rate_hz"] != expected_sample_rate
            )

        if expected_channels is not None and row.get("qc_channels") is not None:
            row["qc_unexpected_channels_flag"] = (
                row["qc_channels"] != expected_channels
            )

        metadata_sample_rate = parse_int(row.get("sample_rate_hz"))
        metadata_channels = parse_int(row.get("channels"))
        row["qc_metadata_mismatch_flag"] = (
            (
                metadata_sample_rate is not None
                and row.get("qc_sample_rate_hz") is not None
                and metadata_sample_rate != row["qc_sample_rate_hz"]
            )
            or (
                metadata_channels is not None
                and row.get("qc_channels") is not None
                and metadata_channels != row["qc_channels"]
            )
        )

        peak = row.get("qc_peak_amplitude")
        rms = row.get("qc_rms_amplitude")
        if peak is not None and rms is not None:
            row["qc_nearly_silent_flag"] = (
                rms <= config.near_silence_rms and peak <= config.near_silence_peak
            )

        clipped_sample_percent = row.get("qc_clipped_sample_percent")
        if clipped_sample_percent is not None:
            row["qc_clipping_flag"] = (
                clipped_sample_percent >= config.clipping_percent_threshold
            )


def finalize_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        reasons = [
            reason
            for fieldname, reason in REASON_BY_FIELD.items()
            if row.get(fieldname)
        ]
        if not row.get("qc_file_found"):
            reasons.insert(0, "file_missing")

        row["qc_flag_reasons"] = ";".join(reasons)
        row["qc_any_flag"] = bool(reasons)


def create_spectrogram(path: Path, output_path: Path, cache_dir: Path) -> None:
    import numpy as np

    matplotlib_cache = cache_dir / "matplotlib"
    xdg_cache = cache_dir / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    samples = np.array(decode_pcm_samples(frames, sample_width), dtype=np.float32)
    if channels > 1:
        usable_sample_count = (samples.size // channels) * channels
        samples = samples[:usable_sample_count].reshape(-1, channels).mean(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4), dpi=120)
    with np.errstate(divide="ignore"), warning_tools.catch_warnings():
        warning_tools.filterwarnings(
            "ignore",
            message="divide by zero encountered in log10",
            category=RuntimeWarning,
        )
        axis.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=768, cmap="magma")
    axis.set_title(path.name)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def add_spectrograms(
    rows: list[dict[str, Any]],
    config: QCConfig,
    spectrogram_dir: Path,
) -> tuple[int, list[str]]:
    warnings = []
    spectrogram_count = 0

    for row in rows:
        row["qc_spectrogram_path"] = ""

    if config.skip_spectrograms:
        return spectrogram_count, warnings

    for row in rows:
        if (
            not row.get("qc_any_flag")
            or not row.get("qc_readable")
            or row.get("qc_invalid_audio_flag")
        ):
            continue

        path_value = row.get("_absolute_path")
        if not path_value:
            continue

        label = sanitize_path_part(row["qc_label_for_group"])
        stem = Path(path_value).stem
        output_path = spectrogram_dir / label / f"{stem}.png"
        try:
            create_spectrogram(
                Path(path_value),
                output_path,
                config.output_dir.parent / ".cache",
            )
        except ImportError as error:
            warnings.append(
                "Spectrogram generation skipped because numpy/matplotlib "
                f"is unavailable: {error}"
            )
            break
        except (EOFError, wave.Error, OSError, ValueError) as error:
            warnings.append(f"Could not create spectrogram for {path_value}: {error}")
            continue

        row["qc_spectrogram_path"] = output_path.as_posix()
        spectrogram_count += 1

    return spectrogram_count, warnings


def sanitize_path_part(value: str) -> str:
    safe_characters = []
    for character in value:
        if character.isalnum() or character in ("-", "_"):
            safe_characters.append(character)
        else:
            safe_characters.append("_")
    return "".join(safe_characters) or "unknown"


def build_rows(
    metadata_fieldnames: list[str],
    metadata_rows: list[dict[str, str]],
    discovered_wavs: list[Path],
    config: QCConfig,
) -> list[dict[str, Any]]:
    discovered_by_relative = {
        relative_to_processed(path, config.processed_dir): path for path in discovered_wavs
    }
    discovered_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in discovered_wavs:
        discovered_by_name[path.name].append(path)

    rows: list[dict[str, Any]] = []
    matched_relative_paths: set[str] = set()

    for metadata_row in metadata_rows:
        row: dict[str, Any] = {fieldname: metadata_row.get(fieldname, "") for fieldname in metadata_fieldnames}
        row["qc_metadata_found"] = True

        relative_path = metadata_relative_path(metadata_row)
        path = None
        if relative_path is not None:
            candidate_path = config.processed_dir / relative_path
            if candidate_path.exists():
                path = candidate_path
            elif relative_path.as_posix() in discovered_by_relative:
                path = discovered_by_relative[relative_path.as_posix()]

        if path is None and metadata_row.get("clip_filename"):
            matching_names = discovered_by_name.get(metadata_row["clip_filename"], [])
            if len(matching_names) == 1:
                path = matching_names[0]

        if path is not None:
            relative_path_text = relative_to_processed(path, config.processed_dir)
            matched_relative_paths.add(relative_path_text)
            row["_absolute_path"] = path.as_posix()
            row["qc_relative_path"] = relative_path_text
            row["qc_file_path"] = path.as_posix()
        else:
            expected_relative_path = relative_path.as_posix() if relative_path else ""
            row["_absolute_path"] = ""
            row["qc_relative_path"] = expected_relative_path
            row["qc_file_path"] = (
                (config.processed_dir / relative_path).as_posix()
                if relative_path is not None
                else ""
            )

        path_for_label = Path(row["_absolute_path"]) if row["_absolute_path"] else None
        row["qc_label_for_group"] = label_for_group(
            metadata_row,
            path_for_label,
            config.processed_dir,
        )
        rows.append(row)

    for wav_path in discovered_wavs:
        relative_path_text = relative_to_processed(wav_path, config.processed_dir)
        if relative_path_text in matched_relative_paths:
            continue

        row = make_extra_metadata_row(wav_path, config.processed_dir, metadata_fieldnames)
        row["qc_metadata_found"] = False
        row["_absolute_path"] = wav_path.as_posix()
        row["qc_relative_path"] = relative_path_text
        row["qc_file_path"] = wav_path.as_posix()
        row["qc_label_for_group"] = label_for_group(row, wav_path, config.processed_dir)
        rows.append(row)

    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: csv_value(row.get(fieldname)) for fieldname in fieldnames})


def csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, float):
        return optional_number(value)
    if value is None:
        return ""
    return str(value)


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({row["qc_label_for_group"] for row in rows})
    summary_rows = []
    for label in labels:
        label_rows = [row for row in rows if row["qc_label_for_group"] == label]
        summary_row = {
            "label": label,
            "total_rows": len(label_rows),
            "metadata_rows": sum(1 for row in label_rows if row.get("qc_metadata_found")),
            "discovered_files": sum(1 for row in label_rows if row.get("qc_file_found")),
            "readable_files": sum(1 for row in label_rows if row.get("qc_readable")),
            "flagged_clips": sum(1 for row in label_rows if row.get("qc_any_flag")),
            "invalid_audio": sum(1 for row in label_rows if row.get("qc_invalid_audio_flag")),
            "duration_flags": sum(1 for row in label_rows if row.get("qc_duration_flag")),
            "sample_rate_flags": sum(1 for row in label_rows if row.get("qc_unexpected_sample_rate_flag")),
            "channel_flags": sum(1 for row in label_rows if row.get("qc_unexpected_channels_flag")),
            "metadata_mismatch_flags": sum(1 for row in label_rows if row.get("qc_metadata_mismatch_flag")),
            "nearly_silent_flags": sum(1 for row in label_rows if row.get("qc_nearly_silent_flag")),
            "clipping_flags": sum(1 for row in label_rows if row.get("qc_clipping_flag")),
            "duplicate_flags": sum(1 for row in label_rows if row.get("qc_duplicate_flag")),
            "amplitude_outlier_flags": sum(1 for row in label_rows if row.get("qc_amplitude_outlier_flag")),
        }
        summary_rows.append(summary_row)

    return summary_rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def write_summary_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    config: QCConfig,
    qc_csv_path: Path,
    flagged_csv_path: Path,
    summary_by_label_path: Path,
    spectrogram_dir: Path,
    warnings: list[str],
) -> None:
    flagged_rows = [row for row in rows if row.get("qc_any_flag")]
    expected_sample_rate = next(
        (
            row.get("qc_expected_sample_rate_hz")
            for row in rows
            if row.get("qc_expected_sample_rate_hz") is not None
        ),
        None,
    )
    expected_channels = next(
        (
            row.get("qc_expected_channels")
            for row in rows
            if row.get("qc_expected_channels") is not None
        ),
        None,
    )
    expected_sample_rate_text = (
        f"{expected_sample_rate} Hz"
        if config.expected_sample_rate_hz is not None
        else f"{expected_sample_rate} Hz (inferred mode)"
        if expected_sample_rate is not None
        else "not available"
    )
    expected_channels_text = (
        str(expected_channels)
        if config.expected_channels is not None
        else f"{expected_channels} (inferred mode)"
        if expected_channels is not None
        else "not available"
    )
    lines = [
        "# Audio Clip QC Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Outputs",
        "",
        f"- QC metadata CSV: `{qc_csv_path.as_posix()}`",
        f"- Flagged clips CSV: `{flagged_csv_path.as_posix()}`",
        f"- Summary by label CSV: `{summary_by_label_path.as_posix()}`",
        f"- Spectrogram directory: `{spectrogram_dir.as_posix()}`",
        "",
        "## Thresholds",
        "",
        f"- Expected duration: {config.expected_duration_seconds:.3f} seconds",
        f"- Duration tolerance: +/- {config.duration_tolerance_seconds:.3f} seconds",
        f"- Expected sample rate: {expected_sample_rate_text}",
        f"- Expected channels: {expected_channels_text}",
        f"- Nearly silent: RMS <= {config.near_silence_rms:g} and peak <= {config.near_silence_peak:g}",
        f"- Near clipping sample: abs(sample) >= {config.near_clipping_threshold:g}",
        f"- Clipping flag: clipped sample percent >= {config.clipping_percent_threshold:g}",
        (
            "- Amplitude outlier: per-label RMS dBFS modified Z >= "
            f"{config.amplitude_outlier_modified_z:g} and at least "
            f"{config.amplitude_outlier_min_db:g} dB from the label median"
        ),
        f"- Minimum label size for outlier detection: {config.min_label_size_for_outlier}",
        "",
        "## Counts By Label",
        "",
    ]

    count_headers = [
        "label",
        "total",
        "readable",
        "flagged",
        "invalid",
        "duration",
        "sr",
        "channels",
        "silent",
        "clipping",
        "duplicates",
        "outliers",
    ]
    count_rows = [
        [
            str(summary_row["label"]),
            str(summary_row["total_rows"]),
            str(summary_row["readable_files"]),
            str(summary_row["flagged_clips"]),
            str(summary_row["invalid_audio"]),
            str(summary_row["duration_flags"]),
            str(summary_row["sample_rate_flags"]),
            str(summary_row["channel_flags"]),
            str(summary_row["nearly_silent_flags"]),
            str(summary_row["clipping_flags"]),
            str(summary_row["duplicate_flags"]),
            str(summary_row["amplitude_outlier_flags"]),
        ]
        for summary_row in summary_rows
    ]
    lines.append(markdown_table(count_headers, count_rows))
    lines.extend(["", "## Flagged Clips", ""])

    if not flagged_rows:
        lines.append("No clips were flagged.")
    else:
        for label in sorted({row["qc_label_for_group"] for row in flagged_rows}):
            label_flagged_rows = [
                row for row in flagged_rows if row["qc_label_for_group"] == label
            ]
            lines.extend([f"### {label}", ""])
            flagged_headers = [
                "clip",
                "reasons",
                "duration",
                "sample_rate",
                "channels",
                "rms_dbfs",
                "clipped_pct",
                "spectrogram",
            ]
            flagged_table_rows = [
                [
                    str(row.get("qc_relative_path", "")),
                    str(row.get("qc_flag_reasons", "")),
                    optional_number(row.get("qc_duration_seconds"), 4),
                    optional_number(row.get("qc_sample_rate_hz"), 0),
                    optional_number(row.get("qc_channels"), 0),
                    optional_number(row.get("qc_rms_dbfs"), 2),
                    optional_number(row.get("qc_clipped_sample_percent"), 4),
                    str(row.get("qc_spectrogram_path", "")),
                ]
                for row in label_flagged_rows
            ]
            lines.append(markdown_table(flagged_headers, flagged_table_rows))
            lines.append("")

    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_qc(config: QCConfig) -> QCResult:
    metadata_fieldnames, metadata_rows = read_metadata(config.metadata_csv)
    discovered_wavs = discover_wav_files(config.processed_dir)
    rows = build_rows(metadata_fieldnames, metadata_rows, discovered_wavs, config)

    for row in rows:
        absolute_path = row.get("_absolute_path")
        metrics = analyze_wav_file(Path(absolute_path), config) if absolute_path else {
            "qc_file_found": False,
            "qc_readable": False,
            "qc_invalid_audio_flag": True,
            "qc_error": "No matching WAV file found",
        }
        row.update(metrics)

    expected_sample_rate, expected_channels = infer_expected_values(rows, config)
    add_duplicate_flags(rows)
    add_threshold_flags(rows, config, expected_sample_rate, expected_channels)
    add_amplitude_outlier_flags(rows, config)
    finalize_flags(rows)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir = config.output_dir / "spectrograms"
    spectrogram_count, warnings = add_spectrograms(rows, config, spectrogram_dir)

    output_fieldnames = metadata_fieldnames + [
        fieldname for fieldname in QC_FIELDNAMES if fieldname not in metadata_fieldnames
    ]
    qc_csv_path = config.output_dir / "clips_qc.csv"
    flagged_csv_path = config.output_dir / "flagged_clips.csv"
    summary_by_label_path = config.output_dir / "summary_by_label.csv"
    summary_report_path = config.output_dir / "summary_report.md"

    write_csv(qc_csv_path, output_fieldnames, rows)
    flagged_rows = [row for row in rows if row.get("qc_any_flag")]
    write_csv(flagged_csv_path, output_fieldnames, flagged_rows)
    summary_rows = build_summary_rows(rows)
    write_csv(summary_by_label_path, list(summary_rows[0].keys()) if summary_rows else ["label"], summary_rows)
    write_summary_report(
        summary_report_path,
        rows,
        summary_rows,
        config,
        qc_csv_path,
        flagged_csv_path,
        summary_by_label_path,
        spectrogram_dir,
        warnings,
    )

    return QCResult(
        output_dir=config.output_dir,
        qc_csv_path=qc_csv_path,
        flagged_csv_path=flagged_csv_path,
        summary_report_path=summary_report_path,
        summary_by_label_path=summary_by_label_path,
        spectrogram_dir=spectrogram_dir,
        row_count=len(rows),
        flagged_count=len(flagged_rows),
        spectrogram_count=spectrogram_count,
        warnings=warnings,
    )


def default_run_output_dir(base_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quality-control checks for one-second processed WAV clips."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="Input clips metadata CSV. Defaults to data/metadata/clips.csv.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory recursively searched for WAV clips. Defaults to data/processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for QC outputs. Defaults to a timestamped directory under "
            "results/qc."
        ),
    )
    parser.add_argument(
        "--expected-duration-seconds",
        type=float,
        default=1.0,
        help="Expected clip duration in seconds.",
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=0.05,
        help="Flag duration if it differs from expected duration by more than this.",
    )
    parser.add_argument(
        "--expected-sample-rate-hz",
        type=int,
        default=None,
        help="Expected sample rate. If omitted, the mode across readable clips is used.",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=None,
        help="Expected channel count. If omitted, the mode across readable clips is used.",
    )
    parser.add_argument(
        "--near-silence-rms",
        type=float,
        default=0.001,
        help="Near-silence RMS threshold on normalized samples.",
    )
    parser.add_argument(
        "--near-silence-peak",
        type=float,
        default=0.005,
        help="Near-silence peak threshold on normalized samples.",
    )
    parser.add_argument(
        "--near-clipping-threshold",
        type=float,
        default=0.98,
        help="Samples with absolute normalized amplitude at or above this are counted.",
    )
    parser.add_argument(
        "--clipping-percent-threshold",
        type=float,
        default=0.01,
        help="Flag clipping when near-clipping samples reach this percentage.",
    )
    parser.add_argument(
        "--amplitude-outlier-modified-z",
        type=float,
        default=3.5,
        help="Per-label robust modified Z threshold for RMS dBFS outliers.",
    )
    parser.add_argument(
        "--amplitude-outlier-min-db",
        type=float,
        default=12.0,
        help="Minimum dB distance from label median before an amplitude outlier is flagged.",
    )
    parser.add_argument(
        "--min-label-size-for-outlier",
        type=int,
        default=5,
        help="Minimum readable clips per label before outlier detection runs.",
    )
    parser.add_argument(
        "--skip-spectrograms",
        action="store_true",
        help="Run all QC checks but do not generate spectrogram PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_run_output_dir(DEFAULT_RESULTS_BASE_DIR)
    config = QCConfig(
        metadata_csv=args.metadata_csv,
        processed_dir=args.processed_dir,
        output_dir=output_dir,
        expected_duration_seconds=args.expected_duration_seconds,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        expected_sample_rate_hz=args.expected_sample_rate_hz,
        expected_channels=args.expected_channels,
        near_silence_rms=args.near_silence_rms,
        near_silence_peak=args.near_silence_peak,
        near_clipping_threshold=args.near_clipping_threshold,
        clipping_percent_threshold=args.clipping_percent_threshold,
        amplitude_outlier_modified_z=args.amplitude_outlier_modified_z,
        amplitude_outlier_min_db=args.amplitude_outlier_min_db,
        min_label_size_for_outlier=args.min_label_size_for_outlier,
        skip_spectrograms=args.skip_spectrograms,
    )
    result = run_qc(config)

    print(f"QC rows written: {result.row_count}")
    print(f"Flagged clips: {result.flagged_count}")
    print(f"Spectrograms written: {result.spectrogram_count}")
    print(f"QC CSV: {result.qc_csv_path}")
    print(f"Summary report: {result.summary_report_path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
