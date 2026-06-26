from __future__ import annotations
 
import json
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLIPS_CSV = PROJECT_ROOT / "data" / "metadata" / "clips.csv"
SESSIONS_CSV = PROJECT_ROOT / "data" / "metadata" / "sessions.csv"
PROCESSED_DICTIONARY = PROJECT_ROOT / "data" / "processed"

MODEL_DIRECTORY = PROJECT_ROOT / "models"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "baseline"

MODEL_PATH = MODEL_DIRECTORY / "baseline_logistic_regression.joblib"
FEATURE_TABLE_PATH = RESULTS_DIRECTORY / "extracted_features.csv"
FEATURE_CONFIG_PATH = RESULTS_DIRECTORY/ "feature_config.json"

TARGET_SAMPLE_RATE = 16000
CLIP_DURATION_SECONDS = 1
EXPECTED_SAMPLE_COUNT = TARGET_SAMPLE_RATE * CLIP_DURATION_SECONDS

N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256

VALID_SPLITS = {"train", "validation", "test"}
TRUE_VALUES = {"true", "1", "yes", "y"}

def append_summary_statistics(
        output_values: list[float],
        output_names: list[str],
        feature_name: str,
        values: np.ndarray,
) -> None:
    flattened = np.asarray(values, dtype = np.float64).reshape(-1)

    output_values.extend(
        [
            float(np.mean(flattened)),
            float(np.std(flattened))
        ]
    )

    output_names.extend(
        [
            f"{feature_name}_mean",
            f"{feature_name}_std"
        ]
    )

def load_audio(clip_path: Path) -> np.ndarray:
    audio, _ = librosa.load(
        clip_path,
        sr = TARGET_SAMPLE_RATE,
        mono = True
    )

    audio = librosa.util.fix_length(
        audio,
        size = EXPECTED_SAMPLE_COUNT
    )

    return np.asarray

def extract_features(
        clip_path: Path
) -> tuple[np.ndarray, list[str]]:
    
    audio = load_audio(clip_path)

    feature_values: list[float] = []
    feature_names: list[str] = []

    mfcc = librosa.feature.mfcc(
        y = audio,
        sr = TARGET_SAMPLE_RATE,
        n_mfcc= N_MFCC,
        n_fft = N_FFT,
        hop_length = HOP_LENGTH
    )

    for coefficient_index in range(N_MFCC):
        append_summary_statistics(
            output_values = feature_values,
            output_names = feature_names,
            feature_name = f"mfcc_{coefficient_index + 1}",
            values = mfcc[coefficient_index]
        )
    
    rms = librosa.feature.rms(
        y = audio,
        frame_length = N_FFT,
        hop_length = HOP_LENGTH,
    )

    append_summary_statistics(
        feature_values,
        feature_name,
        "rms",
        rms
    )

    feature_values.extend(
        [
            float(np.max(rms)),
            float(np.percentile(rms, 90))
        ]
    )

    feature_names.extend(
        [
            "rms_max",
            "rms_90th_percentile"
        ]
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio,
        frame_length = N_FFT,
        hop_length = HOP_LENGTH
    )

    append_summary_statistics(
        feature_values,
        feature_names,
        "zero_crossing_rate",
        zero_crossing_rate
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y = audio,
        sr = TARGET_SAMPLE_RATE,
        n_fft = N_FFT,
        hop_length = HOP_LENGTH
    )

    append_summary_statistics(
        feature_values,
        feature_names,
        "spectral_centroid",
        spectral_centroid
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    append_summary_statistics(
        feature_values,
        feature_names,
        "spectral_bandwidth",
        spectral_bandwidth,
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )

    append_summary_statistics(
        feature_values,
        feature_names,
        "spectral_rolloff",
        spectral_rolloff,
    )

    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    append_summary_statistics(
        feature_values,
        feature_names,
        "spectral_flatness",
        spectral_flatness,
    )

    feature_array = np.asarray(feature_values, dtype=np.float64)

    # Protect the classifier from undefined or infinite feature values.
    feature_array = np.nan_to_num(
        feature_array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return feature_array, feature_names

def load_metadata() -> pd.DataFrame:
    """
    Load clip metadata and assign each clip its session-level dataset split.
    """

    if not CLIPS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find clip metadata: {CLIPS_CSV}"
        )

    if not SESSIONS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find session metadata: {SESSIONS_CSV}"
        )

    clips = pd.read_csv(
        CLIPS_CSV,
        dtype=str,
    ).fillna("")

    sessions = pd.read_csv(
        SESSIONS_CSV,
        dtype=str,
    ).fillna("")

    required_clip_columns = {
        "clip_filename",
        "label",
        "session",
    }

    required_session_columns = {
        "session",
        "dataset_split",
    }

    missing_clip_columns = required_clip_columns - set(clips.columns)
    missing_session_columns = required_session_columns - set(sessions.columns)

    if missing_clip_columns:
        raise ValueError(
            "clips.csv is missing columns: "
            + ", ".join(sorted(missing_clip_columns))
        )

    if missing_session_columns:
        raise ValueError(
            "sessions.csv is missing columns: "
            + ", ".join(sorted(missing_session_columns))
        )

    sessions["dataset_split"] = (
        sessions["dataset_split"]
        .str.strip()
        .str.lower()
    )

    # Every file from a session must use the same dataset split.
    split_counts = (
        sessions.groupby("session")["dataset_split"]
        .nunique()
    )

    conflicting_sessions = split_counts[split_counts > 1]

    if not conflicting_sessions.empty:
        raise ValueError(
            "These sessions have conflicting dataset_split values: "
            + ", ".join(conflicting_sessions.index)
        )

    session_splits = (
        sessions[["session", "dataset_split"]]
        .drop_duplicates(subset=["session"])
    )

    clips = clips.merge(
        session_splits,
        on="session",
        how="left",
        validate="many_to_one",
    )

    clips["dataset_split"] = clips["dataset_split"].fillna("")

    # Ignore pilot clips and any session without a valid split.
    clips = clips[
        clips["dataset_split"].isin(VALID_SPLITS)
    ].copy()

    if clips.empty:
        raise ValueError(
            "No clips belong to train, validation, or test sessions. "
            "Update dataset_split in sessions.csv."
        )

    # Use human quality-control decisions when they exist.
    possible_review_columns = [
        "human_usable",
        "usable",
    ]

    review_column = next(
        (
            column
            for column in possible_review_columns
            if column in clips.columns
        ),
        None,
    )

    if review_column is not None:
        normalized_review = (
            clips[review_column]
            .str.strip()
            .str.lower()
        )

        # Only apply the filter when at least one human-review value was filled.
        if normalized_review.ne("").any():
            clips = clips[
                normalized_review.isin(TRUE_VALUES)
            ].copy()

    return clips

def build_feature_table(metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract numerical features from every accepted clip.
    """

    records: list[dict[str, object]] = []
    expected_feature_names: list[str] | None = None

    total_clips = len(metadata)

    for completed_count, row in enumerate(
        metadata.itertuples(index=False),
        start=1,
    ):
        clip_path = (
            PROCESSED_DIRECTORY
            / row.session
            / row.label
            / row.clip_filename
        )

        if not clip_path.exists():
            raise FileNotFoundError(
                f"Clip listed in metadata does not exist: {clip_path}"
            )

        feature_vector, feature_names = extract_features(clip_path)

        if expected_feature_names is None:
            expected_feature_names = feature_names
        elif feature_names != expected_feature_names:
            raise RuntimeError(
                f"Feature names changed while processing {clip_path}"
            )

        record: dict[str, object] = {
            "clip_filename": row.clip_filename,
            "label": row.label,
            "session": row.session,
            "dataset_split": row.dataset_split,
        }

        record.update(
            dict(zip(feature_names, feature_vector, strict=True))
        )

        records.append(record)

        if completed_count % 50 == 0 or completed_count == total_clips:
            print(
                f"Extracted features from "
                f"{completed_count}/{total_clips} clips"
            )

    if expected_feature_names is None:
        raise RuntimeError("No features were extracted.")

    return pd.DataFrame(records), expected_feature_names

def save_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: np.ndarray,
    split_name: str,
) -> None:
    """
    Save the confusion matrix as both a CSV table and an image.
    """

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_names,
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in class_names],
        columns=[f"predicted_{name}" for name in class_names],
    )

    matrix_table.to_csv(
        RESULTS_DIRECTORY / f"{split_name}_confusion_matrix.csv"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        f"{split_name.capitalize()} confusion matrix"
    )

    figure.tight_layout()
    figure.savefig(
        RESULTS_DIRECTORY / f"{split_name}_confusion_matrix.png",
        dpi=160,
    )

    plt.close(figure)

def evaluate_model(
    model: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> None:
    """
    Evaluate the model and save its metrics.
    """

    predictions = model.predict(features)
    class_names = model.named_steps["classifier"].classes_

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions,
    )

    report = classification_report(
        labels,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_table = pd.DataFrame(report).transpose()

    report_table.to_csv(
        RESULTS_DIRECTORY / f"{split_name}_classification_report.csv"
    )

    save_confusion_matrix(
        true_labels=labels,
        predicted_labels=predictions,
        class_names=class_names,
        split_name=split_name,
    )

    print(f"\n{split_name.capitalize()} results")
    print("-" * 40)
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    print(
        classification_report(
            labels,
            predictions,
            labels=class_names,
            zero_division=0,
        )
    )

def select_split(
    feature_table: pd.DataFrame,
    feature_names: list[str],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the feature matrix and labels for one dataset split.
    """

    split_rows = feature_table[
        feature_table["dataset_split"] == split_name
    ]

    if split_rows.empty:
        raise ValueError(
            f"No clips were assigned to the '{split_name}' split."
        )

    features = split_rows[feature_names].to_numpy(
        dtype=np.float64
    )

    labels = split_rows["label"].to_numpy()

    return features, labels

def main() -> None:
    MODEL_DIRECTORY.mkdir(parents = TRUE, exist_ok = TRUE)
    RESULTS_DIRECTORY.mkdir(parents = TRUE, exist_ok = TRUE)

    metadata = load_metadata()

    print("Clips selected after metadata and QC filtering")
    print(
        metadata.groupby(
            ["dataset_split", "label"]
        ).size()
    )

    feature_table, feature_names = build_feature_table(metadata)

    feature_table.to_csv(
        FEATURE_TABLE_PATH,
        index = False
    )

    feature_config = {
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "clip_duration_seconds": CLIP_DURATION_SECONDS,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "feature_names": feature_names
    }

    FEATURE_CONFIG_PATH.write_text(
        json.dumps(feature_config, indent=2),
        encoding="utf-8",
    )

    x_train, y_train = select_split(
        feature_table,
        feature_names,
        "train",
    )

    x_validation, y_validation = select_split(
        feature_table,
        feature_names,
        "validation",
    )

    x_test, y_test = select_split(
        feature_table,
        feature_names,
        "test",
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining logistic-regression baseline...")
    model.fit(x_train, y_train)

    model_bundle = {
        "model": model,
        "feature_names": feature_names,
        "feature_config": feature_config,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    evaluate_model(
        model,
        x_validation,
        y_validation,
        "validation",
    )

    evaluate_model(
        model,
        x_test,
        y_test,
        "test",
    )

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved feature table to: {FEATURE_TABLE_PATH}")
    print(f"Saved evaluation results to: {RESULTS_DIRECTORY}")

if __name__ == "__main__":
    main()