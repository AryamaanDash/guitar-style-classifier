import csv
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "sessions.csv"
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

CLIP_LENGTH_SECONDS = 1


def split_recording(
    input_path: Path,
    output_directory: Path,
    output_prefix: str,
) -> list[dict[str, object]]:

    clip_records = []

    with wave.open(str(input_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        total_frames = source.getnframes()

        frames_per_clip = sample_rate * CLIP_LENGTH_SECONDS
        complete_clip_count = total_frames // frames_per_clip

        output_directory.mkdir(parents=True, exist_ok=True)

        for clip_index in range(complete_clip_count):
            audio_frames = source.readframes(frames_per_clip)

            clip_filename = f"{output_prefix}_{clip_index:04d}.wav"
            clip_path = output_directory / clip_filename

            with wave.open(str(clip_path), "wb") as destination:
                destination.setnchannels(channels)
                destination.setsampwidth(sample_width)
                destination.setframerate(sample_rate)
                destination.writeframes(audio_frames)

            clip_records.append(
                {
                    "clip_filename": clip_filename,
                    "source_filename": input_path.name,
                    "clip_index": clip_index,
                    "start_seconds": clip_index * CLIP_LENGTH_SECONDS,
                    "end_seconds": (clip_index + 1) * CLIP_LENGTH_SECONDS,
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                }
            )

        discarded_frames = total_frames % frames_per_clip

        if discarded_frames:
            discarded_seconds = discarded_frames / sample_rate
            print(
                f"Discarded final incomplete section of "
                f"{discarded_seconds:.2f} seconds from {input_path.name}"
            )

    return clip_records


def main() -> None:
    all_clip_records = []

    with METADATA_FILE.open(newline="", encoding="utf-8") as metadata_file:
        recordings = csv.DictReader(metadata_file)

        for recording in recordings:
            filename = recording["filename"]
            label = recording["label"]
            session = recording["session"]

            input_path = RAW_DIRECTORY / session / filename
            output_directory = PROCESSED_DIRECTORY / session / label

            if not input_path.exists():
                print(f"Skipping missing file: {input_path}")
                continue

            output_prefix = input_path.stem

            clip_records = split_recording(
                input_path=input_path,
                output_directory=output_directory,
                output_prefix=output_prefix,
            )

            for clip_record in clip_records:
                clip_record["label"] = label
                clip_record["session"] = session

            all_clip_records.extend(clip_records)

            print(
                f"Created {len(clip_records)} clips from {filename}"
            )

    clip_metadata_path = (
        PROJECT_ROOT / "data" / "metadata" / "clips.csv"
    )

    fieldnames = [
        "clip_filename",
        "source_filename",
        "label",
        "session",
        "clip_index",
        "start_seconds",
        "end_seconds",
        "sample_rate_hz",
        "channels",
    ]

    with clip_metadata_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_clip_records)

    print(f"\nCreated {len(all_clip_records)} total clips.")
    print(f"Clip metadata saved to: {clip_metadata_path}")


if __name__ == "__main__":
    main()