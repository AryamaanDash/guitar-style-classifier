# guitar-style-classifier
Machine Learning Model for classifying guitar playing techniques from audio.

## Processed Clip Quality Control

Run the automated QC workflow from the repository root:

```bash
python3 src/qc_audio_clips.py
```

The workflow reads `data/metadata/clips.csv`, recursively scans
`data/processed/` for WAV files, and leaves all WAV files unchanged. Each run
writes results to a new timestamped directory under `results/qc/`:

- `clips_qc.csv`: all original `clips.csv` columns plus clearly named `qc_*`
  columns.
- `flagged_clips.csv`: only clips with one or more QC flags.
- `summary_by_label.csv`: per-label counts for each flag type.
- `summary_report.md`: per-label counts plus the flagged clip list.
- `spectrograms/`: spectrogram PNGs for readable flagged clips only.

Default thresholds are intentionally conservative and can be changed with CLI
flags:

- Duration: expected `1.0` seconds, flagged when outside `+/- 0.05` seconds.
- Sample rate and channels: inferred from the most common readable clip values
  unless `--expected-sample-rate-hz` or `--expected-channels` is provided.
- Near silence: RMS `<= 0.001` and peak amplitude `<= 0.005`, using normalized
  PCM amplitudes.
- Clipping: samples with absolute normalized amplitude `>= 0.98` are counted;
  clips are flagged when those samples are at least `0.01%` of all samples.
- Amplitude outliers: computed within each label from RMS dBFS using robust
  modified Z scores, requiring modified Z `>= 3.5` and at least `12 dB` from
  the label median.
- Exact duplicates: detected by hashing decoded audio parameters plus the audio
  frames.

Examples:

```bash
python3 src/qc_audio_clips.py --expected-sample-rate-hz 44100 --expected-channels 2
python3 src/qc_audio_clips.py --duration-tolerance-seconds 0.02
python3 src/qc_audio_clips.py --skip-spectrograms
python3 src/qc_audio_clips.py --output-dir results/qc/manual_check
```
