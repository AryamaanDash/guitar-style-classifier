# Audio Clip QC Summary

Generated: 2026-06-25T17:52:14

## Outputs

- QC metadata CSV: `/Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/clips_qc.csv`
- Flagged clips CSV: `/Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/flagged_clips.csv`
- Summary by label CSV: `/Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/summary_by_label.csv`
- Spectrogram directory: `/Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/spectrograms`

## Thresholds

- Expected duration: 1.000 seconds
- Duration tolerance: +/- 0.050 seconds
- Expected sample rate: inferred mode Hz
- Expected channels: inferred mode
- Nearly silent: RMS <= 0.001 and peak <= 0.005
- Near clipping sample: abs(sample) >= 0.98
- Clipping flag: clipped sample percent >= 0.01
- Amplitude outlier: per-label RMS dBFS modified Z >= 3.5 and at least 12 dB from the label median
- Minimum label size for outlier detection: 5

## Counts By Label

| label | total | readable | flagged | invalid | duration | sr | channels | silent | clipping | duplicates | outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chords | 80 | 80 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| palm_muted | 68 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| single_note | 81 | 81 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

## Flagged Clips

### chords

| clip | reasons | duration | sample_rate | channels | rms_dbfs | clipped_pct | spectrogram |
| --- | --- | --- | --- | --- | --- | --- | --- |
| session_01/chords/chords_session01_0012.wav | amplitude_outlier | 1.0000 | 44100 | 2 | -34.57 | 0.0000 | /Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/spectrograms/chords/chords_session01_0012.png |
| session_01/chords/chords_session01_0068.wav | amplitude_outlier | 1.0000 | 44100 | 2 | -51.60 | 0.0000 | /Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/spectrograms/chords/chords_session01_0068.png |
| session_01/chords/chords_session01_0079.wav | nearly_silent | 1.0000 | 44100 | 2 | -inf | 0.0000 | /Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/spectrograms/chords/chords_session01_0079.png |

### single_note

| clip | reasons | duration | sample_rate | channels | rms_dbfs | clipped_pct | spectrogram |
| --- | --- | --- | --- | --- | --- | --- | --- |
| session_01/single_note/single_note_session01_0000.wav | clipping | 1.0000 | 44100 | 2 | -11.07 | 0.0363 | /Users/aryamaandash/GitHub/guitar-style-classifier/results/qc/20260625_175201/spectrograms/single_note/single_note_session01_0000.png |
