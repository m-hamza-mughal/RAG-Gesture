This directory holds the raw source dataset RAG-Gesture trains on.

## Download the dataset

Everything you need lives in a single Hugging Face dataset repo:
[`m-hamza-mughal/beat2-additional-annotations`](https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations).
It bundles the BEAT2 English release together with the additional annotations
introduced by RAG-Gesture and MIBURI (discourse relations, prosodic
prominence, whisper transcriptions, 25 fps SMPL-X, etc.) — i.e. everything
[`mogen/datasets/beatx_dataset.py`](../mogen/datasets/beatx_dataset.py) and
the configs in [`configs/_base_/datasets/`](../configs/_base_/datasets/)
expect.

Fetch it with the helper script:

```bash
python tools/download_annotations.py
```

This snapshots the HF repo into `datasets/beat_english_v2.0.0/`. Override the
location if you'd rather keep the data elsewhere:

```bash
python tools/download_annotations.py --target /path/to/beat_english_v2.0.0
```

Optional flags:
- `--revision <branch|tag|commit>` — pin a specific snapshot.
- `--token <hf_token>` — pass an HF access token (or set `HF_TOKEN` in your env).

## Expected layout

After the download, `datasets/beat_english_v2.0.0/` should contain (at least)
these subdirectories:

```
datasets/beat_english_v2.0.0/
├── smplxflame_30/           # 30 fps SMPL-X body + FLAME expressions
├── smplxflame_25/           # 25 fps SMPL-X body + FLAME expressions
├── textgrid/                # forced-aligned word transcripts
├── whisper_transcription/   # cased + punctuated Whisper transcripts
├── wave16k/                 # 16 kHz audio
├── sem/                     # semantic gesture annotations
├── discourse_rels/          # PDTB-style discourse relation annotations
├── prom/                    # prosodic prominence values (.prom files)
├── weights/                 # pretrained motion autoencoder weights
├── train_test_split.csv     # split file
└── readme.md
```

Once the download is complete, no further setup is required — training,
inference, and evaluation scripts will pick up these paths automatically.
