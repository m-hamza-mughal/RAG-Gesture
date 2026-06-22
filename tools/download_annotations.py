"""
Download the BEAT2 English dataset plus all RAG-Gesture / MIBURI additional
annotations from the Hugging Face Hub and place them under
``datasets/beat_english_v2.0.0/``.

The repository at
    https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations
bundles the base BEAT2 release (smplxflame_30, wave16k, textgrid, sem,
weights, ...) together with the new modalities introduced by RAG-Gesture and
MIBURI (discourse relations, prominence, whisper transcriptions, 25 fps
smplx, ...). A single run of this script is enough to set up the dataset.

Usage:
    python tools/download_annotations.py
    python tools/download_annotations.py --target datasets/beat_english_v2.0.0
"""

import argparse
import os

from huggingface_hub import snapshot_download


HF_REPO_ID = "m-hamza-mughal/beat2-additional-annotations"
DEFAULT_TARGET = os.path.join("datasets", "beat_english_v2.0.0")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=(
            "Directory to place the annotations into. Should be your BEAT2 "
            f"English root (default: {DEFAULT_TARGET})."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch, tag, or commit) to pin.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF access token (or set HF_TOKEN env var).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target = os.path.abspath(args.target)
    os.makedirs(target, exist_ok=True)

    print(f"Downloading {HF_REPO_ID} -> {target}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=target,
        revision=args.revision,
        token=args.token or os.getenv("HF_TOKEN"),
    )
    print("Done. Verify the layout matches datasets/DATASETS.md.")


if __name__ == "__main__":
    main()
