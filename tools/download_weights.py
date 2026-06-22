"""
Download the pretrained RAG-Gesture model weights from the Hugging Face Hub.

The HF repo
    https://huggingface.co/m-hamza-mughal/rag-gesture-weights
bundles:
  - vae/          -> placed under experiments/vae/
  - diffusion/    -> placed under experiments/diffusion/
  - assets_deps/  -> placed under datasets/assets_deps/ (SMPL-X model files etc.)

After running this script you should have:

    experiments/
    ├── vae/
    │   ├── 0903_020101_..._upper_.../          # upper-body VAE
    │   ├── 0909_130750_..._face_.../           # face VAE
    │   ├── 0909_132647_..._hands_.../          # hands VAE
    │   └── 1031_142417_..._lowerplustrans_.../ # lower + translation VAE
    └── diffusion/
        └── base_beatx_len150fps15_finalweights/

    datasets/
    └── assets_deps/
        └── smplx_models/

Usage:
    python tools/download_weights.py
    python tools/download_weights.py --experiments-dir experiments --datasets-dir datasets
"""

import argparse
import os

from huggingface_hub import snapshot_download


HF_REPO_ID = "m-hamza-mughal/rag-gesture-weights"
DEFAULT_EXPERIMENTS_DIR = "experiments"
DEFAULT_DATASETS_DIR = "datasets"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--experiments-dir",
        default=DEFAULT_EXPERIMENTS_DIR,
        help=(
            "Where vae/ and diffusion/ should land "
            f"(default: {DEFAULT_EXPERIMENTS_DIR})."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        default=DEFAULT_DATASETS_DIR,
        help=(
            "Where assets_deps/ should land "
            f"(default: {DEFAULT_DATASETS_DIR})."
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


def _fetch(target_dir, patterns, revision, token):
    target = os.path.abspath(target_dir)
    os.makedirs(target, exist_ok=True)
    print(f"Downloading {patterns} from {HF_REPO_ID} -> {target}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=target,
        allow_patterns=patterns,
        revision=revision,
        token=token,
    )


def main():
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")

    # vae/ and diffusion/ go under experiments/
    _fetch(
        args.experiments_dir,
        ["vae/**", "diffusion/**"],
        args.revision,
        token,
    )

    # assets_deps/ goes under datasets/
    _fetch(
        args.datasets_dir,
        ["assets_deps/**"],
        args.revision,
        token,
    )

    print(
        "Done. Configs reference experiments/vae/, experiments/diffusion/, and "
        "datasets/assets_deps/ — verify the layout matches before training or "
        "running inference."
    )


if __name__ == "__main__":
    main()
