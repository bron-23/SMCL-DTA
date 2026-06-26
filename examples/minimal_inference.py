from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import contextlib
import io


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_0428_16_dual import MGraphDTA


MODEL_SOURCE = PROJECT_ROOT / "src" / "model_0428_16_dual.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal SMCL-DTA inference example."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/smcl_dta_kiba_example.pt"
        ),
        help="Path to the trained SMCL-DTA state dictionary.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "examples/assets/example_batch.pt"
        ),
        help="Path to the preprocessed example samples.",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for inference.",
    )

    return parser.parse_args()


def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model() -> MGraphDTA:
    """Construct the model using the paper configuration."""

    return MGraphDTA(
        3,
        25 + 1,
        embedding_size=128,
        filter_num=32,
        out_dim=1,
        mask_rate=0.05,
        temperature=0.1,
        disable_masking=False,
        cl_mode="regression",
        cl_similarity_threshold=0.5,
        use_surface=True,
    )


def main() -> None:
    args = parse_args()
    set_reproducible_seed(42)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available."
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Example input was not found: {args.input}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: {args.checkpoint}"
        )

    device = torch.device(args.device)

    samples = torch.load(
        args.input,
        map_location="cpu",
    )

    if not isinstance(samples, list) or len(samples) == 0:
        raise TypeError(
            "The example input must contain a non-empty list "
            "of PyTorch Geometric Data objects."
        )

    loader = DataLoader(
        samples,
        batch_size=len(samples),
        shuffle=False,
    )

    batch = next(iter(loader)).to(device)

    model_log = io.StringIO()

    with contextlib.redirect_stdout(model_log):
        model = build_model().to(device)

        state_dict = torch.load(
            args.checkpoint,
            map_location=device,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.eval()

        with torch.no_grad():
            predictions = model(batch)
    predictions = (
        predictions.detach()
        .cpu()
        .reshape(-1)
    )

    labels = (
        batch.y.detach()
        .cpu()
        .reshape(-1)
    )

    print("=" * 68)
    print("SMCL-DTA minimal worked inference example")
    print("=" * 68)
    print("Model implementation: src/model_0428_16_dual.py")
    print(f"Device: {device}")
    print(f"Number of drug-target pairs: {len(samples)}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Checkpoint loaded successfully.")
    print(f"Prediction shape: {tuple(predictions.shape)}")
    print(
        "Predicted affinities:",
        [round(float(value), 6) for value in predictions],
    )
    print(
        "Reference affinities:",
        [round(float(value), 6) for value in labels],
    )
    print("Minimal inference completed successfully.")


if __name__ == "__main__":
    main()
