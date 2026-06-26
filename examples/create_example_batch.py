from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# Add the repository root to the Python module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import GNNDataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small preprocessed SMCL-DTA example batch."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing the processed KIBA dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/assets/example_batch.pt"),
        help="Output file for the example samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of examples to extract.",
    )
    return parser.parse_args()


def describe_sample(index: int, sample) -> None:
    print(f"\nSample {index}")
    print(sample)
    print("Keys:", list(sample.keys()))

    for key in sample.keys():
        value = getattr(sample, key)
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)

        print(
            f"  {key}: "
            f"type={type(value).__name__}, "
            f"shape={shape}, "
            f"dtype={dtype}"
        )


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {args.data_root}"
        )

    print("Loading surface-enabled KIBA test data...")
    print(f"Dataset root: {args.data_root}")

    test_dataset = GNNDataset(
        str(args.data_root),
        types="test1",
        use_surface=True,
        use_masif=True,
    )

    print(f"Complete test-set size: {len(test_dataset)}")

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero.")

    if len(test_dataset) < args.num_samples:
        raise RuntimeError(
            f"Requested {args.num_samples} samples, "
            f"but the dataset contains only {len(test_dataset)}."
        )

    sample_indices = list(range(args.num_samples))
    samples = []

    for index in sample_indices:
        sample = test_dataset[index].clone().cpu()
        samples.append(sample)
        describe_sample(index, sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.output)

    size_mb = args.output.stat().st_size / (1024 ** 2)

    print("\n" + "=" * 60)
    print("Example batch created successfully")
    print("=" * 60)
    print(f"Number of samples: {len(samples)}")
    print(f"Output file: {args.output}")
    print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
