
"""Minimal model-loading example for SMCL-DTA."""

from contextlib import redirect_stdout
from pathlib import Path
import io
import sys


# Add the repository root to Python's import path so that this script
# can be executed directly with:
# python examples/minimal_example.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.model_0428_16_dual import MGraphDTA


def main() -> None:
    """Instantiate SMCL-DTA and report its parameter count."""

    # Suppress temporary debug messages printed during model initialization.
    with redirect_stdout(io.StringIO()):
        model = MGraphDTA(
            block_num=3,
            vocab_protein_size=26,
            embedding_size=128,
            use_surface=True,
        )

    model.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("SMCL-DTA model loaded successfully.")
    print(f"Number of trainable parameters: {trainable_parameters}")


if __name__ == "__main__":
    main()
