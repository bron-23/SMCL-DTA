
---

### `examples/minimal_example.py`

```python
import torch

from src.model_0428_16_dual import MGraphDTA

def main():
    model = MGraphDTA(
        block_num=3,
        vocab_protein_size=26,
        embedding_size=128,
        use_surface=True,
        use_masif=True,
    )

    print("SMCL-DTA model loaded successfully.")
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

if __name__ == "__main__":
    main()