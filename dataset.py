"""GSM8K loading. Splits are the official HF train (7473) / test (1319) — keep fixed across experiments."""

from datasets import load_dataset

GSM8K_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"


def load_gsm8k(split: str = "train"):
    """Return the GSM8K split as an HF Dataset with fields: question, answer."""
    return load_dataset(GSM8K_ID, GSM8K_CONFIG, split=split)


if __name__ == "__main__":
    for split in ("train", "test"):
        ds = load_gsm8k(split)
        print(split, len(ds))
    print(ds[0])
