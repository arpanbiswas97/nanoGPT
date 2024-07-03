import os
from rich import progress
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset
from pathlib import Path

num_proc = 8

num_proc_load_dataset = num_proc

TOKENIZER_PATH = "tokenizer.pt"
OCR_TEXTS_PATH = "ocr_texts"
OCR_OUTPUT_PATH = "ocr_output"

# Load some of the OCR text
def get_ocr_texts(data_path: Path):
    data = Path(data_path)
    for file in data.glob("*.txt"):
        with file.open("r") as f:
            text = f.read()
        yield (text)


def get_or_build_tokenizer(ocr_texts_path: Path, tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[U]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[M]", "[S]", "[P]", "[U]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_ocr_texts(ocr_texts_path), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


if __name__ == "__main__":
    dataset = load_dataset("aharley/rvl_cdip", num_proc=num_proc_load_dataset)

    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    # TODO Tesseract OCR code to run through all the data and add OC
    enc = get_or_build_tokenizer(Path(OCR_TEXTS_PATH),Path(TOKENIZER_PATH))
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        out = {"ids": ids, "bboxes": bbox}
        return out
