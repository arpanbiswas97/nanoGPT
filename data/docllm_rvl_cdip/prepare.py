from rich import progress

import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from tokenizers.trainers import WordLevelTrainer

from pathlib import Path

num_proc = 8

num_proc_load_dataset = num_proc

TOKENIZER_PATH = "tokenizer.pt"
OCR_PATH = "/home/arpan/Desktop/vidya-gdrive/huggingface/docllm_data_rvl_cdip/"


# Load some of the OCR text
def get_ocr_texts(data_path: Path):
    for file in progress.track(data_path.iterdir(), description="Loading OCR texts"):
        with open(file, "r") as f:
            data = json.load(f)
        for item in data:
            yield item["text"].strip()


def get_or_build_tokenizer(ocr_texts_path: Path, tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[U]"))
        trainer = WordLevelTrainer(
            special_tokens=["[M]", "[S]", "[U]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_ocr_texts(ocr_texts_path), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def encode_bbox(bbox) -> str:
    return "{:04d}{:04d}{:04d}{:04d}".format(
        int(bbox["top"][:4]),
        int(bbox["left"][:4]),
        int(bbox["height"][:4]),
        int(bbox["width"][:4]),
    )


if __name__ == "__main__":

    enc = get_or_build_tokenizer(Path(OCR_PATH), Path(TOKENIZER_PATH))

    def process(example):
        ids = []
        bboxes = []
        for word in example:
            ids.append(enc.encode(word["text"]))
            bboxes.append(encode_bbox(word["bbox"]))
        out = {"ids": ids, "bboxes": bboxes}
        return out

    tokenized_dataset = []
    for file in progress.track(Path(OCR_PATH).iterdir(), description="Tokenizing OCR"):
        with open(file, "r") as f:
            data = json.load(f)
        tokenized_dataset.append(process(data))

    with open("tokenized_dataset.json", "w") as f:
        json.dump(tokenized_dataset, f)
