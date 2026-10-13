"""Dump Surya's OCR-error classifier (DistilBERT) on a set of texts.

    python tools/reference/dump_surya_ocr_error.py --surya /path/to/surya --model /tmp/hf/surya_ocr_error \
        --out tests/data/reference/surya_ocr_error/ocr_error.json

Records, per text, the token ids transformers' tokenizer produces (what Surya's processor uses),
the ids of the raw ``tokenizers`` pipeline, the classifier's float32 logits for the text on its
own, and the labels/scores of Surya's batched ``OCRErrorPredictor.local()`` (batch size 8,
padding to the longest), which is what callers get.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Optical character recognition turns the image of a page into text that a computer can search.",
    "Th3 qu!ck br0wn f0x jumpz 0ver th3 l@zy d0g;; ,,, ..",
    "lllll IIIII 11111 ||||| lI1| lI1| lI1|",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "敏捷的棕色狐狸跳过了懒狗。",
    "素早い茶色の狐がのろまな犬を飛び越える。",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "x^2 + y^2 <= 1 when the point lies in the disk",
    "",
    "   ",
    "Ünïcödé àccénts naïve café résumé",
    "tab\tseparated\ttext and\nnewlines\r\nhere",
    "control\x00chars\x07and�replacement",
    "[CLS] literal special tokens [SEP] inside [MASK] text",
    "supercalifragilisticexpialidocious" * 4,
    "word " * 700,
    "e-mail: someone@example.com, phone: +1 (555) 010-0199; price $42.50",
    "Ｆｕｌｌｗｉｄｔｈ ｌｅｔｔｅｒｓ ａｎｄ ｄｉｇｉｔｓ １２３",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surya", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.surya)
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from surya.ocr_error import OCRErrorPredictor

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.float32).eval()

    entries = []
    for text in TEXTS:
        ids = tokenizer(text, truncation=True)["input_ids"]
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0].tolist()
        entries.append({"text": text, "ids": ids, "raw_ids": raw.encode(text).ids, "logits": logits})

    predictor = OCRErrorPredictor.local(checkpoint=args.model, device="cpu", dtype=torch.float32)
    result = predictor(TEXTS, batch_size=8)
    for entry, label, score in zip(entries, result.labels, result.scores):
        entry["label"] = label
        entry["score"] = score

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    print(f"{len(entries)} texts -> {out}")
    for e in entries:
        print(e["label"], round(e["score"], 4), e["ids"] == e["raw_ids"], repr(e["text"][:40]))


if __name__ == "__main__":
    main()
