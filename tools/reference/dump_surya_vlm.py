"""Run Surya 2's VLM predictors on test pages with transformers standing in for the vLLM server.

    python tools/reference/dump_surya_vlm.py --surya /path/to/surya --model /tmp/hf/surya-ocr-2 \
        --docs tests/data/documents --out tests/data/reference/surya_vlm/surya_vlm.json

Surya talks to vLLM (or llama.cpp) over the OpenAI API; neither runs here, so this installs a
manager whose ``generate`` does what that path does — ``scale_to_fit``, the chat template, the
processor with the server's ``min_pixels``/``max_pixels``, greedy decoding, the mean token
probability from the raw logits (stop token included) — and then runs Surya's own
``LayoutPredictor``, ``RecognitionPredictor`` (full page, and block mode on the layout) and
``TableRecPredictor`` (simple), whose parsing, label mapping, blank filtering and crops are what the
C# port is checked against. The prompt is tokenized with the ``tokenizers`` library, not
transformers' loader (see dump_vlm.py). Guided decoding is not applied here: the C# side applies it,
and where the greedy output already follows the schema the two agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surya", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.surya)
    from tokenizers import Tokenizer
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from surya.inference.prompts import PROMPT_MAPPING
    from surya.inference.schema import BatchOutputItem
    from surya.inference.util import scale_to_fit
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor
    from surya.table_rec import TableRecPredictor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.model)
    size = {"shortest_edge": 3136, "longest_edge": 6291456}
    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    stops = [tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")]
    requests = []

    def generate(batch):
        outputs = []
        for item in batch:
            prompt = item.prompt or PROMPT_MAPPING[item.prompt_type]
            image = scale_to_fit(item.image)
            text = (
                "<|im_start|>user\n"
                + ("<|vision_start|><|image_pad|><|vision_end|>" + prompt).strip()
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            inputs = processor(text=[text], images=[image], images_kwargs={"size": size}, return_tensors="pt")
            count = int(inputs["image_grid_thw"].prod()) // 4
            ids = torch.tensor([tokenizer.encode(text.replace("<|image_pad|>", "<|image_pad|>" * count)).ids])
            inputs["input_ids"] = ids
            inputs["attention_mask"] = torch.ones_like(ids)
            inputs["mm_token_type_ids"] = (ids == tokenizer.token_to_id("<|image_pad|>")).int()
            max_tokens = item.max_tokens or 2048
            with torch.no_grad():
                result = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    eos_token_id=stops,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            gen = result.sequences[0, ids.shape[1]:].tolist()
            probs = [
                torch.softmax(score[0].float(), -1)[token].item() for score, token in zip(result.scores, gen)
            ]
            text_out = tokenizer.decode([t for t in gen if t not in stops], skip_special_tokens=True)
            requests.append(
                {"prompt_type": item.prompt_type, "image": list(image.size), "max_tokens": max_tokens, "raw": text_out}
            )
            outputs.append(
                BatchOutputItem(
                    raw=text_out,
                    token_count=len(gen),
                    error=False,
                    mean_token_prob=sum(probs) / len(probs) if probs else None,
                    metadata=item.metadata,
                )
            )
            print(item.prompt_type, image.size, repr(text_out[:120]), flush=True)
        return outputs

    manager = SimpleNamespace(generate=generate)
    docs = Path(args.docs)
    page = Image.open(docs / "page_math.png").convert("RGB")
    table = Image.open(docs / "page_two_column.png").convert("RGB").crop((860, 660, 1560, 920))

    layout = LayoutPredictor(manager)([page])
    full = RecognitionPredictor(manager)([page], full_page=True)
    blocks = RecognitionPredictor(manager)([page], layout, full_page=False)
    tables = TableRecPredictor(manager)([table], mode="simple")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "table_crop": [860, 660, 1560, 920],
                "layout": [r.model_dump() for r in layout],
                "full_page": [r.model_dump() for r in full],
                "blocks": [r.model_dump() for r in blocks],
                "tables": [r.model_dump() for r in tables],
                "requests": requests,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
