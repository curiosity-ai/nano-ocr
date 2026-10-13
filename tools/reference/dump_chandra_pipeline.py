"""Run Chandra 2's whole pipeline (``InferenceManager(method="hf")``) on an image.

    python tools/reference/dump_chandra_pipeline.py --chandra /path/to/chandra --model /tmp/hf/chandra-ocr-2 \
        --image tests/data/documents/page_math.png --crop 0 0 800 800 \
        --out tests/data/reference/chandra_pipeline/chandra.json

Chandra's own ``load_file`` (upscaling to MIN_IMAGE_DIM), ``generate_hf`` (scale_to_fit, the chat
template, greedy decoding) and output parsers produce the result. The only change is numeric: the
model keeps its bfloat16 weights but computes in float32 (``dump_vlm.widen_on_use``), the arithmetic
the C# port does — Chandra's default bfloat16 activations would make a token-level comparison
meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_vlm import widen_on_use  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chandra", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--crop", type=int, nargs=4, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.chandra)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import chandra.model.hf as hf
    from chandra.input import load_file
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.settings import settings

    torch.set_num_threads(4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = Path(args.image)
    if args.crop:
        cropped = out.parent / ("input" + source.suffix)
        Image.open(source).convert("RGB").crop(tuple(args.crop)).save(cropped)
        source = cropped

    def load_model():
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager")
        widen_on_use(model)
        model = model.eval()
        processor = AutoProcessor.from_pretrained(args.model)
        processor.tokenizer.padding_side = "left"
        model.processor = processor
        return model

    hf.load_model = load_model
    import chandra.model as cm

    cm.load_model = load_model
    settings.MODEL_CHECKPOINT = args.model

    images = load_file(str(source), {})
    manager = InferenceManager(method="hf")
    results = manager.generate(
        [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images],
        max_output_tokens=args.max_tokens,
    )
    r = results[0]
    data = {
        "input": str(source),
        "loaded_size": list(images[0].size),
        "raw": r.raw,
        "markdown": r.markdown,
        "html": r.html,
        "chunks": r.chunks,
        "page_box": r.page_box,
        "token_count": r.token_count,
        "max_tokens": args.max_tokens,
        "images": {k: list(v.size) for k, v in r.images.items()},
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(r.raw)
    print("----")
    print(r.markdown)


if __name__ == "__main__":
    main()
