"""Dump the intermediate tensors of a Qwen3.5 vision-language checkpoint (Surya 2, Chandra 2).

    python tools/reference/dump_vlm.py --model /tmp/hf/surya-ocr-2 \
        --image tests/data/documents/page_two_column.png --crop 100 120 900 520 \
        --prompt "OCR this block image to HTML." --out tests/data/reference/vlm/surya

The model runs in float32 with eager attention, so the tensors are what the bfloat16 weights
compute without any reduced-precision rounding; IndraOcr.Vlm keeps the weights in bfloat16 and
accumulates in float32, which is the same arithmetic up to summation order.

Writes ``<out>/vlm.npz`` holding

- the preprocessing chain: ``image.raw`` (the crop), ``image.scaled`` (after Surya/Chandra's
  ``scale_to_fit``, PIL Lanczos), ``image.resized`` (after the processor's ``smart_resize`` bicubic
  resize, uint8 HWC), ``pixel_values``, ``grid_thw``, ``input_ids``, ``position_ids``;
- the vision tower: ``vis.patch`` (patch embedding), ``vis.embed`` (plus position embedding),
  ``vis.block{i}``, ``vis.merged`` (the merger's output, in decoder width);
- the decoder over the prompt: ``txt.embeds`` (image features scattered in), ``txt.layer{i}``,
  ``txt.norm``, ``txt.logits`` (the last position), and two layers' internals:
  ``txt.l0.core`` / ``txt.l0.gate`` (the gated delta rule's output before its gated norm, and the
  gate) and ``txt.l0.out`` (layer 0's mixer output), ``txt.l3.attn`` (layer 3's attention output);
- greedy decoding: ``gen.ids`` and ``gen.logits`` (one row per generated token).

and ``<out>/vlm.json`` with the prompt text, the generated text and the shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def scale_to_fit(img, max_size=(3072, 2048), min_size=(1792, 28), grid_size=28):
    """surya.inference.util.scale_to_fit (identical to chandra.model.util.scale_to_fit)."""
    width, height = img.size
    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5
    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1
    new_size = (w_blocks * grid_size, h_blocks * grid_size)
    if new_size == (width, height):
        return img
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--crop", type=int, nargs=4, default=None)
    ap.add_argument("--prompt", default="OCR this block image to HTML.")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    # Surya's vLLM server passes these as --mm-processor-kwargs; Chandra uses the checkpoint's own.
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}

    raw = Image.open(args.image).convert("RGB")
    if args.crop:
        raw = raw.crop(tuple(args.crop))
    scaled = scale_to_fit(raw)
    arrays["image.raw"] = np.asarray(raw)
    arrays["image.scaled"] = np.asarray(scaled)

    processor = AutoProcessor.from_pretrained(args.model)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": scaled}, {"type": "text", "text": args.prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ip = processor.image_processor
    size = dict(ip.size)
    if args.min_pixels:
        size["shortest_edge"] = args.min_pixels
    if args.max_pixels:
        size["longest_edge"] = args.max_pixels
    inputs = processor(text=[text], images=[scaled], images_kwargs={"size": size}, return_tensors="pt")

    # transformers 5 loads these checkpoints' tokenizer as its own Qwen3.5 class, replacing the
    # pipeline in tokenizer.json (Surya's is character-level: ' ' is one token, not byte-level
    # 'Ġ'). The model was trained on tokenizer.json's pipeline, which is what the tokenizers
    # library runs, so the prompt is tokenized with it directly.
    from tokenizers import Tokenizer

    raw_tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    image_token = raw_tok.token_to_id("<|image_pad|>")
    count = int(inputs["image_grid_thw"].prod()) // (ip.merge_size**2)
    ids = raw_tok.encode(text.replace("<|image_pad|>", "<|image_pad|>" * count)).ids
    ids_t = torch.tensor([ids])
    inputs["input_ids"] = ids_t
    inputs["attention_mask"] = torch.ones_like(ids_t)
    inputs["mm_token_type_ids"] = (ids_t == image_token).int()

    # The processor's resized image, before rescale/normalise: the same resize call on uint8.
    tensor = torch.from_numpy(np.asarray(scaled)).permute(2, 0, 1).contiguous()[None]
    from transformers.image_utils import SizeDict

    resized = ip.resize(
        tensor,
        size=SizeDict(shortest_edge=size["shortest_edge"], longest_edge=size["longest_edge"]),
        resample=ip.resample,
        factor=ip.patch_size * ip.merge_size,
    )
    arrays["image.resized"] = resized[0].permute(1, 2, 0).numpy()

    arrays["pixel_values"] = inputs["pixel_values"].float().numpy()
    arrays["grid_thw"] = inputs["image_grid_thw"].numpy().astype(np.int64)
    arrays["input_ids"] = inputs["input_ids"][0].numpy().astype(np.int64)

    dtype = getattr(torch, args.dtype)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    inner = model.model
    visual = inner.visual
    lm = inner.language_model

    position_ids, _ = inner.get_rope_index(
        inputs["input_ids"], inputs["mm_token_type_ids"], image_grid_thw=inputs["image_grid_thw"]
    )
    arrays["position_ids"] = position_ids[:, 0].numpy().astype(np.int64)

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def keep(name, value):
        if name not in captured:  # the prompt pass only, not the decode steps
            captured[name] = value.detach().float().reshape(-1, value.shape[-1]).clone()

    def out_hook(name):
        return lambda m, a, o: keep(name, o[0] if isinstance(o, tuple) else o)

    hooks.append(visual.patch_embed.register_forward_hook(out_hook("vis.patch")))
    hooks.append(visual.blocks[0].register_forward_pre_hook(lambda m, a: keep("vis.embed", a[0])))
    for i, blk in enumerate(visual.blocks):
        hooks.append(blk.register_forward_hook(out_hook(f"vis.block{i}")))
    hooks.append(visual.merger.register_forward_hook(out_hook("vis.merged")))
    hooks.append(
        lm.layers[0].register_forward_pre_hook(lambda m, a, k: keep("txt.embeds", (a or (k["hidden_states"],))[0]), with_kwargs=True)
    )
    for i, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(out_hook(f"txt.layer{i}")))
    hooks.append(lm.norm.register_forward_hook(out_hook("txt.norm")))
    hooks.append(
        lm.layers[0].linear_attn.norm.register_forward_pre_hook(
            lambda m, a: [keep("txt.l0.core", a[0]), keep("txt.l0.gate", a[1])] and None
        )
    )
    hooks.append(lm.layers[0].linear_attn.register_forward_hook(out_hook("txt.l0.out")))
    hooks.append(lm.layers[3].self_attn.register_forward_hook(out_hook("txt.l3.attn")))

    with torch.no_grad():
        result = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    for h in hooks:
        h.remove()

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = result.sequences[0, prompt_len:].numpy().astype(np.int64)
    arrays["gen.ids"] = gen_ids
    arrays["gen.logits"] = torch.cat([s.float() for s in result.scores], 0).numpy()

    for name, value in captured.items():
        arrays[name] = value.numpy()
    arrays["txt.logits"] = arrays["gen.logits"][:1]

    np.savez(out / "vlm.npz", **arrays)
    meta = {
        "prompt": text,
        "generated": raw_tok.decode(gen_ids.tolist(), skip_special_tokens=False),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    (out / "vlm.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("prompt", "generated")}, ensure_ascii=False))
    print({k: v for k, v in meta["shapes"].items() if not k.startswith(("vis.block", "txt.layer"))})


if __name__ == "__main__":
    main()
