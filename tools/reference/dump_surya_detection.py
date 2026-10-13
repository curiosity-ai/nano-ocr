"""Dump Surya's text-line detector (EfficientViT segmentation + CRAFT-style heatmap boxes).

    /tmp/venv-surya/bin/python tools/reference/dump_surya_detection.py \
        --surya /home/user/theolivenbaum/surya --model /tmp/hf/surya_det \
        --out tests/data/reference/surya_detection

Surya pins ``opencv-python-headless==4.11.0.86``; the heatmap post-processing calls
``cv2.connectedComponentsWithStats``, ``dilate``, ``minAreaRect`` and ``boxPoints``, so run this
from a virtual environment holding that version (``python3 -m venv --system-site-packages
/tmp/venv-surya && /tmp/venv-surya/bin/pip install opencv-python-headless==4.11.0.86``) rather
than with the system OpenCV the other scripts use.

The checkpoint is ``https://models.datalab.to/text_detection/2025_05_07/`` (``model.safetensors``,
``config.json``, ``preprocessor_config.json``). It runs in float32, as Surya does on CPU, through
``DetectionPredictor``'s own ``batch_detection`` and ``parallel_get_boxes``, one document per batch
(a document's chunks always share one).

Documents (from ``tests/data/documents``): ``page_two_column.png`` (1700x2200: two chunks, the
second padded), ``page_math.png`` (one chunk), ``tall`` (1700x3000, three chunks: the two-column
page over the top of the skewed one) and ``wide`` (6600x1700, three rotated pages side by side:
two chunks whose thumbnails take Pillow's ``reduce`` path before the Lanczos resize).

Writes ``<out>/detection.npz`` holding, per document ``d``:

- ``d.image`` (the RGB input, uint8 HWC);
- per chunk ``k``: ``d.chunk{k}.thumb`` (after ``thumbnail``), ``d.chunk{k}.resized`` (after the
  second Lanczos resize, uint8 HWC), ``d.chunk{k}.pixels`` (the processor's output, float32 CHW)
  and ``d.chunk{k}.logits`` (the model output after the sigmoid, ``[2, 300, 300]``);
- ``d.heatmap`` / ``d.affinity`` (the stitched, upsampled maps).

and ``<out>/model.npz`` with the model's stages on the first chunk of ``page_two_column``
(``pixels``, its input): ``model.stem``, ``model.stage{0..3}``, ``model.stage3.block1.context`` /
``.local`` (the first EfficientViT block's two residual halves), ``model.head.c0`` /
``model.head.c3`` (the first and last ``linear_c`` projections, upsampled to stage 0's
resolution) and ``model.head.logits`` (before the sigmoid), all NCHW float32.

``<out>/detection.json`` holds per document the chunk heights, the dynamic thresholds, the raw
``detect_boxes`` output (polygons before rescaling, confidences), the final ``TextDetectionResult``
(polygons, bboxes, confidences, image_bbox) and wall-clock timings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surya", default="/home/user/theolivenbaum/surya")
    ap.add_argument("--model", default="/tmp/hf/surya_det")
    ap.add_argument("--documents", default="tests/data/documents")
    ap.add_argument("--out", default="tests/data/reference/surya_detection")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, args.surya)
    import torch
    from PIL import Image

    from surya.detection import DetectionPredictor
    from surya.detection import heatmap as hm
    from surya.detection.model.config import EfficientViTConfig
    from surya.detection.model.encoderdecoder import EfficientViTForSemanticSegmentation
    from surya.detection.processor import SegformerImageProcessor
    from surya.detection.util import split_image
    from surya.settings import settings

    torch.set_num_threads(args.threads)
    config = EfficientViTConfig.from_pretrained(args.model)
    model = EfficientViTForSemanticSegmentation.from_pretrained(args.model, dtype=torch.float32, config=config).eval()
    processor = SegformerImageProcessor.from_pretrained(args.model)

    predictor = DetectionPredictor.__new__(DetectionPredictor)
    predictor.model = model
    predictor.processor = processor
    predictor._client = None
    predictor._disable_tqdm = True

    docs = Path(args.documents)

    def load(name):
        return Image.open(docs / name).convert("RGB")

    two = load("page_two_column.png")
    skew = load("page_skewed.png")
    tall = Image.new("RGB", (1700, 3000))
    tall.paste(two, (0, 0))
    tall.paste(skew.crop((0, 0, 1700, 800)), (0, 2200))
    rot = load("page_rotated.png")
    wide = Image.new("RGB", (6600, 1700))
    for i in range(3):
        wide.paste(rot, (2200 * i, 0))
    documents = {
        "page_two_column": two,
        "page_math": load("page_math.png"),
        "tall": tall,
        "wide": wide,
    }

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    # Capture what prepare_image produces, chunk by chunk.
    captured: list[dict[str, np.ndarray]] = []
    original_prepare = DetectionPredictor.prepare_image

    def prepare(self, img):
        record = {}
        new_size = (self.processor.size["width"], self.processor.size["height"])
        thumb = img.copy()
        thumb.thumbnail(new_size, Image.Resampling.LANCZOS)
        record["thumb"] = np.asarray(thumb, dtype=np.uint8).copy()
        record["resized"] = np.asarray(thumb.resize(new_size, Image.Resampling.LANCZOS), dtype=np.uint8).copy()
        out = original_prepare(self, img)
        record["pixels"] = out.numpy().copy()
        captured.append(record)
        return out

    logits_seen: list[np.ndarray] = []

    def on_output(_module, _inputs, output):
        logits_seen.extend(output.logits.detach().float().numpy().copy())

    hook = model.register_forward_hook(on_output)
    DetectionPredictor.prepare_image = prepare

    thresholds: list = []
    raw_boxes: list = []
    original_detect = hm.detect_boxes

    def detect(linemap, text_threshold, low_text):
        thresholds.append([float(t) for t in hm.get_dynamic_thresholds(linemap, text_threshold, low_text)])
        det, conf = original_detect(linemap, text_threshold, low_text)
        raw_boxes.append({"polygons": [np.asarray(b, dtype=np.float32).tolist() for b in det],
                          "confidences": [float(c) for c in conf]})
        return det, conf

    hm.detect_boxes = detect

    for name, image in documents.items():
        captured.clear()
        logits_seen.clear()
        thresholds.clear()
        raw_boxes.clear()
        arrays[f"{name}.image"] = np.asarray(image, dtype=np.uint8)

        # The stitched maps, exactly as batch_detection yields them.
        maps = []
        for preds, _sizes in predictor.batch_detection([image]):
            maps.extend(preds)
        heatmap, affinity = maps[0]
        arrays[f"{name}.heatmap"] = heatmap.astype(np.float32)
        arrays[f"{name}.affinity"] = affinity.astype(np.float32)
        for k, record in enumerate(captured):
            for key, value in record.items():
                arrays[f"{name}.chunk{k}.{key}"] = value
            arrays[f"{name}.chunk{k}.logits"] = logits_seen[k]

        result = hm.parallel_get_boxes(maps[0], image.size, False)
        _parts, heights = split_image(image, processor.size["height"])
        meta[name] = {
            "size": list(image.size),
            "chunk_heights": heights,
            "heatmap_shape": list(heatmap.shape),
            "thresholds": thresholds[0],
            "raw": raw_boxes[0],
            "bboxes": [
                {"polygon": b.polygon, "bbox": b.bbox, "confidence": b.confidence} for b in result.bboxes
            ],
            "image_bbox": result.image_bbox,
        }
        print(f"{name}: {image.size}, {len(heights)} chunk(s), {len(result.bboxes)} boxes, thresholds {thresholds[0]}")

    hook.remove()
    DetectionPredictor.prepare_image = original_prepare
    hm.detect_boxes = original_detect

    # The model's stages on the first chunk of the two-column page.
    pixels = torch.from_numpy(arrays["page_two_column.chunk0.pixels"])[None]
    stages: dict[str, np.ndarray] = {}

    def keep(name):
        def hook_fn(_module, _inputs, output):
            stages[name] = output.detach().float().numpy()[0].copy()

        return hook_fn

    hooks = [model.vit.stem.register_forward_hook(keep("model.stem"))]
    for i, stage in enumerate(model.vit.stages):
        hooks.append(stage.register_forward_hook(keep(f"model.stage{i}")))
    block = model.vit.stages[3].blocks[1]
    hooks.append(block.context_module.register_forward_hook(keep("model.stage3.block1.context")))
    hooks.append(block.local_module.register_forward_hook(keep("model.stage3.block1.local")))
    head = model.decode_head
    hooks.append(head.classifier.register_forward_hook(keep("model.head.logits")))
    with torch.inference_mode():
        out = model(pixel_values=pixels)
        hidden = out.hidden_states
        size = hidden[0].shape[2:]
        for i, (state, mlp) in enumerate(zip(hidden, head.linear_c)):
            if i not in (0, 3):
                continue
            h, w = state.shape[2], state.shape[3]
            x = mlp(state).permute(0, 2, 1).reshape(1, -1, h, w)
            x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
            stages[f"model.head.c{i}"] = x.float().numpy()[0].copy()
    for h in hooks:
        h.remove()
    # Run alone rather than batched with the page's second chunk, the backend may pick other
    # kernels; the stage tensors are self-consistent with model.head.logits, not bit-equal to
    # page_two_column.chunk0.logits.
    drift = np.abs(out.logits.float().numpy()[0] - arrays["page_two_column.chunk0.logits"]).max()
    print(f"single-chunk vs batched logits: max |diff| {drift:.2e}")

    # Timings: the whole detector on one page, warm, best of three.
    timings = {}
    for name in ("page_two_column", "page_math"):
        image = documents[name]
        best_model = best_total = float("inf")
        for _ in range(4):
            t0 = time.perf_counter()
            maps = []
            for preds, _sizes in predictor.batch_detection([image]):
                maps.extend(preds)
            t1 = time.perf_counter()
            hm.parallel_get_boxes(maps[0], image.size, False)
            t2 = time.perf_counter()
            best_model = min(best_model, t1 - t0)
            best_total = min(best_total, t2 - t0)
        timings[name] = {"detect_ms": round(best_model * 1000, 1), "total_ms": round(best_total * 1000, 1)}
        print(f"{name}: model+stitch {best_model * 1000:.0f} ms, with boxes {best_total * 1000:.0f} ms")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "detection.npz", **arrays)
    np.savez(out_dir / "model.npz", pixels=arrays["page_two_column.chunk0.pixels"], **stages)
    payload = {
        "documents": meta,
        "timings": timings,
        "threads": args.threads,
        "settings": {
            "text_threshold": settings.DETECTOR_TEXT_THRESHOLD,
            "blank_threshold": settings.DETECTOR_BLANK_THRESHOLD,
            "chunk_height": settings.DETECTOR_IMAGE_CHUNK_HEIGHT,
            "y_expand": settings.DETECTOR_BOX_Y_EXPAND_MARGIN,
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "cv2": __import__("cv2").__version__},
    }
    (out_dir / "detection.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
