"""Dump Surya's fast layout model (RF-DETR) and its reading-order head, stage by stage.

    python tools/reference/dump_surya_layout.py --model /tmp/hf/surya_layout2 \
        --surya /home/user/theolivenbaum/surya \
        --out tests/data/reference/surya_layout \
        tests/data/documents/page_two_column.png tests/data/documents/page_math.png

Runs the vendored detector exactly as ``surya.common.rfdetr_torch.RfDetrTorch.detect`` does (float32,
CPU) with forward hooks on every stage, then ``surya.fast_layout.build_layout_result`` with the
``OrderPredictor`` — the in-process half of ``surya.fast_layout.server.LayoutEngine.run_batch``,
without the HTTP server in front of it.

Writes, per page, ``<out>/<stem>.npz`` holding

- preprocessing: ``image.raw`` (uint8 HWC RGB, after ``convert("RGB")``), ``pixel_values``
  ([3, R, R], after ``to_tensor``, ``normalize`` and the antialiased bilinear ``resize``);
- the windowed DINOv2 backbone: ``bb.pos_embed`` (the bicubic-antialias interpolated patch position
  embeddings, [H*W, C]), ``bb.embed`` (the windowed embeddings, [windows, 1+tokens, C]),
  ``bb.layer{i}`` (each layer's output, i = 1..12), ``bb.stage{k}`` (the four layer-normed,
  un-windowed feature maps, [C, H, W]);
- ``proj`` (the multi-scale projector output, [256, H, W] — also the reading-order head's input);
- the two-stage encoder head: ``enc.memory`` (enc_output + norm, [HW, 256]), ``enc.class``
  ([HW, classes]), ``enc.coord`` (re-parameterised proposals, [HW, 4]), ``enc.topk`` (the selected
  proposals), ``enc.refpoints`` ([Q, 4]);
- the decoder: ``dec.refpoints`` (the initial reference boxes after re-parameterisation),
  ``dec.query_pos``, ``dec.layer{i}`` (each layer's output, before the final norm), ``dec.hs``;
- the heads: ``pred_logits`` [Q, classes], ``pred_boxes`` [Q, 4] (cxcywh, normalised);
- post-processing: ``post.scores``, ``post.labels``, ``post.boxes`` (top ``num_select``, pixels),
  and ``det.*`` (after the confidence threshold);
- the reading-order head: ``order.memory`` (the box tokens after the context decoder),
  ``order.logits`` (one row per decode step), ``order.seq`` (the raster positions it emitted).

and ``<out>/<stem>.json`` with the final ``LayoutResult`` (polygons, labels, positions,
confidences) and the shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/hf/surya_layout2")
    ap.add_argument("--surya", default="/home/user/theolivenbaum/surya")
    ap.add_argument("--out", type=Path, default=Path("tests/data/reference/surya_layout"))
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.surya)

    from surya.common.order.predictor import OrderPredictor
    from surya.common.rfdetr_torch import load_detector
    from surya.fast_layout import build_layout_result

    torch.manual_seed(0)
    detector = load_detector(a.model, device="cpu")
    rf = detector.model  # RFDetrDetector
    model = rf.model  # LWDETR
    order = OrderPredictor(str(Path(a.model) / "order"))
    a.out.mkdir(parents=True, exist_ok=True)

    for path in a.images:
        t: dict[str, np.ndarray] = {}
        hooks = []

        def put(name, value):
            # First call wins: LWDETR.forward calls enc_out_class_embed[0] a second time, on the
            # selected queries, after the transformer's call over every proposal.
            if name not in t:
                t[name] = value.detach().cpu().numpy().copy()

        def hook(module, name, fn=lambda o: o):
            hooks.append(module.register_forward_hook(lambda m, i, o: put(name, fn(o))))

        def observe(module, fn):
            # fn returns None, so the hook leaves the module's output alone.
            hooks.append(module.register_forward_hook(lambda m, i, o: fn(o)))

        def stages(o):
            for k, f in enumerate(o):
                put(f"bb.stage{k}", f[0])

        def transformer_out(o):
            put("dec.hs", o[0][-1][0])
            put("enc.refpoints", o[3][0])

        def heads(o):
            put("pred_logits", o["pred_logits"][0])
            put("pred_boxes", o["pred_boxes"][0])

        dino = model.backbone[0].encoder.encoder  # WindowedDinov2WithRegistersBackbone
        hooks.append(
            dino.embeddings.register_forward_hook(lambda m, i, o: put("bb.embed", o))
        )
        for i, layer in enumerate(dino.encoder.layer):
            hook(layer, f"bb.layer{i + 1}", lambda o: o[0])
        observe(model.backbone[0].encoder, stages)
        hook(model.backbone[0].projector, "proj", lambda o: o[0][0])

        tr = model.transformer
        hook(tr.enc_output_norm[0], "enc.memory", lambda o: o[0])
        hook(tr.enc_out_class_embed[0], "enc.class", lambda o: o[0])
        hook(tr.enc_out_bbox_embed[0], "enc.delta", lambda o: o[0])
        hook(tr.decoder.ref_point_head, "dec.query_pos", lambda o: o[0])
        for i, layer in enumerate(tr.decoder.layers):
            hook(layer, f"dec.layer{i}", lambda o: o[0])
        hooks.append(
            tr.decoder.register_forward_pre_hook(
                lambda m, args, kwargs: put("dec.refpoints", kwargs["refpoints_unsigmoid"][0]),
                with_kwargs=True,
            )
        )
        observe(tr, transformer_out)
        observe(model, heads)
        # The order head: the encoded box tokens and every decode step's logits.
        steps: list[np.ndarray] = []
        orig_encode = order.model.encode

        def encode(*args, **kwargs):
            out = orig_encode(*args, **kwargs)
            put("order.memory", out[0])
            return out

        order.model.encode = encode
        hooks.append(
            order.model.out.register_forward_hook(lambda m, i, o: steps.append(o.detach().numpy().copy()))
        )

        image = Image.open(path)
        rgb = image.convert("RGB")
        t["image.raw"] = np.asarray(rgb, dtype=np.uint8)

        # The exact preprocessing of RFDetrDetector.predict, kept for the byte comparison.
        import torchvision.transforms.functional as TF

        from surya.common.rfdetr.predictor import IMAGENET_MEAN, IMAGENET_STD

        x = TF.to_tensor(rgb)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        x = TF.resize(x, (rf.resolution, rf.resolution))
        t["pixel_values"] = x.numpy()
        with torch.no_grad():
            pe = dino.embeddings.interpolate_pos_encoding(
                torch.zeros(1, 1 + (rf.resolution // 16) ** 2, 384), rf.resolution, rf.resolution
            )
        t["bb.pos_embed"] = pe[0, 1:].numpy()

        dets = detector.detect([image], threshold=a.threshold, return_features=True)
        result = build_layout_result(image, dets[0], order)
        for h in hooks:
            h.remove()
        order.model.encode = orig_encode

        # Recompute what the hooks cannot see directly, from what they did.
        props = proposals(t["proj"].shape[1], t["proj"].shape[2])
        delta = t.pop("enc.delta")
        t["enc.coord"] = np.concatenate(
            [delta[:, :2] * props[:, 2:] + props[:, :2], np.exp(delta[:, 2:]) * props[:, 2:]], axis=1
        ).astype(np.float32)
        t["enc.topk"] = torch.topk(torch.from_numpy(t["enc.class"]).max(-1)[0], 300)[1].numpy()

        logits = torch.from_numpy(t["pred_logits"])[None]
        boxes = torch.from_numpy(t["pred_boxes"])[None]
        post = rf.postprocess(
            {"pred_logits": logits, "pred_boxes": boxes},
            target_sizes=torch.tensor([[image.height, image.width]]),
        )[0]
        for k in ("scores", "labels", "boxes"):
            t[f"post.{k}"] = post[k].numpy()
        keep = post["scores"] > a.threshold
        for k in ("scores", "labels", "boxes"):
            t[f"det.{k}"] = post[k][keep].numpy()
        if steps:
            t["order.logits"] = np.stack(steps)
            seq = []
            # Replay the constrained argmax over the recorded logits.
            k = t["order.logits"].shape[0]
            used = np.zeros(k, dtype=bool)
            for row in t["order.logits"]:
                masked = row[:k].copy()
                masked[used] = -np.inf
                nxt = int(masked.argmax())
                seq.append(nxt)
                used[nxt] = True
            t["order.seq"] = np.array(seq, dtype=np.int64)

        stem = Path(path).stem
        np.savez_compressed(a.out / f"{stem}.npz", **t)
        meta = {
            "image": str(path),
            "size": [image.width, image.height],
            "threshold": a.threshold,
            "detections": [
                {"label": d["label"], "label_id": d["label_id"], "score": d["score"], "bbox": d["bbox"]}
                for d in dets[0]
            ],
            "result": {
                "image_bbox": result.image_bbox,
                "bboxes": [
                    {
                        "polygon": b.polygon,
                        "label": b.label,
                        "raw_label": b.raw_label,
                        "position": b.position,
                        "confidence": b.confidence,
                    }
                    for b in result.bboxes
                ],
            },
            "shapes": {k: list(v.shape) for k, v in t.items()},
        }
        (a.out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        print(f"{stem}: {len(result.bboxes)} boxes, {len(t)} tensors")


def proposals(h: int, w: int) -> np.ndarray:
    """gen_encoder_output_proposals(unsigmoid=False) for one unpadded level."""
    gy, gx = torch.meshgrid(
        torch.linspace(0, h - 1, h, dtype=torch.float32),
        torch.linspace(0, w - 1, w, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
    scale = torch.tensor([w, h]).view(1, 1, 2)
    grid = (grid + 0.5) / scale
    wh = torch.ones_like(grid) * 0.05
    return torch.cat((grid, wh), -1).view(-1, 4).numpy()


if __name__ == "__main__":
    main()
