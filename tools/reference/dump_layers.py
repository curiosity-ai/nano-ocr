"""Dump every intermediate tensor of every model, as ONNX Runtime computes it.

    python tools/reference/dump_layers.py --models /path/to/models --docs tests/data/documents \
        --out tests/data/reference/layers

For each model this writes ``<model>.npz`` holding the graph inputs (``in.<k>``), and the output
of every node in execution order (``v.<k>``), plus ``<model>.json`` naming them:
{"inputs": [names], "values": [names]}. The C# parity tests (LayerParityTests) feed the same
inputs to NanoOcr.Onnx.InferenceSession and compare each node output.

The inputs are the real ones the pipeline builds — a detector input made by page_to_input, line
crops made by crop_line, order features made by line_features, ... — so the tensors cover the
value ranges the models actually see. They are kept small so the dumps stay a few tens of MB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]

from indra_ocr.det_post import decode, page_to_input  # noqa: E402
from indra_ocr.order_post import line_features  # noqa: E402
from indra_ocr.pipeline import orient_input  # noqa: E402
from indra_ocr.rec_preproc import crop_line, pad_batch  # noqa: E402

sys.path.insert(0, str(ROOT / "tools/pagetype"))
from pagetype_input import model_input  # noqa: E402


def all_outputs_session(path: Path) -> tuple[ort.InferenceSession, list[str], list[str]]:
    """A session whose outputs are every node output (in node order), and those names."""
    model = onnx.load(str(path))
    graph = model.graph
    existing = {o.name for o in graph.output}
    values = []
    for node in graph.node:
        if node.op_type == "Constant":
            continue
        for name in node.output:
            if name and name not in values:
                values.append(name)
                if name not in existing:
                    graph.output.append(onnx.ValueInfoProto(name=name))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    inputs = [i.name for i in sess.get_inputs()]
    return sess, inputs, values


def dump(models: Path, out: Path, name: str, feed: dict) -> dict:
    sess, inputs, values = all_outputs_session(models / f"{name}.onnx")
    res = sess.run(values, feed)
    arrays = {f"in.{k}": np.asarray(feed[n]) for k, n in enumerate(inputs)}
    for k, v in enumerate(res):
        arrays[f"v.{k}"] = np.asarray(v)
    np.savez_compressed(out / f"{name}.npz", **arrays)
    (out / f"{name}.json").write_text(json.dumps({"inputs": inputs, "values": values}))
    total = sum(a.nbytes for a in arrays.values())
    print(f"{name}: {len(values)} values, {total / 1e6:.1f} MB raw")
    return dict(zip(values, res))


def gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--docs", type=Path, default=ROOT / "tests/data/documents")
    ap.add_argument("--out", type=Path, default=ROOT / "tests/data/reference/layers")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    page = gray(a.docs / "page_two_column.png")
    meta = json.loads((a.models / "meta.json").read_text())

    # Detector: the page at a small long side keeps the dump compact.
    x, _ = page_to_input(page, 384)
    det = dump(a.models, a.out, "det", {"x": x[None, None]})
    boxes = decode(det["out"][0])
    print("det boxes at 384:", len(boxes))

    # Full-size detection, for real line boxes to crop.
    x, s = page_to_input(page, meta["det_long_side"])
    full = ort.InferenceSession(str(a.models / "det.onnx"), providers=["CPUExecutionProvider"])
    boxes = decode(full.run(None, {"x": x[None, None]})[0][0])
    boxes[:, :4] /= s
    boxes = boxes[np.argsort(boxes[:, 1])]

    # Recognizer, script and math classifiers: a batch of four real line crops.
    crops = [crop_line(page, b, meta["rec_height"]) for b in boxes[[2, 5, 9, 14], :4]]
    arr, widths = pad_batch(crops)
    for name in ("rec", "script", "mathcls"):
        dump(a.models, a.out, name, {"x": arr, "widths": widths})

    # CJK recognizer on the Chinese page.
    zh = gray(a.docs / "page_chinese.png")
    xz, sz = page_to_input(zh, meta["det_long_side"])
    zb = decode(full.run(None, {"x": xz[None, None]})[0][0])
    zb[:, :4] /= sz
    zb = zb[np.argsort(zb[:, 1])]
    arr_z, w_z = pad_batch([crop_line(zh, b, meta["rec_height"]) for b in zb[:3, :4]])
    dump(a.models, a.out, "rec_cjk", {"x": arr_z, "widths": w_z})

    # Orientation classifier.
    dump(a.models, a.out, "orient", {"x": orient_input(page)[None, None]})

    # Page-type classifier (not Indra's: NanoOCR's own, from tools/pagetype). It lives with the
    # embedded models rather than in the Indra model folder.
    pagetype = ROOT / "src/NanoOcr/Models/Assets"
    dump(pagetype if (pagetype / "pagetype.onnx").exists() else a.models, a.out, "pagetype", {"x": model_input(page)})

    # Order model: the page's own line boxes and some texts.
    texts = [f"line {i} text." if i % 3 else "1. item" for i in range(len(boxes))]
    feats = line_features(boxes[:, :4], (page.shape[1], page.shape[0]), texts)[None]
    dump(a.models, a.out, "order", {"feats": feats, "mask": np.ones((1, len(boxes)), bool)})

    # Math encoder and decoder on the math page.
    mp = gray(a.docs / "page_math.png")
    xm, sm = page_to_input(mp, meta["det_long_side"])
    mb = decode(full.run(None, {"x": xm[None, None]})[0][0])
    mb[:, :4] /= sm
    mb = mb[np.argsort(mb[:, 1])]
    mmeta = json.loads((a.models / "math_meta.json").read_text())["config"]
    arr_m, w_m = pad_batch([crop_line(mp, b, mmeta["height"]) for b in mb[:2, :4]])
    enc = dump(a.models, a.out, "math_enc", {"x": arr_m, "widths": w_m})
    mem, mask = enc["memory"], enc["mask"]
    init = dump(a.models, a.out, "math_dec_init", {"memory": mem})
    layers, heads = mmeta["dec_layers"], mmeta["dec_heads"]
    hd = mmeta["channels"] // heads
    B = mem.shape[0]
    sk = np.zeros((layers, B, heads, 0, hd), np.float32)
    sv = sk
    tok = np.full(B, 1, np.int64)
    step = ort.InferenceSession(str(a.models / "math_dec_step.onnx"), providers=["CPUExecutionProvider"])
    names = {i.name for i in step.get_inputs()}
    for t in range(3):
        feed = {"token": tok, "pos": np.array(t, np.int64), "self_k": sk, "self_v": sv,
                "cross_k": init["cross_k"], "cross_v": init["cross_v"], "mask": mask}
        feed = {k: v for k, v in feed.items() if k in names}
        if t == 2:
            dump(a.models, a.out, "math_dec_step", feed)
        logits, sk, sv = step.run(None, feed)
        tok = logits.argmax(-1).astype(np.int64)


if __name__ == "__main__":
    main()
