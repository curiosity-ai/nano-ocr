"""Scores pagetype.onnx on a held-out set built by build_eval.py: accuracy per source and the
confusion matrix, through ONNX Runtime with the same input as NanoOCR builds.

    python evaluate.py --model out/pagetype.onnx --eval eval/
"""
import argparse, collections, json, os
import cv2
import numpy as np
import onnxruntime as ort
from pagetype_input import CLASSES, model_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    items = [json.loads(l) for l in open(os.path.join(a.eval, "manifest.jsonl"))]
    rows = []
    for it in items:
        g = cv2.imread(os.path.join(a.eval, it["file"]), cv2.IMREAD_GRAYSCALE)
        p = sess.run(None, {"x": model_input(g)})[0][0]
        rows.append({**it, "pred": CLASSES[int(p.argmax())], "conf": float(p.max()), "probs": [float(v) for v in p]})
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    scored = [r for r in rows if r["class"] in CLASSES]
    by = collections.defaultdict(list)
    for r in scored:
        key = r["source"] if r["source"] != "rvl-cdip" else f"rvl-cdip/{r['rvl']}"
        by[key].append(r)
    print(f"{'source':34} {'n':>4} {'acc':>6}  predictions")
    for k in sorted(by):
        rs = by[k]
        acc = sum(r["pred"] == r["class"] for r in rs) / len(rs)
        c = collections.Counter(r["pred"] for r in rs)
        print(f"{k:34} {len(rs):4} {acc:6.2f}  {dict(c)}")
    acc = sum(r["pred"] == r["class"] for r in scored) / len(scored)
    print(f"\noverall accuracy {acc:.3f} on {len(scored)} pages")
    cm = np.zeros((4, 4), int)
    for r in scored:
        cm[CLASSES.index(r["class"]), CLASSES.index(r["pred"])] += 1
    print("rows = truth, columns = prediction:", CLASSES)
    print(cm)
    for name in ("advertisement", "file folder"):
        rs = [r for r in rows if r.get("rvl") == name]
        if rs:
            print(f"rvl-cdip/{name} (unscored): {dict(collections.Counter(r['pred'] for r in rs))}")
    cocos = [r for r in rows if r["source"] == "coco-text-val"]
    if cocos:
        print("coco without text called photo:", sum(r["pred"] == "photo" for r in cocos if not r.get("has_text")), "of", sum(not r.get("has_text") for r in cocos))


if __name__ == "__main__":
    main()
