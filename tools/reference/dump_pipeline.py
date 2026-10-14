"""Run the reference Python pipeline on the test corpus and dump every stage.

    python tools/reference/dump_pipeline.py --models /path/to/models --docs tests/data/documents \
        --out tests/data/reference/pipeline

For each image: the orientation, the skew, the upright page, the detected boxes, the per-line
recognition (text, confidence, tokens), and the final page. For each PDF: the 200-dpi renders,
the text-layer runs, and the final pages in auto and ocr mode. PipelineParityTests compares the
C# pipeline stage by stage against these.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".reference"))

import pypdfium2 as pdfium  # noqa: E402

from indra_ocr import load  # noqa: E402
from indra_ocr.pipeline import OrtOCR  # noqa: E402
from indra_ocr.textlayer import read_text_layer  # noqa: E402


def page_dict(page) -> dict:
    return {
        "source": page.source,
        "rotation": page.rotation,
        "skew": page.skew,
        "note": page.note,
        "lines": [asdict(ln) for ln in page.lines],
        "markdown": page.markdown,
        "blocks": page.blocks,
    }


def tok(t):
    return [t[0] if isinstance(t[0], str) else int(t[0]), float(t[1])] + ([int(t[2])] if len(t) > 2 else [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--docs", type=Path, default=ROOT / "tests/data/documents")
    ap.add_argument("--out", type=Path, default=ROOT / "tests/data/reference/pipeline")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ocr = OrtOCR(a.models, threads=4)
    for name in ("page_two_column", "page_rotated", "page_skewed", "page_chinese", "page_math"):
        gray = cv2.imread(str(a.docs / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        k = ocr.orientation(gray)
        up, k2, skew = ocr.upright(gray)
        boxes = ocr.detect(up)
        rec = ocr.recognize(up, boxes)
        page = ocr(gray)
        out = {
            "orientation": k,
            "skew": skew,
            "rec": [{"text": r[0], "conf": r[1], "tokens": [tok(t) for t in r[2]]} for r in rec],
            "page": page_dict(page),
        }
        np.savez_compressed(a.out / f"{name}.npz", upright=up, boxes=boxes.astype(np.float32))
        (a.out / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(name, "k", k, "skew", round(skew, 2), "lines", len(page.lines))

    math = OrtOCR(a.models, threads=4, math=True)
    gray = cv2.imread(str(a.docs / "page_math.png"), cv2.IMREAD_GRAYSCALE)
    page = math(gray)
    (a.out / "page_math.math.json").write_text(json.dumps({"page": page_dict(page)}, ensure_ascii=False))
    print("math:", [ln.text for ln in page.lines])

    for name in ("digital", "scanned"):
        path = a.docs / f"{name}.pdf"
        doc = pdfium.PdfDocument(str(path))
        renders, layers = {}, []
        for i in range(len(doc)):
            pg = doc[i]
            renders[f"render{i}"] = pg.render(scale=200 / 72, grayscale=True).to_numpy()
            low = pg.render(scale=1.0, grayscale=True).to_numpy()
            tl = read_text_layer(pg, low, 1.0)
            layers.append({
                "trusted": tl.trusted, "reason": tl.reason, "stats": tl.stats,
                "runs": [{"bbox": list(r.bbox), "text": r.text} for r in tl.lines],
            })
        doc.close()
        np.savez_compressed(a.out / f"{name}.npz", **renders)
        result = {"layers": layers}
        for mode in ("auto", "ocr"):
            conv = load(a.models, threads=4, mode=mode)
            result[mode] = [page_dict(p) for _, p in conv.convert(path)]
        (a.out / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False))
        print(name, [p["source"] for p in result["auto"]], [p["note"] for p in result["auto"]])


if __name__ == "__main__":
    main()
