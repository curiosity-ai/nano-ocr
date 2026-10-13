"""Dump OpenCV's outputs for the image operations the pipeline uses.

    python tools/reference/dump_imaging.py --docs tests/data/documents --out tests/data/reference/imaging

Writes ``imaging.npz`` and ``imaging.json``; ImagingParityTests checks the C# ports
(IndraOcr.Imaging.CvResize / CvWarp / ConnectedComponents / ImageIO) against them.
Each case is a named input, an operation with its arguments, and OpenCV's output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=ROOT / "tests/data/documents")
    ap.add_argument("--out", type=Path, default=ROOT / "tests/data/reference/imaging")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    page = cv2.imread(str(a.docs / "page_two_column.png"), cv2.IMREAD_GRAYSCALE)
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, (97, 131), dtype=np.uint8)
    crop = page[300:345, 150:790].copy()
    arrays: dict[str, np.ndarray] = {"in.page": page, "in.noise": noise, "in.crop": crop}
    cases = []

    def add(name, op, src, args, out):
        arrays[f"out.{name}"] = np.ascontiguousarray(out)
        cases.append({"name": name, "op": op, "input": src, "args": args})

    inputs = {"page": page, "noise": noise, "crop": crop}
    sizes = {
        "page": [(1024 * 1700 // 2200, 1024), (850, 1100), (566, 733), (2000, 2588), (777, 1005), (1700 // 3, 2200 // 3)],
        "noise": [(65, 48), (262, 194), (131, 50), (40, 97), (300, 220), (43, 32)],
        "crop": [(455, 32), (911, 64), (320, 22), (1280, 90)],
    }
    for src, img in inputs.items():
        for w, h in sizes[src]:
            for interp, iname in ((cv2.INTER_LINEAR, "linear"), (cv2.INTER_AREA, "area")):
                add(f"resize_{src}_{w}x{h}_{iname}", "resize", src, {"w": w, "h": h, "interp": iname},
                    cv2.resize(img, (w, h), interpolation=interp))

    # Rotation (deskew) on 8 bits with replicated borders, and on a float ink map with zero borders.
    for deg in (1.7, -2.3, 0.4, 4.9):
        h, w = page.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        add(f"warp_page_{deg}", "warp_replicate", "page", {"deg": deg},
            cv2.warpAffine(page, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        small = cv2.resize(page, (618, 800), interpolation=cv2.INTER_AREA)
        ink = (255.0 - small.astype(np.float32)) / 255.0
        arrays["in.ink"] = ink
        M = cv2.getRotationMatrix2D((618 / 2, 800 / 2), deg, 1.0)
        add(f"warp_ink_{deg}", "warp_zero", "ink", {"deg": deg}, cv2.warpAffine(ink, M, (618, 800)))

    # Connected components of a thresholded map.
    mask = (cv2.GaussianBlur(noise, (5, 5), 0) > 140).astype(np.uint8)
    arrays["in.mask"] = mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    arrays["out.cc_labels"] = lab.astype(np.int64)
    arrays["out.cc_stats"] = stats.astype(np.int64)
    cases.append({"name": "cc", "op": "cc", "input": "mask", "args": {"n": int(n)}})

    # Grayscale decoding of a colour PNG (libpng's rgb_to_gray).
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(a.out / "color.png"), rgb)
    add("decode_color_png", "decode", "color.png", {}, cv2.imread(str(a.out / "color.png"), cv2.IMREAD_GRAYSCALE))

    # TIFF pages (cv2.imreadmulti), every compression the corpus holds.
    tiffs = sorted((a.docs / "tiff").glob("*.tif")) + [a.docs / "inputs" / "three.tiff"]
    for f in tiffs:
        ok, frames = cv2.imreadmulti(str(f), flags=cv2.IMREAD_GRAYSCALE)
        name = "tiff_" + f.name
        for i, fr in enumerate(frames):
            arrays[f"out.{name}.{i}"] = fr
        cases.append({"name": name, "op": "tiff", "input": str(f.relative_to(ROOT)), "args": {"frames": len(frames)}})

    np.savez_compressed(a.out / "imaging.npz", **arrays)
    (a.out / "imaging.json").write_text(json.dumps(cases, indent=1))
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
