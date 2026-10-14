# Page-type model

`src/NanoOcr/Models/Assets/pagetype.onnx` tells four kinds of page apart from the page image alone,
so that `PageRouter` can send each page to the engine that reads it best:

| Class | What it is | Where a router typically sends it |
| --- | --- | --- |
| `printed` | paragraphs, headings, lists, columns — scanned or rendered | NanoOCR's own reading |
| `table` | dominated by tables or forms: invoices, budgets, specifications, questionnaires | a layout-aware VLM (PaddleOCR-VL) |
| `handwritten` | mostly handwriting: notes, letters, filled-in pages | a VLM (PaddleOCR-VL) |
| `photo` | a camera photo: a scene, a sign, a receipt, a page photographed rather than scanned | a scene-text reader (Florence-2) |

It is NanoOCR's own, not part of Indra OCR: MobileNetV3-Small (timm's
`mobilenetv3_small_100.lamb_in1k`, ImageNet weights, Apache-2.0) with a one-channel stem and a
four-way head, fine-tuned end to end. 1.5M parameters, 6 MB.

## Input

`pagetype_input.model_input` — and `NanoOcr.Pipeline.PageTypeModel.Input`, which must stay
identical: the upright grayscale page, long side scaled to 512 with OpenCV's `INTER_AREA` (which
NanoOCR's `CvResize` reproduces to the byte), placed top-left on a white 512×512 square, as
`gray / 255`. The graph normalizes internally and ends in a softmax, so it returns probabilities
in `printed, table, handwritten, photo` order. Gray rather than colour because that is the image
the NanoOCR pipeline holds; 512 rather than 224 so that text strokes survive the downscale.

## Data

The weights ship inside a commercial-friendly library, so every **training** source is
permissively licensed. 3,800 pages, about 12% held out for validation:

| Class | Source | Licence | Pages |
| --- | --- | --- | --- |
| printed | DocLayNet v1.2 (pages with < 5% table and < 15% picture area) | CDLA-Permissive-1.0 | 350 |
| printed | PubLayNet | CDLA-Permissive-1.0 | 300 |
| table | DocLayNet v1.2 (pages with > 35% table area) | CDLA-Permissive-1.0 | 350 |
| table | FATURA2 invoices | CC-BY-4.0 | 200 |
| table | katanaml invoices | MIT | 150 |
| handwritten | NoTeS-Bank (ICDAR 2025 handwritten notes) | Apache-2.0 | 400 |
| handwritten | GNHK (GoodNotes Handwriting Kollection), train split | CC-BY-4.0 | 300 |
| photo | COCO-Text | CC-BY-4.0 | 400 |
| photo | Total-Text, train images | BSD-3-Clause | 200 |
| photo | CORD v2 receipts, train split | CC-BY-4.0 | 150 |
| printed | rendered letters, memos, e-mails, resumes, reports, slides (`synthesize.py`) | generated | 600 |
| table | rendered budgets, forms, questionnaires, specification sheets (`synthesize.py`) | generated | 400 |

The rendered pages are degraded like a fax, a photocopy or a noisy scan two times in three.
Training augments towards what real inputs look like and the clean sources do not: scanner and
fax degradation (downscale, binarize, noise, JPEG), contrast, blur, a residual tilt and loose
crops.

The **evaluation** set (`build_eval.py`) shares nothing with training and is scored only: 325
RVL-CDIP pages (real 1990s scans, 100 per class, mapped onto the four classes), 20 handwritten
pages composed from IAM test lines, and 240 photos (CORD test, Total-Text test, COCO-Text val).
RVL-CDIP and IAM carry licences that would not allow training on them; scoring is all they are
used for.

## Results

The held-out set, through ONNX Runtime (`evaluate.py`; the model alone, argmax):

| Pages | Right | Mostly confused with |
| --- | --- | --- |
| photos: COCO-Text val, Total-Text test, CORD test (240) | **100%** | — |
| handwritten: IAM pages, RVL-CDIP handwritten (45) | **98%** | table (1) |
| RVL-CDIP printed: letters, memos, e-mail, reports, articles, resumes, slides (200) | 64% | table (62) |
| RVL-CDIP tables: forms, invoices, budgets, specifications, questionnaires (125) | 70% | handwritten (26) — RVL forms are often filled in by hand |
| **all (610)** | **81.6%** | |

Through the router (`nano-ocr classify`, `score_classify.py`), where a verdict below 0.4 is
`Mixed` and a page that looks printed but that NanoOCR read poorly goes to the fallback: 83.1% of
the pages reach the engine of their class (printed → NanoOCR; table, handwritten, Mixed → a VLM;
photo → Florence-2), against 42.5% for the reading rules alone; no handwritten page goes to
NanoOCR (the rules sent 8 of 45), and every photo is recognised (the rules cannot).

The rendered office pages (`synthesize.py`) are what lifted the printed scans: without them the
model had seen print only as dense scientific and financial pages, and called 98 of the 200
memos, letters and e-mails tables (74.3% overall). The 0.4 threshold was chosen on these same
pages (0.6 left 23% of them to the fallback for 80.0% routed right), so the router's 83.1% is
slightly optimistic; the model-alone figures above involve no such choice.

## Reproducing

CPU only. The PyTorch CPU wheels come from conda-forge (`micromamba create -c conda-forge
python=3.12 "pytorch=*=cpu*" torchvision timm onnx onnxruntime py-opencv pillow pyarrow`).

```shell
python build_dataset.py data/          # ~2.8k pages, a few GB downloaded, ~300 MB kept
python synthesize.py data/             # 1k rendered office pages
python build_eval.py eval/             # the held-out pages
python train.py --data data/ --out out/ --epochs 8
python evaluate.py --model out/pagetype.onnx --eval eval/
cp out/pagetype.onnx ../../src/NanoOcr/Models/Assets/
```

Then regenerate the layer fixtures (`tools/reference/dump_layers.py`) so `LayerParityTests`
checks the new graph against ONNX Runtime, and score the router end to end:
`nano-ocr classify $(cat eval/files.txt)` from `eval/`, then `python score_classify.py`.
