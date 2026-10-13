# Reference fixtures

Python scripts that run the **original** pipeline (in `.reference/`) and the libraries it is
built on, and dump what they compute, so the C# port can be checked against them. They need
`numpy onnx onnxruntime opencv-python-headless pypdfium2 pillow reportlab pikepdf` and the
model folder (`huggingface-cli download datalab-to/indra-ocr --revision v3.1 --local-dir models`).

| Script | Writes | Checked by |
| --- | --- | --- |
| `make_test_documents.py [dir]` | the test corpus: rendered pages (two-column, rotated, skewed, Chinese, math), a digital and a scanned PDF, TIFFs in every compression, and input-handling files. Deterministic; committed under `tests/data/documents` | everything below |
| `dump_layers.py --models M` | `tests/data/reference/layers/<model>.npz/.json`: every node output of each model on real inputs, from ONNX Runtime | `LayerParityTests` |
| `dump_imaging.py` | `tests/data/reference/imaging/`: OpenCV's resize, warpAffine, connected components, gray decoding and TIFF pages | `ImagingParityTests` |
| `dump_pipeline.py --models M` | `tests/data/reference/pipeline/`: per document the orientation, skew, upright page, boxes, per-line recognition and final page; for PDFs the renders, text layers and pages in auto and OCR mode; the math page with `--math` | `PipelineParityTests` |
| `dump_surya_layout.py PAGE...` | `tests/data/reference/surya_layout/<page>.npz/.json`: Surya's fast layout (RF-DETR) and its reading-order head, run through the vendored `surya.common.rfdetr` code — preprocessing, every backbone layer, the projector, the two-stage proposals, every decoder layer, the heads, post-processing, the order head's logits and the final `LayoutResult`. Needs torch, torchvision, transformers, the `datalab-to/surya_layout2` folder (`--model`, also `SURYA_LAYOUT_MODEL_DIR` or `tests/data/models/surya_layout2` for the tests) and a Surya checkout (`--surya`) | `SuryaLayoutParityTests` |
| `dump_vlm.py --model M --image I` | `tests/data/reference/vlm/<name>/vlm.npz`: a Qwen3.5 checkpoint (Surya 2, Chandra 2) through transformers — the resizes, pixel values, every vision block, every decoder layer, two layers' internals, greedy logits. `--dtype bf16-weights` for Chandra (float32 compute without a float32 copy) | `VlmParityTests` |
| `dump_html.py --chandra C --surya S` | `tests/data/reference/html/html.json`: BeautifulSoup, markdownify 1.1.0 and the two packages' parsers on 356 HTML strings | `HtmlParityTests` |
| `dump_surya_vlm.py --surya S --model M` | `tests/data/reference/surya_vlm/surya_vlm.json`: Surya's predictors with transformers in place of the vLLM server | `SuryaVlmParityTests` |
| `dump_chandra_pipeline.py --chandra C --model M --image I` | `tests/data/reference/chandra_pipeline/chandra.json`: Chandra's whole hf pipeline on one image | `ChandraPipelineTests` |
| `dump_surya_ocr_error.py --surya S --model M` | `tests/data/reference/surya_ocr_error/ocr_error.json`: the OCR-error classifier's ids, logits, labels and scores | `SuryaOcrErrorParityTests` |

`tests/data/reference/` is git-ignored (a few hundred MB). The parity tests skip, rather than
fail, when it is absent.
