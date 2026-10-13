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

`dump_surya_detection.py` runs Surya's text-line detector (`surya/detection`) from a checkout of
Surya with the checkpoint at `https://models.datalab.to/text_detection/2025_05_07/`, and needs the
OpenCV Surya pins (4.11.0.86) in a virtual environment of its own; its docstring has the setup. It
writes `tests/data/reference/surya_detection/` (per document the chunks, both resizes, the
processor output, the model's maps, the stitched heatmaps, the thresholds and the boxes; the model's
stages on one chunk), checked by `SuryaDetectionParityTests` with the checkpoint in
`SURYA_DET_MODEL_DIR` or `tests/data/models/surya_det`.

`tests/data/reference/` is git-ignored (a few hundred MB). The parity tests skip, rather than
fail, when it is absent.
