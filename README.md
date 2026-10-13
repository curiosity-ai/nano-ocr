# Indra OCR for .NET

A pure C# port of [Indra OCR](https://github.com/datalab-to/indra) (Datalab's 1M-parameter CPU
OCR pipeline): line detection, CTC line recognition, reading order, layout classes and markdown,
plus the PDF text layer, Chinese routing and the optional LaTeX math reader.

- **No ONNX Runtime, no OpenCV, no Python.** The ten Indra models run on this library's own ONNX
  interpreter and SIMD kernels (`System.Numerics.Tensors`, `Vector<T>`, AVX2 gathers); the image
  operations the pipeline needs are bit-exact ports of OpenCV's.
- **Models embedded.** The `v3.1` model set (det, rec, rec_cjk, script, order, orient and the four
  math models, 38 MB) is compiled into `IndraOcr.dll` — no download, no model folder.
- **Validated layer by layer.** Every node output of every model matches ONNX Runtime, every image
  operation matches OpenCV byte for byte, and every pipeline stage matches the Python reference
  — down to identical markdown on the test corpus (see [Validation](#validation)).
- SkiaSharp decodes images; PDFium (the same engine pypdfium2 wraps) renders PDFs and reads their
  text layer; multi-page TIFF (including CCITT G3/G4 fax) is decoded in-library.

The original Python implementation is kept unchanged in [`.reference/`](.reference/) as the
normative specification.

## Quick start

```csharp
using IndraOcr;

using var ocr = new IndraOCR();                 // embedded models, all cores
foreach (OcrResult res in ocr.Predict("paper.pdf"))
{
    res.Print();                                 // lines with boxes and confidences
    res.SaveToMarkdown("output");                // output/paper_0.md
    res.SaveToJson("output");                    // output/paper_0.json
    Console.WriteLine(res.Markdown);
}
```

The API follows PaddleOCR's: `Predict` returns one result per page, and each result exposes the
PaddleOCR fields alongside Indra's own.

| PaddleOCR | Indra OCR for .NET |
| --- | --- |
| `PaddleOCR(...)` | `new IndraOCR(new IndraOcrOptions { ... })` |
| `ocr.predict(input)` | `ocr.Predict(path)` — a file or a directory; also `Predict(byte[])`, `Predict(SKBitmap)`, `Predict(GrayImage)` |
| `ocr.ocr(img)` (2.x) | `ocr.Ocr(path)` — per page, `(corner points, (text, score))` |
| `res["rec_texts"]`, `rec_scores`, `rec_boxes`, `dt_polys` | `res.RecTexts`, `RecScores`, `RecBoxes`, `DtPolys` |
| `res.print()`, `save_to_json`, `save_to_markdown`, `save_to_img` | `res.Print()`, `SaveToJson`, `SaveToMarkdown`, `SaveToImg` |
| `use_doc_orientation_classify` | `IndraOcrOptions.UseDocOrientationClassify` |
| — | `res.Markdown`, `res.Blocks` (labelled layout blocks in reading order), `res.Page` (everything) |

Many files on many cores (the counterpart of Python's `convert_many`): pages run one per core,
each single-threaded, results in input order.

```csharp
foreach (var (path, pages, error) in ocr.PredictMany(files, cores: 8))
{
    if (error is not null) { Console.Error.WriteLine(error.Message); continue; }
    foreach (OcrResult page in pages!) { /* ... */ }
}
```

### Options

`IndraOcrOptions` carries the arguments of Python's `indra_ocr.load`:

| Option | Default | Meaning |
| --- | --- | --- |
| `Mode` | `Auto` | `Auto`: the PDF text layer where it passes the checks, OCR elsewhere. `Ocr`: never read the layer. `Text`: use it even when the checks fail |
| `Dpi` | 200 | PDF render resolution (pages over 100 MP render lower) |
| `PdfOrder` | true | keep the PDF content order when it is coherent |
| `ImageOcr` | true | on text-layer pages, OCR text inside images |
| `Furniture` | false | keep page headers and footers in the markdown |
| `Math` | false | read math as LaTeX (`$...$`); slower on math pages |
| `Threads` | all cores | kernel threads for one page |
| `ModelDir` | embedded | a folder laid out like the Hugging Face repository instead (`INDRA_OCR_MODEL_DIR` also works) |
| `KeepImages` | false | keep page images on the results (for `SaveToImg`) |

Stage-level access is public too: `IndraEngine` (the port of `OrtOCR`: `Detect`, `Recognize`,
`Upright`, `Assemble`, …), `DocumentConverter`, `IndraOcr.Onnx.InferenceSession` (run any of the
models, or another ONNX file using the same operators), and the OpenCV ports in `IndraOcr.Imaging`.

## Command line

`src/IndraOcr.Cli` builds `indra-ocr`, with the Python command's options:

```shell
indra-ocr paper.pdf                       # markdown to stdout
indra-ocr *.pdf --out results/            # one .md per input
indra-ocr scan.png --format json          # lines, boxes, layout blocks, confidences, timings
indra-ocr paper.pdf --mode ocr --timings  # ignore the text layer; timings on stderr
indra-ocr paper.pdf --math                # write math as LaTeX
indra-ocr scan.png --save-img out/        # draw blocks and lines over the page
indra-ocr bench page.png                  # warm stage timings, per-operator profile
```

## Chandra 2 and Surya 2

`src/IndraOcr.Vlm` (NuGet `IndraOCR.Vlm`) ports Datalab's vision-language OCR to pure C#:
[Chandra 2](https://github.com/datalab-to/chandra) (`datalab-to/chandra-ocr-2`, 5.3B parameters:
page → layout-aware HTML, markdown and figure crops) and
[Surya 2](https://github.com/datalab-to/surya) (`datalab-to/surya-ocr-2`, 0.69B: layout, full-page
and block OCR, table rows/columns), plus Surya's non-VLM models — the text-line detector, the
RF-DETR fast layout with its reading-order head, and the OCR-error classifier. Both VLMs are
Qwen3.5 (a ViT plus a hybrid gated-delta-net / gated-attention decoder), run by one engine with
bf16 weights and float32 arithmetic. No libtorch, no vLLM, no ONNX Runtime, no Python.

The checkpoints are too large to embed (1.4 GB and 10.6 GB): they are downloaded from the
Hugging Face Hub (or Datalab's model store) on first use into `~/.cache/indra-ocr`, or read from a
folder you pass.

```csharp
using IndraOcr.Vlm.Chandra;
using IndraOcr.Vlm.Surya;

using var chandra = ChandraOcr.Load();                         // or Load("/models/chandra-ocr-2")
List<ChandraPage> pages = chandra.PredictFile("paper.pdf");    // markdown, html, chunks, figures
ChandraOcr.SaveMergedOutput("out", "paper.pdf", pages);        // out/paper/paper.md, .html, _metadata.json

using var surya = SuryaInference.Load();
using RgbImage page = RgbImageIO.Read("scan.png");
LayoutResult layout = new SuryaLayoutPredictor(surya).Predict([page])[0];
PageOcrResult ocr = new SuryaRecognitionPredictor(surya).Predict([page], fullPage: true)[0];
TableResult table = new SuryaTableRecPredictor(surya).PredictSimple([page])[0];
```

```shell
indra-vlm chandra paper.pdf out/                     # chandra INPUT OUTPUT, same flags as the Python CLI
indra-vlm surya ocr scan.png --output-dir out/       # surya_ocr  -> out/scan/results.json
indra-vlm surya layout page.pdf                      # surya_layout
indra-vlm surya table page.pdf --mode full           # surya_table
indra-vlm surya fast-layout page.png                 # RF-DETR layout + reading order
indra-vlm surya detect page.png                      # text lines
```

What is checked against the originals (the reference scripts are in
[`tools/reference`](tools/reference/README.md)):

| Check | Result |
| --- | --- |
| `VlmParityTests` — preprocessing, every vision block, every decoder layer, greedy decoding vs transformers, both checkpoints | preprocessing **byte-exact**; layers within 1.8e-4 (Surya) and 4.7e-4 (Chandra) of the 1e-3 tolerance; 96 and 64 greedy tokens identical |
| `HtmlParityTests` — 356 HTML strings (model-shaped, malformed, seeded fuzz) through BeautifulSoup, markdownify 1.1.0, Chandra's and Surya's parsers | every string identical |
| `SuryaVlmParityTests` — Surya's own layout, full-page OCR, block OCR and table predictors | raw outputs and parsed results identical |
| `ChandraPipelineTests` — Chandra's own `load_file` → generate → parse | raw output, markdown, HTML, chunks identical |
| `SuryaLayoutParityTests`, `SuryaDetectionParityTests`, `SuryaOcrErrorParityTests` | per-stage tensors within tolerance; boxes, labels, order and classifications identical |

On CPU these are slow models: on the 4-core test VM a Surya 2 block decodes at ~70 ms/token and
Chandra 2 at ~0.6 s/token (both bound by reading the weights), and a page's vision tower and
prefill take tens of seconds to minutes. Pages and blocks are decoded in batches, which reads the
weights once per step for the whole batch.

## Output

Each page (`OcrResult.Page`) has `Lines` (box, text, confidence, in reading order), `Blocks`
(`Text`, `Section-Header`, `Page-Header`, `Page-Footer`, `List-Group`, `Table`, `Caption`,
`Footnote`, `Table-Of-Contents`, `Form`), `Markdown` (headings, paragraphs, lists, tables as HTML),
`Source` (`ocr`, `text`, `hybrid` or `error`), `Rotation`, `Skew`, `Timings` and `Note` — the
fields of the Python `Page`, and `ToJson()` writes the Python CLI's JSON layout.

## Validation

Three layers of checks, each against the original implementation running in the same container
(Python 3.13, onnxruntime 1.29, OpenCV 5.0, pypdfium2 5.13). The fixtures are generated by the
scripts in [`tools/reference`](tools/reference/README.md); they are hundreds of MB, so they are
produced on demand rather than committed, and the tests that need them skip when they are absent.

| Check | Scope | Result |
| --- | --- | --- |
| `LayerParityTests` | every node output of all 10 models (1,857 tensors) vs ONNX Runtime | all within tolerance (worst at 7% of it); CTC ids and lengths exact |
| `ImagingParityTests` | resize (linear, area, fast area), warpAffine 8-bit and float, connected components, PNG gray decode, TIFF in 7 compressions vs OpenCV | **byte-exact**, 54 cases |
| `PipelineParityTests` | orientation, deskewed page, boxes, per-line text/confidence/tokens, final pages; PDF renders and text layers; auto and OCR modes; math | upright pages byte-identical, boxes within 1e-4 px, every line and token identical, **markdown identical** on every page |
| `UnitTests` | the Python package's own unit tests, ported | pass |

```shell
python tools/reference/make_test_documents.py tests/data/documents       # the corpus (committed)
python tools/reference/dump_layers.py   --models /path/to/indra-ocr        # per-layer fixtures
python tools/reference/dump_imaging.py                                     # OpenCV fixtures
python tools/reference/dump_pipeline.py --models /path/to/indra-ocr        # stage fixtures
dotnet test tests/IndraOcr.Tests
```

## Performance

On a 4-core shared VM (Xeon @ 2.8 GHz, AVX2), the 1700×2200 test page, warm, best of 5-7,
measured interleaved with the Python pipeline in the same sitting (spreads of 15-40% between
samples are normal on this machine):

| | Python + ONNX Runtime | C# |
| --- | --- | --- |
| one page, 4 threads | 302-310 ms | 334-379 ms |
| one page, 1 thread | 397-407 ms | 577-587 ms |
| 12-page scan, `--cores 4`, process start included | 2.6-2.9 s | 3.6-3.8 s |
| math page (`--math`), 4 threads | 483-523 ms | 409-535 ms |

Detection runs at parity with ONNX Runtime; the gap is in the recognizer's CTC head (memory-bound
element-wise passes over 1,678-class logits) and the deskew search, and a ~0.5 s longer cold
start (JIT). Details, and the experiments that did not pay, are in [`CLAUDE.md`](CLAUDE.md).

## Repository layout

```
src/IndraOcr/            the library (NuGet: IndraOCR)
  Onnx/                  protobuf reader, InferenceSession, operator kernels
  Core/                  GEMM, parallelism
  Imaging/               OpenCV ports, SkiaSharp IO, TIFF/CCITT
  Pdf/                   PDFium bindings
  Pipeline/              the Indra pipeline (one file per Python module)
  Models/Assets/         the embedded models (OpenRAIL-M, see MODEL_LICENSE)
  IndraOCR.cs            the PaddleOCR-style API
src/IndraOcr.Cli/        indra-ocr
src/IndraOcr.Vlm/        Chandra 2 and Surya 2 (NuGet: IndraOCR.Vlm)
src/IndraOcr.Vlm.Cli/    indra-vlm
tests/IndraOcr.Tests/    parity and unit tests
tools/reference/         Python scripts that produce the reference fixtures
.reference/              the original Python package, unchanged
```

## License

The code is Apache 2.0 (`LICENSE`). The model weights are Datalab's, under their modified
OpenRAIL-M license (`src/IndraOcr/Models/Assets/MODEL_LICENSE`): free for research, personal use
and startups under $20M in funding or revenue, and not for use competing with Datalab's API.
