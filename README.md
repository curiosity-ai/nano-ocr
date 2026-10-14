# NanoOCR

NanoOCR is a pure C# port of [Indra OCR](https://github.com/datalab-to/indra) (Datalab's 1M-parameter CPU
OCR pipeline): line detection, CTC line recognition, reading order, layout classes and markdown,
plus the PDF text layer, Chinese routing and the optional LaTeX math reader.

- **No ONNX Runtime, no OpenCV, no Python.** The ten Indra models run on this library's own ONNX
  interpreter and SIMD kernels (`System.Numerics.Tensors`, `Vector<T>`, AVX2 gathers); the image
  operations the pipeline needs are bit-exact ports of OpenCV's.
- **Models embedded.** The `v3.1` model set (det, rec, rec_cjk, script, order, orient and the four
  math models, 38 MB) is compiled into `NanoOcr.dll` — no download, no model folder.
- **Validated layer by layer.** Every node output of every model matches ONNX Runtime, every image
  operation matches OpenCV byte for byte, and every pipeline stage matches the Python reference
  — down to identical markdown on the test corpus (see [Validation](#validation)).
- SkiaSharp decodes images; PDFium (the same engine pypdfium2 wraps) renders PDFs and reads their
  text layer; multi-page TIFF (including CCITT G3/G4 fax) is decoded in-library.

The original Python implementation is kept unchanged in [`.reference/`](.reference/) as the
normative specification.

## Quick start

```csharp
using NanoOcr;

using var ocr = new NanoOCR();                 // embedded models, all cores
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

| PaddleOCR | NanoOCR |
| --- | --- |
| `PaddleOCR(...)` | `new NanoOCR(new NanoOcrOptions { ... })` |
| `ocr.predict(input)` | `ocr.Predict(path)` — a file or a directory; also `Predict(byte[])`, `Predict(SKBitmap)`, `Predict(GrayImage)` |
| `ocr.ocr(img)` (2.x) | `ocr.Ocr(path)` — per page, `(corner points, (text, score))` |
| `res["rec_texts"]`, `rec_scores`, `rec_boxes`, `dt_polys` | `res.RecTexts`, `RecScores`, `RecBoxes`, `DtPolys` |
| `res.print()`, `save_to_json`, `save_to_markdown`, `save_to_img` | `res.Print()`, `SaveToJson`, `SaveToMarkdown`, `SaveToImg` |
| `use_doc_orientation_classify` | `NanoOcrOptions.UseDocOrientationClassify` |
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

`NanoOcrOptions` carries the arguments of Python's `indra_ocr.load`:

| Option | Default | Meaning |
| --- | --- | --- |
| `Mode` | `Auto` | `Auto`: the PDF text layer where it passes the checks, OCR elsewhere. `Ocr`: never read the layer. `Text`: use it even when the checks fail |
| `Dpi` | 200 | PDF render resolution (pages over 100 MP render lower) |
| `PdfOrder` | true | keep the PDF content order when it is coherent |
| `ImageOcr` | true | on text-layer pages, OCR text inside images |
| `Furniture` | false | keep page headers and footers in the markdown |
| `Math` | false | read math as LaTeX (`$...$`); slower on math pages |
| `Threads` | all cores | kernel threads for one page |
| `ModelDir` | embedded | a folder laid out like the Hugging Face repository instead |
| `KeepImages` | false | keep page images on the results (for `SaveToImg`) |
| `Fusion` | true | run the models with element-wise nodes fused into the convolutions; false runs the graphs as written (same results, slower) |

Everything is configured in code (or, for `nano-ocr`, with command-line arguments: `nano-ocr --help`);
nothing reads environment variables.

Stage-level access is public too: `NanoOcrEngine` (the port of `OrtOCR`: `Detect`, `Recognize`,
`Upright`, `Assemble`, …), `DocumentConverter`, `NanoOcr.Onnx.InferenceSession` (run any of the
models, or another ONNX file using the same operators), and the OpenCV ports in `NanoOcr.Imaging`.

## Command line

`src/NanoOcr.Cli` builds `nano-ocr`, with the Python `indra-ocr` command's options:

```shell
nano-ocr paper.pdf                       # markdown to stdout
nano-ocr *.pdf --out results/            # one .md per input
nano-ocr scan.png --format json          # lines, boxes, layout blocks, confidences, timings
nano-ocr paper.pdf --mode ocr --timings  # ignore the text layer; timings on stderr
nano-ocr paper.pdf --math                # write math as LaTeX
nano-ocr scan.png --no-orientation       # the page is known to be upright
nano-ocr scan.png --no-fusion --gemm-256 # graphs as written, 256-bit kernels (A/B; same output)
nano-ocr scan.png --save-img out/        # draw blocks and lines over the page
nano-ocr bench page.png                  # warm stage timings, per-operator profile
```

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
dotnet test tests/NanoOcr.Tests
```

## Performance

On a 4-vCPU shared VM (Intel Xeon @ 2.10 GHz, AVX-512), the 1700×2200 test page, warm, best of 7,
measured interleaved with the Python pipeline in the same sitting (spreads of 10-40% between
samples are normal on this machine):

| | Python + ONNX Runtime 1.31 | C# |
| --- | --- | --- |
| one page, 1 thread | 281-290 ms | 214-230 ms |
| one page, 4 threads | 277-327 ms | 157-168 ms |
| allocation per warm page | — | 8 MiB, no GC |

The C# figures are about twice as fast as this port was before its optimisation pass (452-510 ms
on one thread, 297-326 on four, same sitting): 512-bit GEMM kernels, graph fusion (activation,
residual, squeeze-excite gate and sequence mask folded into the convolutions; the CTC head's
softmax peak fused with its convolution), rewritten depthwise and pooling kernels, a deskew search
that skips what it can prove is blank, vectorized OpenCV ports, and pooled buffers — with every
model output, image byte and markdown line unchanged. `nano-ocr bench page.png --threads 1` prints
a machine header (CPU, ISA, FMA and bandwidth ceilings, jitter) before its stage timings. Details,
and the experiments that did not pay, are in [`CLAUDE.md`](CLAUDE.md).

## Repository layout

```
src/NanoOcr/             the library (NuGet: NanoOCR)
  Onnx/                  protobuf reader, InferenceSession, operator kernels
  Core/                  GEMM, parallelism
  Imaging/               OpenCV ports, SkiaSharp IO, TIFF/CCITT
  Pdf/                   PDFium bindings
  Pipeline/              the Indra pipeline (one file per Python module)
  Models/Assets/         the embedded models (OpenRAIL-M, see MODEL_LICENSE)
  NanoOCR.cs             the PaddleOCR-style API
src/NanoOcr.Cli/         nano-ocr
tests/NanoOcr.Tests/     parity and unit tests
tools/reference/         Python scripts that produce the reference fixtures
.reference/              the original Python package, unchanged
```

## License

The code is Apache 2.0 (`LICENSE`). The model weights are Datalab's, under their modified
OpenRAIL-M license (`src/NanoOcr/Models/Assets/MODEL_LICENSE`): free for research, personal use
and startups under $20M in funding or revenue, and not for use competing with Datalab's API.
