# NanoOCR

[![NuGet](https://img.shields.io/nuget/v/NanoOCR.svg?label=NanoOCR)](https://www.nuget.org/packages/NanoOCR/) [![Downloads](https://img.shields.io/nuget/dt/NanoOCR.svg?label=downloads)](https://www.nuget.org/packages/NanoOCR/) [![NuGet](https://img.shields.io/nuget/v/NanoOCR.Cli.svg?label=NanoOCR.Cli)](https://www.nuget.org/packages/NanoOCR.Cli/)

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

The original Python implementation is kept unchanged in [`.reference/`](https://github.com/curiosity-ai/nano-ocr/tree/main/.reference) as the
normative specification.

## Install

| Package | | |
| --- | --- | --- |
| [`NanoOCR`](https://www.nuget.org/packages/NanoOCR/) | [![NuGet](https://img.shields.io/nuget/v/NanoOCR.svg?label=)](https://www.nuget.org/packages/NanoOCR/) | the library, models embedded: `dotnet add package NanoOCR` |
| [`NanoOCR.Cli`](https://www.nuget.org/packages/NanoOCR.Cli/) | [![NuGet](https://img.shields.io/nuget/v/NanoOCR.Cli.svg?label=)](https://www.nuget.org/packages/NanoOCR.Cli/) | the `nano-ocr` command: `dotnet tool install -g NanoOCR.Cli` |

.NET 10, Windows, Linux and macOS (PDFium and SkiaSharp native assets for all three come with the
package). Versions are CalVer (`yy.M.build`), published from `main` by
[`.devops/azure-pipelines.yml`](https://github.com/curiosity-ai/nano-ocr/blob/main/.devops/azure-pipelines.yml).

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
| `Markdown` | `MarkdownOptions.Default` | the markdown flavour and the document-level passes (see [Markdown](#markdown)) |

Everything is configured in code (or, for `nano-ocr`, with command-line arguments: `nano-ocr --help`);
nothing reads environment variables.

Stage-level access is public too: `NanoOcrEngine` (the port of `OrtOCR`: `Detect`, `Recognize`,
`Upright`, `Assemble`, …), `DocumentConverter`, `NanoOcr.Onnx.InferenceSession` (run any of the
models, or another ONNX file using the same operators), and the OpenCV ports in `NanoOcr.Imaging`.

## Command line

`src/NanoOcr.Cli` builds `nano-ocr` (on NuGet as the `NanoOCR.Cli` tool), with the Python `indra-ocr` command's options:

```shell
nano-ocr paper.pdf                       # markdown to stdout
nano-ocr *.pdf --out results/            # one .md per input
nano-ocr scan.png --format json          # lines, boxes, layout blocks, confidences, timings
nano-ocr paper.pdf --mode ocr --timings  # ignore the text layer; timings on stderr
nano-ocr paper.pdf --math                # write math as LaTeX
nano-ocr scan.png --no-orientation       # the page is known to be upright
nano-ocr scan.png --no-fusion --gemm-256 # graphs as written, 256-bit kernels (A/B; same output)
nano-ocr scan.png --save-img out/        # draw blocks and lines over the page
nano-ocr paper.pdf --merge-tables --title-levels   # rejoin split tables, heading depth from the document
nano-ocr paper.pdf --md-style reference  # the Python original's markdown, byte for byte
nano-ocr classify *.pdf                  # each page's category and signals (see Routing pages)
nano-ocr bench page.png                  # warm stage timings, per-operator profile
```

## Output

Each page (`OcrResult.Page`) has `Lines` (box, text, confidence, in reading order), `Blocks`
(`Text`, `Section-Header`, `Page-Header`, `Page-Footer`, `List-Group`, `Table`, `Caption`,
`Footnote`, `Table-Of-Contents`, `Form`), `Markdown` (headings, paragraphs, lists, tables as HTML),
`Source` (`ocr`, `text`, `hybrid` or `error`), `Rotation`, `Skew`, `Timings` and `Note` — the
fields of the Python `Page`, and `ToJson()` writes the Python CLI's JSON layout.

### Markdown

By default the markdown is written the way PaddleOCR's document parser writes it (PaddleX's
`MarkdownConverter`, as [PaddleOcrSharp](https://github.com/curiosity-ai/PaddleOCR/tree/main/dotnet)
reproduces it), so the output of the two engines can be consumed the same way:

| | PaddleOCR style (default) | `--md-style reference` (the Python original) |
| --- | --- | --- |
| document title | `# Title` — the opening heading of the first page, when its text is ≥1.25× every other heading's | `## Title` |
| headings | depth from the numbering: `## Results`, `### 1. Introduction`, `#### 1.2.3 Detail` | `## ` always |
| tables | one line of HTML, every cell `<td>`, border and centring styles, after an empty line | `<table>`, one `<tr>` per line, header row in `<th>` |
| lists | items separated by a blank line | one item per line |
| captions | centred in a `<div>` | plain paragraph |

On the test corpus (`page_two_column.png`, `scanned.pdf`, `digital.pdf`, …) the default output is
byte-identical to PaddleOCR-VL 1.6's wherever the two read the same text. Two things stay NanoOCR's
own because they are content rather than format: inline math is `$…$` (PaddleOCR pads it to
`␣$ … $`, which not every renderer reads as math), and what each model reads.

`MarkdownOptions` (and the CLI flags) also carry PaddleOCR's document-level passes, ported onto
NanoOCR's layout blocks:

| Option | CLI | |
| --- | --- | --- |
| `Pretty` = false | `--plain` | PaddleOCR's plain variant: unstyled tables, captions as text |
| `MergeTables` | `--merge-tables` | rejoin a table split by a page break (`MergeTablesAcrossPages`): the last table of a page and the first of the next, with only page furniture or footnotes between, widths within 10%, and the same columns; a repeated header is dropped |
| `TitleLevels` | `--title-levels` | heading depth from the whole document (`AssignTitleLevels`): the numbering's depth, the numbering style's first appearance and the text size ranked among all headings, two of three deciding; section names (Abstract, References, …) are level one |
| `DocTitle` = false | `--no-doc-title` | never write the opening heading as the title |

With `MergeTables` or `TitleLevels`, `NanoOCR.Predict` returns a file's pages once its last page is
done. A caller driving `NanoOcrEngine` or `DocumentConverter` directly calls
`MarkdownDocument.Finish(pages, options)` on a document's pages for the same result.

## Routing pages

`PageRouter<TOutput>` reads each page with NanoOCR, sorts it into a `PageCategory` —
`PrintedText`, `Handwritten`, `TableOrComplex` or `Mixed` — and hands it to the recognizer
registered for that category, falling back to a default. It is generic on the output, so
NanoOCR's own reading and another engine (PaddleOCR-VL, say) can share it through a conversion of
their results:

```csharp
using var nano = new NanoOCR(new NanoOcrOptions { KeepImages = true });
var router = new PageRouter<string>(nano)
    .When(PageCategory.PrintedText, page => page.Ocr.Markdown)          // NanoOCR's reading is the answer
    .When(c => c.Signals.Tables > 2, page => VlMarkdown(page))          // predicates refine; first match wins
    .When(PageCategory.Handwritten, page => VlMarkdown(page))
    .Otherwise(page => VlMarkdown(page));                               // TableOrComplex, Mixed, …

foreach (RoutedResult<string> r in router.Route("scans/batch.pdf"))
    Console.WriteLine($"page {r.Page.PageIndex + 1}: {r.Page.Category} ({r.Page.Classification.Reason})");
```

A recognizer gets a `RoutedPage`: NanoOCR's `OcrResult` (the page image on it when `KeepImages`
is on), the input path and page index (or the bytes, for `Route(byte[])`), and the
classification. Asynchronous recognizers (`(page, ct) => ValueTask<T>`) go through the same
`When`/`Otherwise`, with `RouteAsync` returning an `IAsyncEnumerable`. Routes are tried in the
order they were added; with no `Otherwise`, a page no route takes throws.

The default classifier (`PageClassifier`) needs no model of its own — it decides from the reading
NanoOCR already made, so classifying costs nothing more than reading the page (~0.1-0.2 s):

| Category | Rule (every figure relative to the page, so independent of resolution) |
| --- | --- |
| `Mixed` | fewer than 3 lines, or a failed page |
| `Handwritten` | most lines unread (≥45% below 0.6 confidence, mean < 0.75) — the recognizer is trained on print — **and** written large (median line ≥1.2% of the page height) or in long strokes (≥8% of lines wider than a quarter of the page); never a PDF text layer |
| `Mixed` | mostly unread, in small type: degraded print, a fax, noise |
| `TableOrComplex` | ≥25% of the lines in the order model's `Table` or `Form` blocks |
| `Mixed` | printed, but mean confidence < 0.85 or ≥35% of lines unread |
| `PrintedText` | everything else |

It was calibrated on 160 RVL-CDIP pages and checked on 400 others (fax-quality 1990s scans, the
hard case). On the 400 it finds 80% of the handwritten pages, and 56% of its handwriting calls are
right (large-type adverts make up most of the rest); 83% of its printed calls are printed. **It
never called a handwritten page printed** — the one error that would hand a page NanoOCR cannot
read to NanoOCR. The thresholds are `PageClassifierOptions`; a trained classifier plugs in through
`new PageRouter<T>(nano, result => …)`, and `nano-ocr classify` prints every signal per page for
tuning on your own documents.

## Validation

Three layers of checks, each against the original implementation running in the same container
(Python 3.13, onnxruntime 1.29, OpenCV 5.0, pypdfium2 5.13). The fixtures are generated by the
scripts in [`tools/reference`](https://github.com/curiosity-ai/nano-ocr/blob/main/tools/reference/README.md); they are hundreds of MB, so they are
produced on demand rather than committed, and the tests that need them skip when they are absent.

| Check | Scope | Result |
| --- | --- | --- |
| `LayerParityTests` | every node output of all 10 models (1,857 tensors) vs ONNX Runtime | all within tolerance (worst at 7% of it); CTC ids and lengths exact |
| `ImagingParityTests` | resize (linear, area, fast area), warpAffine 8-bit and float, connected components, PNG gray decode, TIFF in 7 compressions vs OpenCV | **byte-exact**, 54 cases |
| `PipelineParityTests` | orientation, deskewed page, boxes, per-line text/confidence/tokens, final pages; PDF renders and text layers; auto and OCR modes; math | upright pages byte-identical, boxes within 1e-4 px, every line and token identical, **markdown identical** on every page (in the reference style) |
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
| one page, 4 threads | 277-327 ms | 106-128 ms |
| allocation per warm page | — | 8 MiB, no GC |

The C# figures are two to three times as fast as this port was before its optimisation passes
(452-510 ms on one thread, 297-326 on four, same sitting). Single-threaded: 512-bit GEMM kernels, graph fusion (activation,
residual, squeeze-excite gate and sequence mask folded into the convolutions; the CTC head's
softmax peak fused with its convolution), rewritten depthwise and pooling kernels, a deskew search
that skips what it can prove is blank, vectorized OpenCV ports, and pooled buffers — with every
model output, image byte and markdown line unchanged. Multi-threaded, the recognizer's width
batches run concurrently with thread shares proportional to their size, and the deskew estimate
overlaps the orientation model. `nano-ocr bench page.png --threads 1` prints
a machine header (CPU, ISA, FMA and bandwidth ceilings, jitter) before its stage timings. Details,
and the experiments that did not pay, are in [`CLAUDE.md`](https://github.com/curiosity-ai/nano-ocr/blob/main/CLAUDE.md).

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
src/NanoOcr.Cli/         nano-ocr (NuGet: NanoOCR.Cli, a .NET tool)
tests/NanoOcr.Tests/     parity and unit tests
tools/reference/         Python scripts that produce the reference fixtures
.devops/                 the Azure DevOps pipeline that publishes the packages
.reference/              the original Python package, unchanged
```

## License

The code is Apache 2.0 ([`LICENSE`](https://github.com/curiosity-ai/nano-ocr/blob/main/LICENSE)). The model weights are Datalab's, under their modified
OpenRAIL-M license ([`MODEL_LICENSE`](https://github.com/curiosity-ai/nano-ocr/blob/main/src/NanoOcr/Models/Assets/MODEL_LICENSE)): free for research, personal use
and startups under $20M in funding or revenue, and not for use competing with Datalab's API.
