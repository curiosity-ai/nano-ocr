# Indra OCR — Pure C# Port

This repository is a **pure C# re-implementation of Indra OCR** (the Python package `indra-ocr`
0.3.1, Datalab). The original Python sources live unchanged in [`.reference/`](.reference/) and
are the **normative specification: when C# and Python disagree, Python is right.** The original
PyPI publish workflow is disabled (`.github/workflows/publish.yml.disabled`); do not re-enable it.

The port owns its tensor math, ONNX graph interpreter, image operations and PDF text layer. **It
does not use ONNX Runtime, OpenCV, or any other inference or vision library.** SkiaSharp decodes
images; PDFium (native, via P/Invoke — the engine pypdfium2 wraps) renders PDFs.

## Layout

| Path | Port of |
| --- | --- |
| `src/IndraOcr/Onnx/` | ONNX Runtime: `ProtoReader` + `OnnxModel` (ModelProto, no protobuf runtime), `InferenceSession` (slot-compiled, constant-folded graph), `Operators` (≈55 ops) |
| `src/IndraOcr/Core/` | MLAS's role: `Gemm` (packed 6×16 `Vector<T>` micro-kernel), `Parallelism` |
| `src/IndraOcr/Imaging/` | OpenCV: `CvResize` (`resize` linear/area), `CvWarp` (`warpAffine`, `getRotationMatrix2D`), `ConnectedComponents`; `ImageIO` (SkiaSharp + libpng's gray formula), `TiffDecoder` + `Ccitt` (`imreadmulti`) |
| `src/IndraOcr/Pdf/` | pypdfium2: `Pdfium` (P/Invoke), `PdfDocument`, `PdfPage.Render` |
| `src/IndraOcr/Pipeline/Detection.cs` | `det_post.py` |
| `src/IndraOcr/Pipeline/LineCrops.cs` | `rec_preproc.py`, `line_mask.crop_lines` |
| `src/IndraOcr/Pipeline/Vocab.cs` | `vocab.py` (the inference parts) |
| `src/IndraOcr/Pipeline/ReadingOrder.cs` | `order_post.py` |
| `src/IndraOcr/Pipeline/Layout.cs` | `layout_post.py` |
| `src/IndraOcr/Pipeline/MathReader.cs` | `math.py` |
| `src/IndraOcr/Pipeline/TextLayer.cs` | `textlayer.py` |
| `src/IndraOcr/Pipeline/DocumentConverter.cs` | `document.py` |
| `src/IndraOcr/Pipeline/IndraEngine.cs` | `pipeline.py` (`OrtOCR`, deskew, orientation) |
| `src/IndraOcr/Pipeline/Py.cs` | Python/numpy semantics (see below) |
| `src/IndraOcr/IndraOCR.cs`, `OcrResult.cs` | `__init__.load`, `parallel.convert_many`; the PaddleOCR-style API |
| `src/IndraOcr.Cli/` | `cli.py` (`indra-ocr`), plus `bench` and `bench-gemm` |
| `src/IndraOcr/Models/Assets/` | the `v3.1` models of `datalab-to/indra-ocr`, embedded as resources (hashes in its README) |

Not ported: `viewer.py` (a local web viewer), `bench/opb_indra.py` (OmniParseBench harness), the
CenterMask path of `line_mask.py` (the v3.1 model set ships no `mask.onnx`), and the training-only
helpers (`encode_targets`, vocabulary builders, label canonicalization).

## Engineering conventions

- **.NET 10, C# preview.** `System.Numerics.Tensors` (`TensorPrimitives`), `Vector<T>`, x86
  intrinsics where they pay (AVX2 gathers in the deskew warp), `ArrayPool<T>.Shared` for every
  tensor buffer. Native-AOT compatible: no reflection, JSON through a source-generated context.
- **Reproduce numpy's float32 semantics exactly** where they decide an output. `Py.cs` holds the
  rules the port needed: numpy's pairwise float32 summation (`sum`, `mean`), `np.median` of
  float32, Python's `round(x, n)` on the exact binary value, banker's rounding, code-point string
  predicates (`isdigit`, `isalpha`, `strip`). And **NEP 50**: a Python float meeting a float32
  array is narrowed to float32 — in arithmetic *and in comparisons* (`boxes[:, 0] <= cx` compares
  in float32). Several call sites cast for that reason; keep the casts.
- **Bit-exact OpenCV.** The resizes use OpenCV's fixed-point arithmetic (11-bit coefficients,
  `((b0·(S0≫4))≫16 + (b1·(S1≫4))≫16 + 2)≫2`), the area tables, and the 2× area fast path. The
  warp follows OpenCV 5's float-coordinate kernel: the row term `y·M1 + M2` is a plain
  multiply-add and the per-pixel step a fused one (anything else moves ~0.006% of pixels by a
  level). `ImagingParityTests` pins all of it to the byte.
- **Model inference matches ONNX Runtime within float tolerance**, not bit-exactly (summation
  order differs); the pipeline's outputs are nevertheless identical on the corpus because the
  decisions (thresholds, argmaxes) are far from their boundaries.
- Read the Python for a stage before writing the C#, and name the Python function in the doc
  comment.

## Validation

See the README's *Validation* section and [`tools/reference/README.md`](tools/reference/README.md).
Any change to a kernel, an image operation or a pipeline stage must keep all of
`LayerParityTests`, `ImagingParityTests`, `PipelineParityTests` and `UnitTests` green with the
fixtures generated; a change that moves an output must say why.

## Building and testing

```bash
dotnet build IndraOcr.slnx -c Release
dotnet test tests/IndraOcr.Tests            # or: dotnet run -c Release --project tests/IndraOcr.Tests
```

## Performance record

Method: [`measuring-performance`](https://github.com/theolivenbaum/PaddleOCR/tree/main/.claude/skills/measuring-performance)
(the sibling port's skill) — warm runs, best of N with the spread, interleaved A/B against Python,
an untouched stage as control, outputs pinned by the parity tests. `indra-ocr bench page.png
[--threads N]` prints stage timings, the spread, cold start, allocation, GC pauses and a
per-operator profile (`BENCH_MODELS=rec` picks the models, `BENCH_MATH=1` the math reader);
`indra-ocr bench-gemm` prints GEMM throughput at the models' shapes.

Measured on a 4-core shared VM (Xeon @ 2.8 GHz, AVX2), 1700×2200 page, against the Python
pipeline in the same sitting:

| | Python + ORT | C# |
| --- | --- | --- |
| page, 4 threads (best of 7) | 302-310 ms | 334-379 ms |
| page, 1 thread | 397-407 ms | 577-587 ms |
| 12 pages, `--cores 4`, cold process | 2.6-2.9 s | 3.6-3.8 s |
| math page, 4 threads | 483-523 ms | 409-535 ms |
| allocation per warm page | — | 39 MiB, GC pause 1.5 ms |

Where it stands, stage by stage (4 threads): orientation + deskew 62-70 ms (Python 103), detection
62-68 (Python 54-61, the control throughout), recognition 194-233 (Python 153), order ~10 (7).

What moved it, each with outputs unchanged:

- **Deskew search, 206 → 32 ms.** The 32 candidate warps were serial and each produced a full
  image before summing its rows. Angles now run in parallel, each row is reduced as it is warped,
  and an AVX2 gather path does the interior with the scalar path's exact arithmetic.
- **GEMM micro-kernel.** B packed into contiguous 2-vector strips and a 6×16 register tile (12
  accumulators): 512³ 66 → 80 GFLOP/s on one thread, 191 → 242 on four; the recognizer's
  256×134×128 29 → 73. Each output is still one in-order FMA chain, so results are unchanged.
- **Batched 1×1 convolutions** over the recognizer's short sequences run as one packed GEMM
  (`[C, B·T]`) instead of B narrow ones with scalar tails.
- **CTC head**: batched parallel transpose, SIMD argmax, parallel suffix reductions.
- **Math decoder**: the session compiles the graph to integer slots (it was hashing names on every
  node of a 600-node graph run per token), folds constant subgraphs at load, pads 2-3-row products
  into the 4-row kernel so the weight is streamed once, keeps sub-1M-MAC products on one thread,
  and evaluates erf four lanes at a time.

### Things that looked like wins and were not

- **ReadyToRun for the CLI.** One page 1.49-1.52 s against 1.63-1.75 s JIT, but twelve pages
  4.05-4.40 s against 3.70-4.01 s, interleaved: the precompiled code targets a baseline ISA and the
  hot kernels re-JIT anyway. Kept: JIT with `TieredCompilation=false` (tiering on was worse still:
  4.6-5.1 s).
- **A pointer-based scalar warp** (no bounds checks) instead of the AVX2 gathers: 222-235 ms
  against 97-123 ms for the gathers, single-threaded, interleaved — no better than the checked
  scalar loop (218-221 ms). The scalar per-pixel cost is in the float→int conversions and
  floors, not the loads.
- **The first packed GEMM kernel** kept its edge-tile handling (`stackalloc` scratch) inside the
  hot method: 512³ fell from 66 to 20 GFLOP/s, because the JIT stopped enregistering the twelve
  accumulators. Moving the edge case into a separate non-inlined method restored and passed the
  original (80).
- **Parallelizing every broadcast `Mul`.** The recognizer's `[B,C,T]·[B,C,1]` multiplies run at
  ~14 GB/s, which is this VM's memory bandwidth; splitting them across threads measured no better
  (noise band 18-25 ms either way). They are bound by memory traffic that ORT pays too.

### Where the remaining time is

Recognition's operators, single-threaded: pointwise convolutions ~55%, the CTC head's
transpose/exp/argmax/reductions over 1,678-class logits ~20% (memory-bound), broadcast multiplies
~10%. Closing the rest needs operator fusion (softmax-peak and argmax in one pass over the logits,
activation fused into the convolution epilogue) rather than faster kernels.
