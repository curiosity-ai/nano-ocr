# NanoOCR — Pure C# Port of Indra OCR

This repository is **NanoOCR, a pure C# re-implementation of Indra OCR** (the Python package `indra-ocr`
0.3.1, Datalab). The original Python sources live unchanged in [`.reference/`](.reference/) and
are the **normative specification: when C# and Python disagree, Python is right.** The original
PyPI publish workflow is disabled (`.github/workflows/publish.yml.disabled`); do not re-enable it.

The port owns its tensor math, ONNX graph interpreter, image operations and PDF text layer. **It
does not use ONNX Runtime, OpenCV, or any other inference or vision library.** SkiaSharp decodes
images; PDFium (native, via P/Invoke — the engine pypdfium2 wraps) renders PDFs.

## Layout

| Path | Port of |
| --- | --- |
| `src/NanoOcr/Onnx/` | ONNX Runtime: `ProtoReader` + `OnnxModel` (ModelProto, no protobuf runtime), `InferenceSession` (slot-compiled, constant-folded, fused graph), `Operators` (≈55 ops, plus the fused `ConvSoftmaxPeak` / `SoftmaxPeak` in `Fused.cs`) |
| `src/NanoOcr/Core/` | MLAS's role: `Gemm` (packed 8×32 512-bit micro-kernel where AVX-512 is accelerated, 6×16 `Vector<T>` otherwise; segmented batched products with a strip epilogue), `Parallelism` |
| `src/NanoOcr/Imaging/` | OpenCV: `CvResize` (`resize` linear/area), `CvWarp` (`warpAffine`, `getRotationMatrix2D`), `ConnectedComponents`; `ImageIO` (SkiaSharp + libpng's gray formula), `TiffDecoder` + `Ccitt` (`imreadmulti`) |
| `src/NanoOcr/Pdf/` | pypdfium2: `Pdfium` (P/Invoke), `PdfDocument`, `PdfPage.Render` |
| `src/NanoOcr/Pipeline/Detection.cs` | `det_post.py` |
| `src/NanoOcr/Pipeline/LineCrops.cs` | `rec_preproc.py`, `line_mask.crop_lines` |
| `src/NanoOcr/Pipeline/Vocab.cs` | `vocab.py` (the inference parts) |
| `src/NanoOcr/Pipeline/ReadingOrder.cs` | `order_post.py` |
| `src/NanoOcr/Pipeline/Layout.cs` | `layout_post.py` |
| `src/NanoOcr/Pipeline/MathReader.cs` | `math.py` |
| `src/NanoOcr/Pipeline/TextLayer.cs` | `textlayer.py` |
| `src/NanoOcr/Pipeline/DocumentConverter.cs` | `document.py` |
| `src/NanoOcr/Pipeline/NanoOcrEngine.cs` | `pipeline.py` (`OrtOCR`, deskew, orientation) |
| `src/NanoOcr/Pipeline/Py.cs` | Python/numpy semantics (see below) |
| `src/NanoOcr/NanoOCR.cs`, `OcrResult.cs` | `__init__.load`, `parallel.convert_many`; the PaddleOCR-style API |
| `src/NanoOcr.Cli/` | `cli.py` (`indra-ocr`; here `nano-ocr`), plus `bench` and `bench-gemm` |
| `src/NanoOcr/Models/Assets/` | the `v3.1` models of `datalab-to/indra-ocr`, embedded as resources (hashes in its README) |

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
- **Fusion must not change arithmetic.** The session's fusion pass (`InferenceSession.Fuse`)
  folds element-wise nodes into the convolution that produces or consumes them; each fused step
  is the same operation, in the same order, rounded once, so `FusedGraphEqualsUnfused` can demand
  bit equality (the softmax-peak sum is the documented exception). Shapes a fused path does not
  cover fall back to the unfused operation at run time. A new fusion needs that test green.
- **Wider kernels must keep each output's FMA chain.** The 512-bit GEMM and depthwise kernels
  produce the 256-bit kernels' bits because every output is one in-order chain over k; a kernel
  that splits a reduction across lanes changes results and needs its own justification.
- **Pooled buffers are not zeroed.** Everything from `ArrayPool<T>.Shared` (tensors, scratch,
  `GrayImage.Rent`) starts with garbage; a kernel that reads before writing must clear first (the
  INTER_AREA row sum and the summed-area table's first row are the examples in the code).
- Read the Python for a stage before writing the C#, and name the Python function in the doc
  comment.

## Validation

See the README's *Validation* section and [`tools/reference/README.md`](tools/reference/README.md).
Any change to a kernel, an image operation or a pipeline stage must keep all of
`LayerParityTests`, `ImagingParityTests`, `PipelineParityTests` and `UnitTests` green with the
fixtures generated; a change that moves an output must say why.

## Building and testing

```bash
dotnet build NanoOcr.slnx -c Release
dotnet test tests/NanoOcr.Tests            # or: dotnet run -c Release --project tests/NanoOcr.Tests
```

## Performance record

Method: [`measuring-performance`](https://github.com/theolivenbaum/PaddleOCR/tree/main/.claude/skills/measuring-performance)
(the sibling port's skill) — warm runs, best of N with the spread, interleaved A/B, an untouched
stage as control, outputs pinned by the parity tests.

`nano-ocr bench page.png [--threads N] [--runs N] [--warmup N]` starts with a **machine header**
(CPU model, ISA, single-thread FMA rate at 256 and 512 bits, all-thread FMA, L1/L2/DRAM read
bandwidth, jitter — `--no-machine` skips it). Two runs are only compared when their headers
agree; a jitter above 5% means the host is busy. It then takes `--warmup` untimed pages (tiering,
pools), times `--runs` pages **without** the operator profile (its per-node bookkeeping would be
inside the figure), prints each stage's best, median and spread, allocation and GC pauses per
page, then sub-stages one at a time with their time and allocation (thumbnail, quarter turn,
skew, orientation, upright, detect, crops, script, recognize, assemble), and finally a separate
profiled pass per model (`BENCH_MODELS=rec` picks the models, `BENCH_TOP` the rows, `BENCH_MATH=1`
the math reader). `nano-ocr bench-gemm` prints GEMM throughput at the models' shapes.

Two switches exist for A/B runs: `NANOOCR_GEMM_256=1` (the 256-bit GEMM kernels instead of the
512-bit ones) and `NANOOCR_NO_FUSION=1` (run graphs as written). For an A/B against older code,
build it in a `git worktree` with the current `Bench.cs`/`MachineProfile.cs` copied in and
alternate the two binaries.

The per-operator profile **overstates small, fused operators** (a stopwatch per node, and it
runs in its own pass): it put `ConvSoftmaxPeak` at 26-29 ms where instrumentation inside the
kernel measured 16. Read it for shares within a model, and decide with whole-stage A/B runs.

### Where it stands

Measured 2026-10 on a 4-vCPU shared VM — Intel Xeon @ 2.10 GHz (Sapphire Rapids class, AVX-512,
`Vector<float>` 256 bits but `Vector512` accelerated), single-thread FMA 77-90 GF/s at 256 bits
and 165 at 512, DRAM ~11 GB/s, jitter 12-50% — 1700×2200 page (`page_two_column.png`), best of
7 warm runs, the three columns interleaved in one sitting:

| | C# before this work | C# now | Python + ORT 1.31 |
| --- | --- | --- | --- |
| page, 1 thread | 452-510 ms | **214-230 ms** | 281-290 ms |
| page, 4 threads | 297-326 ms | **157-168 ms** | 277-327 ms |
| allocation per warm page | 37 MiB, GC every page | **8 MiB, no GC** | — |

Stage by stage, 1 thread: orientation + deskew 98-102 → 41-44 ms, detection 112-116 → 54-57,
recognition 240-250 → 113-121 (script model 36 → 14), assembly/order 5.6 → 2.6. The other test
pages, 1 thread: skewed 519-534 → 226 ms (it is the one that rotates the full page), rotated
488-491 → 228-250, Chinese 203-207 → 86-88, math 170-171 → 76-79, and the math page with the
math reader (`BENCH_MATH=1`) 561-574 → 294-318. Every parity test passes throughout: model
outputs within the ONNX Runtime tolerance (margins unchanged), every image operation byte-exact
against OpenCV, every line, token and markdown identical. The CLI's markdown for all 24 files of
`tests/data/documents` (images, PDFs, TIFFs, with and without `--math`) is byte-identical between
the build before this work and now.

### What moved it

In the order they landed; each is its own commit with its measurement.

- **512-bit GEMM kernels.** `Vector<T>` stays 256 bits on AVX-512 hardware, so every product ran
  at half the FMA width the machine has. `Gemm.Wide` (= `Vector512.IsHardwareAccelerated`)
  selects an 8×32 register tile (16 accumulators) with 4-, 2- and 1-row tails; each output is
  still one in-order FMA chain, so results equal the 256-bit path's bit for bit. Rec shapes
  1.6-1.9× (1678×1742×128: 69 → 118 GF/s); page 467-481 → 395-402 ms.
- **Graph fusion** (`InferenceSession.Fuse`, ONNX Runtime's ConvActivation/ConvAdd fusions and
  more): `Conv [→ Add(residual)] [→ HardSwish|Relu|Sigmoid] [→ Mul(mask)]` and `Mul(x, sigmoid(…))
  → Conv` (a squeeze-excite gate, applied per channel while the GEMM packs its input) become one
  convolution whose epilogue runs while the output is in cache; the CTC head's `Conv → Transpose
  → ReduceMax / ArgMax / Sub → Exp → ReduceSum` becomes `ConvSoftmaxPeak`, which computes the
  1678-class logits 64 time steps at a time into an L2-resident tile and reduces them there
  (26 MB a page never reach memory). A run with an observer executes the unfused graph, so the
  layer parity tests still see every node; `FusedGraphEqualsUnfused` pins the fused outputs to the
  unfused ones bit for bit, except the softmax sum, whose order changes (within the layer tolerance).
- **Depthwise convolution**: it called `TensorPrimitives.MultiplyAdd` once per row per tap and was
  scalar at stride 2. Now: a zero-padded, stride-phase-split scratch plane, four independent FMA
  chains per output row, an overlapping last vector instead of a scalar tail, and the same path
  for one-channel stems with several filters (bands of rows, all filters per band).
- **Deskew**: a summed-area table of nonzero ink lets warp chunks (and whole rows) that can only
  read zeros be written as zeros without computing them — exactly what computing them gives; ~80%
  of chunks. INTER_AREA's horizontal pass accumulated through memory entry by entry; it is now a
  padded `[taps, dw]` table evaluated eight outputs at a time with AVX2 gathers, and its byte
  store is vectorized (5 → 3.9 ms per thumbnail, two a page). The full-page rotation of a skewed
  page (`WarpReplicate`) gained the same AVX2 gather path: 59 → 14 ms. All byte-exact.
- **Batched pointwise convolutions in place**: `Gemm.MultiplyBatched` treats `[B, C, T]` as one
  product over `B·T` columns whose packing reads each item where it lies (two 512-bit vectors a
  strip row) and whose strips that straddle two items store through a scratch tile; bias,
  residual, HardSwish and mask are applied per finished 32-column strip (`StripEpilogue`). No
  copy into `[C, B·T]` and back, and no epilogue pass: recognition 172-183 → 115 ms over the steps.
- **The recognizer's 32×4 stem** matched the multi-filter depthwise test after that change and ran
  128-tap chains; it now goes to a batched im2col + one segmented GEMM: ~13 → 2.3 ms.
- **Allocation**: `Broadcast.RunPlan` walks broadcast runs by index instead of building a list of
  one tuple per channel row (17 MiB a page); thumbnails, ink maps, the nonzero table, detector
  and orientation inputs and line crops come from `ArrayPool<T>.Shared` (`CvResize.ResizeInto`,
  `GrayImage.Rent`). 37 → 8 MiB a page, no GC in warm runs.
- **Smaller kernels**: vectorized 2×2 pooling (script MaxPool 9 → ~1 ms), im2col that copies the
  in-bounds part of a padded row instead of falling back to a checked scalar loop (script model
  17 → 14 ms), `Softplus` in vector blocks, `Erf` by Abramowitz-Stegun 7.1.26 in float lanes
  (order model 5 → 2.6 ms), numpy's pairwise sum as one `Vector256` (its eight accumulators are
  the lanes, bit-identical), vectorized nearest-2× `Resize`, stride-2 transposes, `DepthToSpace`
  and HardSwish on 512-bit lanes, a blocked quarter turn.

Earlier work, before the rename (each also with outputs unchanged): the deskew angles in
parallel with each row reduced as it is warped (206 → 32 ms at 4 threads); the packed 6×16 GEMM
micro-kernel (now the non-AVX-512 path); the math decoder's slot-compiled graph and constant
folding.

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
  (noise band 18-25 ms either way). They are bound by memory traffic that ORT pays too. (They are
  now fused into the convolutions instead.)
- **A cheaper `exp` in the softmax peak** (Cephes polynomial instead of `Vector256.Exp`): 9.4 →
  9.6 ms, no change. The operator was bound by streaming 26 MB of logits back from memory; folding
  the head convolution in so they never get there was the fix. The cheap exp stayed, since the
  fused kernel is compute-bound.
- **Bias-initialized GEMM strips with the epilogue per strip** for the detector's and orientation
  model's convolutions (instead of filling each output tile with the bias, accumulating, then a
  pass per row). Interleaved: detection 54-57 ms off against 58-65 on. The tiles were already
  L2-resident, and per-32-column fills and epilogue calls cost more than full-row passes. Kept
  only for the batched pointwise path, where the rows are 134 long and the pass was a real one.
- **Memoizing the deskew fine search by angle** (the fine grid revisits coarse angles): 18.7 →
  19.0 ms. `arange` rarely produces the same double twice. Reverted.
- **Banding the multi-filter stem depthwise** (all sixteen filters over a band of rows before the
  next): neutral (5.5 → 5.1 ms). The stem is bound by writing its 13 MB output, not by reading its
  input. Kept as harmless.
- **A cache-blocked quarter turn** with pointer writes: 5.9 → 5.1 ms — mostly the 3.7 MB result's
  allocation and first touch. Kept, small.

### Where the remaining time is

From a `perf` profile of a single-thread run (`DOTNET_PerfMapEnabled=1
DOTNET_EnableWriteXorExecute=0 perf record -e cpu-clock`, `apt install linux-tools-generic`): the
8×32 GEMM kernel is ~25% of all samples, depthwise rows ~7%, the first write of convolution
outputs (`Fill`) ~4%, INTER_AREA ~6%, packing ~4%, the deskew's gathers and chunk tests ~5%.

- **Recognition (~115 ms)**: pointwise GEMMs run at ~90 GF/s in the pipeline against ~135 for the
  same shape in isolation — strips split across items, packing, the gate multiply; the fused CTC
  head ~16 ms (12 of them its GEMM); the script model ~14 ms; line crops ~5 ms.
- **Detection (~55 ms)** is now largely memory-bound: depthwise layers spend 8 ms a page in the
  first read of their input (the padding copy) and the network writes ~100 MB of activations a
  page. The next step is depth-first fusion of each inverted-residual block (expand → depthwise →
  project per band of rows), which keeps those intermediates in cache.
- **Orientation + deskew (~42 ms)**: two INTER_AREA thumbnails (~4 ms each), the orientation model
  (~18 ms, the same structure as the detector) and the 32 warps of the deskew search (~14 ms).
