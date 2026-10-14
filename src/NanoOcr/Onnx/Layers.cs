using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using NanoOcr.Core;

namespace NanoOcr.Onnx;

internal static partial class Operators
{
    internal static Tensor Conv(OnnxNode node, List<Tensor?> x)
    {
        Tensor input = x[0]!, weight = x[1]!;
        Tensor? bias = x.Count > 2 ? x[2] : null;
        Tensor? residual = x.Count > 3 ? x[3] : null;
        Tensor? gate = x.Count > 4 ? x[4] : null;
        Tensor? mask = x.Count > 5 ? x[5] : null;
        string act = node.Str(FusedActivation, "");
        if (input.Rank == 3)
        {
            // A 1-D convolution is a 2-D one over a height of 1.
            Tensor in2 = input.Reshape([input.Shape[0], input.Shape[1], 1, input.Shape[2]]);
            Tensor w2 = weight.Reshape([weight.Shape[0], weight.Shape[1], 1, weight.Shape[2]]);
            var attrs = new OnnxNode { OpType = "Conv", Name = node.Name };
            attrs.Attributes["group"] = new OnnxAttribute { Name = "group", I = node.Int("group", 1) };
            if (act.Length > 0)
            {
                attrs.Attributes[FusedActivation] = new OnnxAttribute { Name = FusedActivation, S = act };
            }
            long[] k = node.Ints("strides") ?? [1];
            long[] d = node.Ints("dilations") ?? [1];
            long[] pd = node.Ints("pads") ?? [0, 0];
            var a = new OnnxAttribute { Name = "strides" };
            a.Ints.AddRange([1, k[0]]);
            attrs.Attributes["strides"] = a;
            a = new OnnxAttribute { Name = "dilations" };
            a.Ints.AddRange([1, d[0]]);
            attrs.Attributes["dilations"] = a;
            a = new OnnxAttribute { Name = "pads" };
            a.Ints.AddRange([0, pd[0], 0, pd[1]]);
            attrs.Attributes["pads"] = a;
            // The fused inputs (residual, gate, mask) follow the input into 2-D.
            var args = new List<Tensor?>(x.Count) { in2, w2, bias };
            for (int i = 3; i < x.Count; i++)
            {
                args.Add(x[i]?.Rank == 3 ? x[i]!.Reshape([x[i]!.Shape[0], x[i]!.Shape[1], 1, x[i]!.Shape[2]]) : x[i]);
            }

            Tensor out2 = Conv(attrs, args);
            Tensor out1 = out2.Reshape([out2.Shape[0], out2.Shape[1], out2.Shape[3]]);
            in2.Release();
            w2.Release();
            out2.Release();
            for (int i = 3; i < x.Count; i++)
            {
                if (args[i] is not null && !ReferenceEquals(args[i], x[i]))
                {
                    args[i]!.Release();
                }
            }

            return out1;
        }

        if (input.Rank != 4)
        {
            throw new NotSupportedException($"Rank-{input.Rank} convolutions are not implemented.");
        }

        int n = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        int m = weight.Shape[0], cg = weight.Shape[1], kh = weight.Shape[2], kw = weight.Shape[3];
        int group = (int)node.Int("group", 1);
        long[] strides = node.Ints("strides") ?? [1, 1];
        long[] dil = node.Ints("dilations") ?? [1, 1];
        long[] pads = node.Ints("pads") ?? [0, 0, 0, 0];
        string autoPad = node.Str("auto_pad", "NOTSET");
        int sh = (int)strides[0], sw = (int)strides[1], dh = (int)dil[0], dw = (int)dil[1];
        int pt = (int)pads[0], pl = (int)pads[1], pb = (int)pads[2], pr = (int)pads[3];
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            int oh0 = (h + sh - 1) / sh, ow0 = (w + sw - 1) / sw;
            int padH = Math.Max(0, ((oh0 - 1) * sh) + ((kh - 1) * dh) + 1 - h);
            int padW = Math.Max(0, ((ow0 - 1) * sw) + ((kw - 1) * dw) + 1 - w);
            pt = autoPad == "SAME_UPPER" ? padH / 2 : padH - (padH / 2);
            pb = padH - pt;
            pl = autoPad == "SAME_UPPER" ? padW / 2 : padW - (padW / 2);
            pr = padW - pl;
        }

        int oh = ((h + pt + pb - (dh * (kh - 1)) - 1) / sh) + 1;
        int ow = ((w + pl + pr - (dw * (kw - 1)) - 1) / sw) + 1;
        Tensor output = Tensor.Float([n, m, oh, ow]);
        var p = new ConvParams(c, h, w, m, cg, kh, kw, sh, sw, dh, dw, pt, pl, oh, ow, group);
        float[]? b = bias?.FloatArray;

        // Whatever has a shape the fused paths do not cover is applied as the unfused node
        // would have, before or after: a broadcast residual, a gate that is not [N, C, 1, 1] or
        // whose convolution does not take the batched pointwise path, a mask that is not
        // [N, 1, Oh, Ow] (or that would follow an unfused step or the whole-tensor sigmoid).
        bool fuseResidual = residual is not null && residual.Type == ElementType.Float && residual.Shape.AsSpan().SequenceEqual(output.Shape);
        bool fuseGate = gate is not null && gate.Type == ElementType.Float && gate.Shape.AsSpan().SequenceEqual([n, c, 1, 1])
            && UsesPointwiseBatched(p, n);
        Activation activation = ParseActivation(residual is null || fuseResidual ? act : "");
        bool fuseMask = mask is not null && mask.Type == ElementType.Float && mask.Shape.AsSpan().SequenceEqual([n, 1, oh, ow])
            && (residual is null || fuseResidual) && activation != Activation.Sigmoid;
        Tensor? gated = null;
        if (gate is not null && !fuseGate)
        {
            gated = Binary(input, gate, BinaryOp.Mul);
            input = gated;
        }

        var epilogue = new Epilogue(fuseResidual ? residual!.FloatArray : null, activation, fuseMask ? mask!.FloatArray : null, m, oh * ow);
        if (group == c && cg == 1 && m % c == 0 && (m == c || kh * kw <= 25))
        {
            Depthwise(input.FloatArray, weight.FloatArray, b, output.FloatArray, n, p, epilogue);
        }
        else
        {
            GroupedGemm(input.FloatArray, weight.FloatArray, b, output.FloatArray, n, p, epilogue, fuseGate ? gate!.FloatArray : null);
        }

        gated?.Release();
        epilogue.ApplyWhole(output.FloatArray, output.Length);
        if (residual is not null && !fuseResidual)
        {
            Tensor sum = Binary(output, residual, BinaryOp.Add);
            output.Release();
            output = sum;
            if (act.Length > 0)
            {
                Tensor activated = Unary(output, Enum.Parse<UnaryOp>(act));
                output.Release();
                output = activated;
            }
        }

        if (mask is not null && !fuseMask)
        {
            Tensor masked = Binary(output, mask, BinaryOp.Mul);
            output.Release();
            output = masked;
        }

        return output;
    }

    /// <summary>Whether <see cref="GroupedGemm"/> will run this convolution as <see cref="PointwiseBatched"/> on the segmented GEMM.</summary>
    private static bool UsesPointwiseBatched(ConvParams p, int n) =>
        Gemm.Wide && p.Group == 1 && n > 1 && p.Oh * p.Ow < 1024
        && p.Kh == 1 && p.Kw == 1 && p.Sh == 1 && p.Sw == 1 && p.Pt == 0 && p.Pl == 0 && p.Oh == p.H && p.Ow == p.W;

    /// <summary>
    /// The attribute <see cref="InferenceSession"/>'s fusion pass puts on a Conv that absorbed the
    /// activation after it: <c>HardSwish</c>, <c>Relu</c> or <c>Sigmoid</c>. Inputs past the
    /// bias are absorbed element-wise nodes: [3] a residual from an <c>Add</c>, added before the
    /// activation; [4] a gate <c>[N, C, 1]</c> from a <c>Mul</c> on the input; [5] a mask
    /// <c>[N, 1, T]</c> from a <c>Mul</c> on the output, applied last.
    /// </summary>
    internal const string FusedActivation = "nanoocr.activation";

    internal enum Activation { None, HardSwish, Relu, Sigmoid }

    private static Activation ParseActivation(string act) => act switch
    {
        "" => Activation.None,
        "HardSwish" => Activation.HardSwish,
        "Relu" => Activation.Relu,
        "Sigmoid" => Activation.Sigmoid,
        _ => throw new NotSupportedException($"Fused activation {act}."),
    };

    /// <summary>
    /// What a fused convolution does to a stretch of its output once it is computed, while the
    /// stretch is still in cache: add the residual, then apply the activation — the same
    /// operations, in the same order, as the <c>Add</c> and activation nodes it replaced.
    /// </summary>
    internal readonly record struct Epilogue(float[]? Residual, Activation Act, float[]? Mask = null, int Channels = 0, int Pixels = 0)
    {
        public bool IsEmpty => Residual is null && Act == Activation.None && Mask is null;

        /// <summary>Applies the epilogue to <paramref name="dst"/>, which starts at <paramref name="offset"/> of the output.</summary>
        public void Apply(Span<float> dst, int offset)
        {
            if (Residual is not null)
            {
                TensorPrimitives.Add(dst, Residual.AsSpan(offset, dst.Length), dst);
            }

            switch (Act)
            {
                case Activation.HardSwish: Activations.HardSwish(dst, dst); break;
                case Activation.Relu: TensorPrimitives.Max(dst, 0f, dst); break;
            }

            if (Mask is not null)
            {
                // A stretch never crosses an output row; the mask is [N, 1, pixels].
                int item = offset / (Channels * Pixels), at = offset % Pixels;
                TensorPrimitives.Multiply(dst, Mask.AsSpan((item * Pixels) + at, dst.Length), dst);
            }
        }

        /// <summary>
        /// The part of the epilogue that runs over the whole output instead: the sigmoid, whose
        /// vectorized and scalar paths round differently, so it is applied over the same spans the
        /// unfused <c>Sigmoid</c> node would use (it follows only the small squeeze-excite convolutions).
        /// </summary>
        public void ApplyWhole(float[] output, int length)
        {
            if (Act == Activation.Sigmoid)
            {
                Parallelism.For(length, (long)length * 8, (s, e) => TensorPrimitives.Sigmoid(output.AsSpan(s, e - s), output.AsSpan(s, e - s)));
            }
        }
    }

    internal readonly record struct ConvParams(
        int C, int H, int W, int M, int Cg, int Kh, int Kw, int Sh, int Sw, int Dh, int Dw, int Pt, int Pl, int Oh, int Ow, int Group);

    /// <summary>
    /// Convolution as im2col + GEMM, tiled over output pixels so each worker's column buffer stays
    /// cache-sized. A 1×1, stride-1, unpadded convolution reads its input in place.
    /// </summary>
    private static void GroupedGemm(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p, Epilogue epilogue, float[]? gate = null)
    {
        int mg = p.M / p.Group;
        int k = p.Cg * p.Kh * p.Kw;
        int pixels = p.Oh * p.Ow;
        bool pointwise = p.Kh == 1 && p.Kw == 1 && p.Sh == 1 && p.Sw == 1 && p.Pt == 0 && p.Pl == 0 && p.Oh == p.H && p.Ow == p.W;
        int tile = Math.Clamp(65536 / Math.Max(k, 1), 32, 2048) / 32 * 32;
        int tiles = (pixels + tile - 1) / tile;
        int jobs = n * p.Group * tiles;
        long work = (long)n * p.M * k * pixels;

        void RunJob(int job, float[]? col)
        {
            int t = job % tiles;
            int g = job / tiles % p.Group;
            int b = job / tiles / p.Group;
            int p0 = t * tile;
            int p1 = Math.Min(pixels, p0 + tile);
            int cols = p1 - p0;
            int inBase = ((b * p.C) + (g * p.Cg)) * p.H * p.W;
            int outBase = ((b * p.M) + (g * mg)) * pixels;
            ReadOnlySpan<float> wg = weight.AsSpan(g * mg * k, mg * k);
            Span<float> outSpan = output.AsSpan(outBase);
            for (int r = 0; r < mg; r++)
            {
                outSpan.Slice((r * pixels) + p0, cols).Fill(bias is null ? 0f : bias[(g * mg) + r]);
            }

            if (pointwise)
            {
                GemmSerial.Multiply(mg, cols, k, wg, k, input.AsSpan(inBase + p0), p.H * p.W, outSpan.Slice(p0), pixels, accumulate: true);
            }
            else
            {
                Im2Col(input.AsSpan(inBase), p, p0, p1, col!);
                GemmSerial.Multiply(mg, cols, k, wg, k, col, cols, outSpan.Slice(p0), pixels, accumulate: true);
            }

            if (!epilogue.IsEmpty)
            {
                for (int r = 0; r < mg; r++)
                {
                    epilogue.Apply(outSpan.Slice((r * pixels) + p0, cols), outBase + (r * pixels) + p0);
                }
            }
        }

        if (pointwise && p.Group == 1 && n > 1 && pixels < 1024)
        {
            PointwiseBatched(input, weight, bias, output, n, p, epilogue, gate);
            return;
        }

        if (Gemm.Wide && !pointwise && p.Group == 1 && n > 1 && (long)n * k * pixels <= 1 << 19)
        {
            Im2ColBatched(input, weight, bias, output, n, p, epilogue);
            return;
        }

        int workers = Parallelism.Workers;
        if (workers <= 1 || jobs == 1 || work < 1 << 20)
        {
            float[]? col = pointwise ? null : ArrayPool<float>.Shared.Rent(k * tile);
            for (int j = 0; j < jobs; j++)
            {
                RunJob(j, col);
            }

            if (col is not null)
            {
                ArrayPool<float>.Shared.Return(col);
            }

            return;
        }

        Parallel.For(0, jobs, Parallelism.Options,
            () => pointwise ? null : ArrayPool<float>.Shared.Rent(k * tile),
            (j, _, col) =>
            {
                RunJob(j, col);
                return col;
            },
            col =>
            {
                if (col is not null)
                {
                    ArrayPool<float>.Shared.Return(col);
                }
            });
    }

    /// <summary>
    /// A 1×1 convolution over a batch of short sequences (the recognizers' <c>[B, C, T]</c> with T
    /// around a hundred) as one wide product: the batch is packed to <c>[C, B·T]</c>, multiplied,
    /// and unpacked. Per item the product would be 128 columns wide with a scalar tail; packed it is
    /// thousands wide and splits across every core.
    /// </summary>
    private static void PointwiseBatched(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p, Epilogue epilogue, float[]? gate = null)
    {
        int t = p.H * p.W, c = p.C, m = p.M, cols = n * t;
        if (Gemm.Wide)
        {
            // The product reads the items in place and writes them in place. Bias, residual,
            // HardSwish and mask are applied to each finished strip inside the product; a ReLU or
            // sigmoid epilogue (the small squeeze-excite convolutions) runs as a pass afterwards.
            // Either way: Σ, + bias, + residual, activation, × mask, each rounded once.
            bool inStrip = epilogue.Act is Activation.None or Activation.HardSwish;
            unsafe
            {
                fixed (float* pb = bias)
                fixed (float* pr = epilogue.Residual)
                fixed (float* pm = epilogue.Mask)
                {
                    var strip = inStrip
                        ? new Gemm.StripEpilogue((nint)pb, (nint)pr, epilogue.Act == Activation.HardSwish, (nint)pm)
                        : new Gemm.StripEpilogue((nint)pb, 0, false, 0);
                    Gemm.MultiplyBatched(m, t, n, c, weight, c, input, (long)c * t, output, (long)m * t, rowScale: gate is null ? default : gate.AsSpan(0, n * c), epilogue: strip);
                }
            }

            if (!inStrip && !epilogue.IsEmpty)
            {
                Parallelism.For(n, (long)m * cols, (b0, b1) =>
                {
                    for (int b = b0; b < b1; b++)
                    {
                        for (int r = 0; r < m; r++)
                        {
                            epilogue.Apply(output.AsSpan(((b * m) + r) * t, t), ((b * m) + r) * t);
                        }
                    }
                });
            }

            return;
        }

        float[] packed = System.Buffers.ArrayPool<float>.Shared.Rent(c * cols);
        float[] product = System.Buffers.ArrayPool<float>.Shared.Rent(m * cols);
        try
        {
            Parallelism.For(n, (long)c * cols, (b0, b1) =>
            {
                for (int b = b0; b < b1; b++)
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        Array.Copy(input, ((b * c) + ch) * t, packed, (ch * cols) + (b * t), t);
                    }
                }
            });
            Gemm.Multiply(m, cols, c, weight, c, packed, cols, product, cols);
            Parallelism.For(n, (long)m * cols, (b0, b1) =>
            {
                for (int b = b0; b < b1; b++)
                {
                    for (int r = 0; r < m; r++)
                    {
                        Span<float> dst = output.AsSpan(((b * m) + r) * t, t);
                        ReadOnlySpan<float> src = product.AsSpan((r * cols) + (b * t), t);
                        if (bias is null)
                        {
                            src.CopyTo(dst);
                        }
                        else
                        {
                            TensorPrimitives.Add(src, bias[r], dst);
                        }

                        if (!epilogue.IsEmpty)
                        {
                            epilogue.Apply(dst, ((b * m) + r) * t);
                        }
                    }
                }
            });
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(packed);
            System.Buffers.ArrayPool<float>.Shared.Return(product);
        }
    }

    /// <summary>
    /// A convolution over a batch of small images (the recognizer's 32×4 stem, one output row per
    /// line): every item's im2col columns <c>[k, pixels]</c> go into one buffer, and one segmented
    /// product covers the whole batch (<see cref="Gemm.MultiplyBatched"/>) instead of one narrow
    /// product per item with a partial strip at its end. Bias first, then the FMA chain, as the
    /// per-item path computes it.
    /// </summary>
    private static void Im2ColBatched(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p, Epilogue epilogue)
    {
        int k = p.Cg * p.Kh * p.Kw;
        int pixels = p.Oh * p.Ow;
        float[] col = System.Buffers.ArrayPool<float>.Shared.Rent(n * k * pixels);
        try
        {
            Parallelism.For(n, (long)n * k * pixels, (b0, b1) =>
            {
                float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(k * pixels);
                for (int b = b0; b < b1; b++)
                {
                    Im2Col(input.AsSpan(b * p.C * p.H * p.W), p, 0, pixels, scratch);
                    scratch.AsSpan(0, k * pixels).CopyTo(col.AsSpan(b * k * pixels));
                    for (int r = 0; r < p.M; r++)
                    {
                        output.AsSpan(((b * p.M) + r) * pixels, pixels).Fill(bias is null ? 0f : bias[r]);
                    }
                }

                System.Buffers.ArrayPool<float>.Shared.Return(scratch);
            });
            Gemm.MultiplyBatched(p.M, pixels, n, k, weight, k, col, (long)k * pixels, output, (long)p.M * pixels, accumulate: true);
            if (!epilogue.IsEmpty)
            {
                Parallelism.For(n * p.M, (long)n * p.M * pixels, (r0, r1) =>
                {
                    for (int r = r0; r < r1; r++)
                    {
                        epilogue.Apply(output.AsSpan(r * pixels, pixels), r * pixels);
                    }
                });
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(col);
        }
    }

    /// <summary>Fills <c>col[k, p1-p0]</c> with the receptive fields of output pixels <c>[p0, p1)</c>.</summary>
    private static void Im2Col(ReadOnlySpan<float> input, ConvParams p, int p0, int p1, float[] col)
    {
        int cols = p1 - p0;
        int row = 0;
        for (int c = 0; c < p.Cg; c++)
        {
            ReadOnlySpan<float> plane = input.Slice(c * p.H * p.W, p.H * p.W);
            for (int ky = 0; ky < p.Kh; ky++)
            {
                for (int kx = 0; kx < p.Kw; kx++, row++)
                {
                    Span<float> dst = col.AsSpan(row * cols, cols);
                    int i = 0;
                    int pix = p0;
                    while (pix < p1)
                    {
                        int oy = pix / p.Ow;
                        int ox = pix - (oy * p.Ow);
                        int run = Math.Min(p.Ow - ox, p1 - pix);
                        int iy = (oy * p.Sh) - p.Pt + (ky * p.Dh);
                        if ((uint)iy >= (uint)p.H)
                        {
                            dst.Slice(i, run).Clear();
                        }
                        else
                        {
                            ReadOnlySpan<float> src = plane.Slice(iy * p.W, p.W);
                            int ix = (ox * p.Sw) - p.Pl + (kx * p.Dw);
                            if (p.Sw == 1)
                            {
                                // In-bounds middle copied, padding at either end zeroed.
                                int lead = Math.Clamp(-ix, 0, run);
                                int end = Math.Clamp(p.W - ix, lead, run);
                                dst.Slice(i, lead).Clear();
                                src.Slice(ix + lead, end - lead).CopyTo(dst.Slice(i + lead, end - lead));
                                dst.Slice(i + end, run - end).Clear();
                            }
                            else
                            {
                                for (int q = 0; q < run; q++, ix += p.Sw)
                                {
                                    dst[i + q] = (uint)ix < (uint)p.W ? src[ix] : 0f;
                                }
                            }
                        }

                        i += run;
                        pix += run;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Depthwise convolution — one input channel per group, <c>M / C</c> filters per channel (one
    /// for an ordinary depthwise layer, sixteen for a stem over a single gray channel). Each input
    /// plane is first copied into a zero-padded scratch plane split into <c>stride</c> column
    /// phases, so that for every tap the inputs of consecutive output columns are consecutive
    /// floats (a stride-2 row becomes two stride-1 rows); then each vector of output columns
    /// accumulates all <c>kh·kw</c> taps in a register, bias first and taps in row-major order,
    /// one fused multiply-add per tap, four vectors at a time so the chains overlap. A padded tap
    /// adds <c>0·w</c>, which leaves the sum unchanged, so the padding needs no bounds checks.
    /// </summary>
    private static void Depthwise(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p, Epilogue epilogue)
    {
        int inPlanes = n * p.C;
        int mult = p.M / p.C;
        int outPlane = p.Oh * p.Ow;

        // The padded plane: rows -Pt .. (Oh-1)·Sh + (Kh-1)·Dh - Pt, likewise columns; each of the
        // Sw phases holds every Sw-th padded column, with slack for the widest vector step.
        int hp = ((p.Oh - 1) * p.Sh) + ((p.Kh - 1) * p.Dh) + 1;
        int wp = ((p.Ow - 1) * p.Sw) + ((p.Kw - 1) * p.Dw) + 1;
        int wq = ((wp + p.Sw - 1) / p.Sw) + 16;
        int scratch = p.Sw * hp * wq;
        int[] taps = new int[p.Kh * p.Kw];
        for (int ky = 0, t = 0; ky < p.Kh; ky++)
        {
            for (int kx = 0; kx < p.Kw; kx++, t++)
            {
                int c = kx * p.Dw;
                taps[t] = ((c % p.Sw) * hp * wq) + (ky * p.Dh * wq) + (c / p.Sw);
            }
        }

        void Plane(float[] pad, int ip, int j, int oy0, int oy1)
        {
            int m = ((ip % p.C) * mult) + j;
            int o = ((ip / p.C) * p.M) + m;
            Span<float> dst = output.AsSpan(o * outPlane, outPlane);
            DepthwisePlane(pad, taps, weight.AsSpan(m * p.Kh * p.Kw, p.Kh * p.Kw), bias is null ? 0f : bias[m], p, wq, dst, oy0, oy1);
            if (!epilogue.IsEmpty)
            {
                epilogue.Apply(dst.Slice(oy0 * p.Ow, (oy1 - oy0) * p.Ow), (o * outPlane) + (oy0 * p.Ow));
            }
        }

        long work = (long)n * p.M * outPlane * p.Kh * p.Kw;
        if (mult == 1 || inPlanes >= Parallelism.Workers)
        {
            Parallelism.For(inPlanes, work, (s, e) =>
            {
                float[] pad = System.Buffers.ArrayPool<float>.Shared.Rent(scratch);
                for (int ip = s; ip < e; ip++)
                {
                    PadPhases(input.AsSpan(ip * p.H * p.W, p.H * p.W), p, hp, wq, pad);
                    for (int j = 0; j < mult; j++)
                    {
                        Plane(pad, ip, j, 0, p.Oh);
                    }
                }

                System.Buffers.ArrayPool<float>.Shared.Return(pad);
            });
            return;
        }

        // Few input planes, many filters each (the stems): pad once, then bands of output rows,
        // every filter over a band before the next, so the band's input stays in cache instead
        // of the whole padded plane being streamed once per filter.
        float[] shared = System.Buffers.ArrayPool<float>.Shared.Rent(scratch);
        const int Band = 8;
        int bands = (p.Oh + Band - 1) / Band;
        for (int ip = 0; ip < inPlanes; ip++)
        {
            PadPhases(input.AsSpan(ip * p.H * p.W, p.H * p.W), p, hp, wq, shared);
            Parallelism.For(bands, work / inPlanes, (s, e) =>
            {
                for (int band = s; band < e; band++)
                {
                    for (int j = 0; j < mult; j++)
                    {
                        Plane(shared, ip, j, band * Band, Math.Min(p.Oh, (band + 1) * Band));
                    }
                }
            });
        }

        System.Buffers.ArrayPool<float>.Shared.Return(shared);
    }

    /// <summary>Copies a plane into the zero-padded, phase-split layout <see cref="Depthwise"/> reads.</summary>
    private static void PadPhases(ReadOnlySpan<float> src, ConvParams p, int hp, int wq, float[] pad)
    {
        for (int q = 0; q < p.Sw; q++)
        {
            // Phase q, column j holds padded column j·Sw + q, i.e. input column j·Sw + q - Pl.
            int off = q - p.Pl;
            int jlo = off >= 0 ? 0 : (-off + p.Sw - 1) / p.Sw;
            int jhi = Math.Min(wq, ((p.W - 1 - off) / p.Sw) + 1);
            for (int y = 0; y < hp; y++)
            {
                Span<float> dst = pad.AsSpan((q * hp * wq) + (y * wq), wq);
                int iy = y - p.Pt;
                if ((uint)iy >= (uint)p.H || jhi <= jlo)
                {
                    dst.Clear();
                    continue;
                }

                dst[..jlo].Clear();
                dst[jhi..].Clear();
                ReadOnlySpan<float> row = src.Slice(iy * p.W, p.W);
                if (p.Sw == 1)
                {
                    row.Slice(jlo + off, jhi - jlo).CopyTo(dst[jlo..]);
                }
                else if (p.Sw == 2)
                {
                    Deinterleave(row, (2 * jlo) + off, dst.Slice(jlo, jhi - jlo));
                }
                else
                {
                    for (int j = jlo; j < jhi; j++)
                    {
                        dst[j] = row[(j * p.Sw) + off];
                    }
                }
            }
        }
    }

    /// <summary><c>dst[j] = src[start + 2j]</c>.</summary>
    private static void Deinterleave(ReadOnlySpan<float> src, int start, Span<float> dst)
    {
        int j = 0;
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            for (; start + (2 * j) + 16 <= src.Length && j + 8 <= dst.Length; j += 8)
            {
                Vector256<float> a = Vector256.Create(src.Slice(start + (2 * j), 8));
                Vector256<float> b = Vector256.Create(src.Slice(start + (2 * j) + 8, 8));
                Vector256<float> even = System.Runtime.Intrinsics.X86.Avx.Shuffle(a, b, 0b10_00_10_00);
                System.Runtime.Intrinsics.X86.Avx2.Permute4x64(even.AsDouble(), 0b11_01_10_00).AsSingle().CopyTo(dst.Slice(j, 8));
            }
        }

        for (; j < dst.Length; j++)
        {
            dst[j] = src[start + (2 * j)];
        }
    }

    private static unsafe void DepthwisePlane(float[] pad, int[] taps, ReadOnlySpan<float> w, float bias, ConvParams p, int wq, Span<float> dst, int oy0, int oy1)
    {
        int nt = taps.Length;
        fixed (float* src = pad)
        fixed (int* tp = taps)
        fixed (float* wp = w)
        fixed (float* d = dst)
        {
            for (int oy = oy0; oy < oy1; oy++)
            {
                float* row = src + (oy * p.Sh * wq);
                float* o = d + (oy * p.Ow);
                int ox = Gemm.Wide ? DepthwiseRow512(row, tp, wp, nt, bias, o, p.Ow)
                    : Vector256.IsHardwareAccelerated ? DepthwiseRow256(row, tp, wp, nt, bias, o, p.Ow)
                    : 0;
                for (; ox < p.Ow; ox++)
                {
                    float acc = bias;
                    float* r = row + ox;
                    for (int t = 0; t < nt; t++)
                    {
                        acc = MathF.FusedMultiplyAdd(r[tp[t]], wp[t], acc);
                    }

                    o[ox] = acc;
                }
            }
        }
    }

    /// <summary>One output row, 16 columns per vector; returns the first column not done.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static unsafe int DepthwiseRow512(float* row, int* tp, float* wp, int nt, float bias, float* o, int ow)
    {
        int ox = 0;
        var b = Vector512.Create(bias);
        for (; ox + 64 <= ow; ox += 64)
        {
            Vector512<float> a0 = b, a1 = b, a2 = b, a3 = b;
            float* r = row + ox;
            for (int t = 0; t < nt; t++)
            {
                float* q = r + tp[t];
                var wv = Vector512.Create(wp[t]);
                a0 = Vector512.FusedMultiplyAdd(Vector512.Load(q), wv, a0);
                a1 = Vector512.FusedMultiplyAdd(Vector512.Load(q + 16), wv, a1);
                a2 = Vector512.FusedMultiplyAdd(Vector512.Load(q + 32), wv, a2);
                a3 = Vector512.FusedMultiplyAdd(Vector512.Load(q + 48), wv, a3);
            }

            a0.Store(o + ox);
            a1.Store(o + ox + 16);
            a2.Store(o + ox + 32);
            a3.Store(o + ox + 48);
        }

        for (; ox < ow && ow >= 16; ox += 16)
        {
            // The last vector may overlap the previous one: each column is computed from scratch,
            // so the overlap rewrites the same values.
            ox = Math.Min(ox, ow - 16);
            Vector512<float> a0 = b;
            float* r = row + ox;
            for (int t = 0; t < nt; t++)
            {
                a0 = Vector512.FusedMultiplyAdd(Vector512.Load(r + tp[t]), Vector512.Create(wp[t]), a0);
            }

            a0.Store(o + ox);
        }

        return ox;
    }

    /// <summary>One output row, 8 columns per vector; returns the first column not done.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static unsafe int DepthwiseRow256(float* row, int* tp, float* wp, int nt, float bias, float* o, int ow)
    {
        int ox = 0;
        var b = Vector256.Create(bias);
        for (; ox + 32 <= ow; ox += 32)
        {
            Vector256<float> a0 = b, a1 = b, a2 = b, a3 = b;
            float* r = row + ox;
            for (int t = 0; t < nt; t++)
            {
                float* q = r + tp[t];
                var wv = Vector256.Create(wp[t]);
                a0 = Vector256.FusedMultiplyAdd(Vector256.Load(q), wv, a0);
                a1 = Vector256.FusedMultiplyAdd(Vector256.Load(q + 8), wv, a1);
                a2 = Vector256.FusedMultiplyAdd(Vector256.Load(q + 16), wv, a2);
                a3 = Vector256.FusedMultiplyAdd(Vector256.Load(q + 24), wv, a3);
            }

            a0.Store(o + ox);
            a1.Store(o + ox + 8);
            a2.Store(o + ox + 16);
            a3.Store(o + ox + 24);
        }

        for (; ox < ow && ow >= 8; ox += 8)
        {
            // The last vector may overlap the previous one (each column is computed from scratch).
            ox = Math.Min(ox, ow - 8);
            Vector256<float> a0 = b;
            float* r = row + ox;
            for (int t = 0; t < nt; t++)
            {
                a0 = Vector256.FusedMultiplyAdd(Vector256.Load(r + tp[t]), Vector256.Create(wp[t]), a0);
            }

            a0.Store(o + ox);
        }

        return ox;
    }

    internal static Tensor MatMul(Tensor a, Tensor b)
    {
        int[] sa = a.Rank == 1 ? [1, a.Shape[0]] : a.Shape;
        int[] sb = b.Rank == 1 ? [b.Shape[0], 1] : b.Shape;
        int m = sa[^2], k = sa[^1], n = sb[^1];
        if (sb[^2] != k)
        {
            throw new InvalidOperationException($"MatMul inner dimensions differ: {k} and {sb[^2]}.");
        }

        int[] batchA = sa[..^2], batchB = sb[..^2];
        int[] batch = Broadcast.Shape(batchA, batchB);
        var outShape = new List<int>(batch) { m, n };
        if (a.Rank == 1)
        {
            outShape.RemoveAt(outShape.Count - 2);
        }

        if (b.Rank == 1)
        {
            outShape.RemoveAt(outShape.Count - 1);
        }

        Tensor result = Tensor.Float([.. outShape]);
        float[] x = a.FloatArray, y = b.FloatArray, z = result.FloatArray;

        // A shared 2-D right operand (a weight): fold every batch row into one product.
        if (batchB.Length == 0 || Tensor.SizeOf(batchB) == 1 && batchA.Length >= batchB.Length)
        {
            int rows = Tensor.SizeOf(batchA) * m;
            Gemm.Multiply(rows, n, k, x, k, y, n, z, n);
            return result;
        }

        int count = Tensor.SizeOf(batch);
        int[] offA = Broadcast.Offsets(batchA, batch);
        int[] offB = Broadcast.Offsets(batchB, batch);
        long per = (long)m * n * k;
        if (count >= Parallelism.Workers && per < 1 << 22)
        {
            Parallelism.For(count, per * count, (s, e) =>
            {
                for (int i = s; i < e; i++)
                {
                    GemmSerial.Multiply(m, n, k, x.AsSpan(offA[i] * m * k), k, y.AsSpan(offB[i] * k * n), n, z.AsSpan(i * m * n), n);
                }
            });
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                Gemm.Multiply(m, n, k, x.AsSpan(offA[i] * m * k), k, y.AsSpan(offB[i] * k * n), n, z.AsSpan(i * m * n), n);
            }
        }

        return result;
    }

    internal static Tensor GemmOp(OnnxNode node, List<Tensor?> x)
    {
        Tensor a = x[0]!, b = x[1]!;
        Tensor? c = x.Count > 2 ? x[2] : null;
        float alpha = node.Float("alpha", 1f), beta = node.Float("beta", 1f);
        bool ta = node.Int("transA", 0) != 0, tb = node.Int("transB", 0) != 0;
        int m = ta ? a.Shape[1] : a.Shape[0];
        int k = ta ? a.Shape[0] : a.Shape[1];
        int n = tb ? b.Shape[0] : b.Shape[1];
        float[] av = a.FloatArray, bv = b.FloatArray;
        if (ta)
        {
            av = new float[m * k];
            Gemm.Transpose(a.Floats, k, m, av);
        }

        if (tb)
        {
            bv = new float[k * n];
            Gemm.Transpose(b.Floats, n, k, bv);
        }

        Tensor result = Tensor.Float([m, n]);
        Span<float> z = result.Floats;
        Gemm.Multiply(m, n, k, av, k, bv, n, z, n);
        if (alpha != 1f)
        {
            TensorPrimitives.Multiply(z, alpha, z);
        }

        if (c is not null && beta != 0f)
        {
            int[] offsets = Broadcast.Offsets(c.Shape, [m, n]);
            Span<float> cv = c.Floats;
            for (int i = 0; i < z.Length; i++)
            {
                z[i] += beta * cv[offsets[i]];
            }
        }

        return result;
    }

    internal static Tensor BatchNorm(OnnxNode node, List<Tensor?> x)
    {
        Tensor input = x[0]!;
        Span<float> scale = x[1]!.Floats, bias = x[2]!.Floats, mean = x[3]!.Floats, variance = x[4]!.Floats;
        float eps = node.Float("epsilon", 1e-5f);
        int n = input.Shape[0], c = input.Shape[1];
        int inner = input.Length / Math.Max(1, n * c);
        Tensor result = Tensor.Float((int[])input.Shape.Clone());
        float[] src = input.FloatArray, dst = result.FloatArray;
        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                float s = scale[ch] / MathF.Sqrt(variance[ch] + eps);
                float t = bias[ch] - (mean[ch] * s);
                int off = ((b * c) + ch) * inner;
                TensorPrimitives.Multiply(src.AsSpan(off, inner), s, dst.AsSpan(off, inner));
                TensorPrimitives.Add(dst.AsSpan(off, inner), t, dst.AsSpan(off, inner));
            }
        }

        return result;
    }

    internal static Tensor LayerNorm(OnnxNode node, List<Tensor?> x)
    {
        Tensor input = x[0]!;
        Tensor scale = x[1]!;
        Tensor? bias = x.Count > 2 ? x[2] : null;
        int axis = Axis(node.Int("axis", -1), input.Rank);
        float eps = node.Float("epsilon", 1e-5f);
        int inner = 1;
        for (int i = axis; i < input.Rank; i++)
        {
            inner *= input.Shape[i];
        }

        int rows = inner == 0 ? 0 : input.Length / inner;
        Tensor result = Tensor.Float((int[])input.Shape.Clone());
        float[] src = input.FloatArray, dst = result.FloatArray, g = scale.FloatArray;
        float[]? bb = bias?.FloatArray;
        Parallelism.For(rows, (long)rows * inner * 4, (s, e) =>
        {
            for (int r = s; r < e; r++)
            {
                ReadOnlySpan<float> row = src.AsSpan(r * inner, inner);
                Span<float> outRow = dst.AsSpan(r * inner, inner);
                float mean = TensorPrimitives.Sum(row) / inner;
                TensorPrimitives.Subtract(row, mean, outRow);
                float variance = TensorPrimitives.SumOfSquares(outRow) / inner;
                float inv = 1f / MathF.Sqrt(variance + eps);
                TensorPrimitives.Multiply(outRow, inv, outRow);
                TensorPrimitives.Multiply(outRow, g.AsSpan(0, inner), outRow);
                if (bb is not null)
                {
                    TensorPrimitives.Add(outRow, bb.AsSpan(0, inner), outRow);
                }
            }
        });
        return result;
    }

    internal static Tensor Pool(OnnxNode node, Tensor input, bool max)
    {
        long[] kernel = node.Ints("kernel_shape")!;
        long[] strides = node.Ints("strides") ?? [1, 1];
        long[] pads = node.Ints("pads") ?? [0, 0, 0, 0];
        bool ceil = node.Int("ceil_mode", 0) != 0;
        bool includePad = node.Int("count_include_pad", 0) != 0;
        int n = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        int kh = (int)kernel[0], kw = (int)kernel[1], sh = (int)strides[0], sw = (int)strides[1];
        int pt = (int)pads[0], pl = (int)pads[1], pb = (int)pads[2], pr = (int)pads[3];
        int oh = ceil ? (int)Math.Ceiling((h + pt + pb - kh) / (double)sh) + 1 : ((h + pt + pb - kh) / sh) + 1;
        int ow = ceil ? (int)Math.Ceiling((w + pl + pr - kw) / (double)sw) + 1 : ((w + pl + pr - kw) / sw) + 1;
        Tensor result = Tensor.Float([n, c, oh, ow]);
        float[] src = input.FloatArray, dst = result.FloatArray;
        if (kh == 2 && kw == 2 && sh == 2 && sw == 2 && pt == 0 && pl == 0 && pb == 0 && pr == 0 && !ceil)
        {
            Parallelism.For(n * c, (long)n * c * oh * ow * 4, (s, e) =>
            {
                for (int plane = s; plane < e; plane++)
                {
                    for (int oy = 0; oy < oh; oy++)
                    {
                        Pool2x2Row(src.AsSpan((plane * h * w) + (2 * oy * w), w), src.AsSpan((plane * h * w) + (((2 * oy) + 1) * w), w), dst.AsSpan((plane * oh * ow) + (oy * ow), ow), max);
                    }
                }
            });
            return result;
        }

        Parallelism.For(n * c, (long)n * c * oh * ow * kh * kw, (s, e) =>
        {
            for (int plane = s; plane < e; plane++)
            {
                int ib = plane * h * w, ob = plane * oh * ow;
                for (int oy = 0; oy < oh; oy++)
                {
                    for (int ox = 0; ox < ow; ox++)
                    {
                        int y0 = (oy * sh) - pt, x0 = (ox * sw) - pl;
                        float acc = max ? float.NegativeInfinity : 0f;
                        int count = 0;
                        for (int ky = 0; ky < kh; ky++)
                        {
                            int iy = y0 + ky;
                            if ((uint)iy >= (uint)h)
                            {
                                continue;
                            }

                            for (int kx = 0; kx < kw; kx++)
                            {
                                int ix = x0 + kx;
                                if ((uint)ix >= (uint)w)
                                {
                                    continue;
                                }

                                float v = src[ib + (iy * w) + ix];
                                acc = max ? MathF.Max(acc, v) : acc + v;
                                count++;
                            }
                        }

                        if (!max)
                        {
                            int denom = includePad
                                ? (Math.Min(y0 + kh, h + pb) - y0) * (Math.Min(x0 + kw, w + pr) - x0)
                                : count;
                            acc /= Math.Max(denom, 1);
                        }

                        dst[ob + (oy * ow) + ox] = acc;
                    }
                }
            }
        });
        return result;
    }

    /// <summary>
    /// One output row of a 2×2, stride-2, unpadded pool: <c>max(a, b, c, d)</c>, or
    /// <c>(((a + b) + c) + d) / 4</c> — the general loop's order — with the window's columns
    /// split into even and odd lanes.
    /// </summary>
    private static void Pool2x2Row(ReadOnlySpan<float> r0, ReadOnlySpan<float> r1, Span<float> dst, bool max)
    {
        int x = 0;
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            for (; (2 * x) + 16 <= r0.Length && x + 8 <= dst.Length; x += 8)
            {
                Vector256<float> a0 = Vector256.Create(r0.Slice(2 * x, 8)), a1 = Vector256.Create(r0.Slice((2 * x) + 8, 8));
                Vector256<float> b0 = Vector256.Create(r1.Slice(2 * x, 8)), b1 = Vector256.Create(r1.Slice((2 * x) + 8, 8));
                Vector256<float> e0 = Even(a0, a1), o0 = Odd(a0, a1), e1 = Even(b0, b1), o1 = Odd(b0, b1);
                Vector256<float> v = max
                    ? Vector256.Max(Vector256.Max(Vector256.Max(e0, o0), e1), o1)
                    : (((e0 + o0) + e1) + o1) / Vector256.Create(4f);
                v.CopyTo(dst.Slice(x, 8));
            }
        }

        for (; x < dst.Length; x++)
        {
            float a = r0[2 * x], b = r0[(2 * x) + 1], c = r1[2 * x], d = r1[(2 * x) + 1];
            dst[x] = max ? MathF.Max(MathF.Max(MathF.Max(a, b), c), d) : (((a + b) + c) + d) / 4;
        }

        static Vector256<float> Even(Vector256<float> a, Vector256<float> b) =>
            System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx.Shuffle(a, b, 0b10_00_10_00).AsDouble(), 0b11_01_10_00).AsSingle();

        static Vector256<float> Odd(Vector256<float> a, Vector256<float> b) =>
            System.Runtime.Intrinsics.X86.Avx2.Permute4x64(System.Runtime.Intrinsics.X86.Avx.Shuffle(a, b, 0b11_01_11_01).AsDouble(), 0b11_01_10_00).AsSingle();
    }

    internal static Tensor Resize(OnnxNode node, List<Tensor?> x)
    {
        Tensor input = x[0]!;
        Tensor? scales = x.Count > 2 ? x[2] : null;
        Tensor? sizes = x.Count > 3 ? x[3] : null;
        string mode = node.Str("mode", "nearest");
        string transform = node.Str("coordinate_transformation_mode", "half_pixel");
        string nearestMode = node.Str("nearest_mode", "round_prefer_floor");
        if (mode != "nearest" || input.Rank != 4)
        {
            throw new NotSupportedException($"Resize mode '{mode}' on rank {input.Rank}.");
        }

        int rank = input.Rank;
        var outShape = new int[rank];
        var scale = new float[rank];
        if (sizes is not null && sizes.Length > 0)
        {
            long[] s = sizes.ToLongArray();
            for (int i = 0; i < rank; i++)
            {
                outShape[i] = (int)s[i];
                scale[i] = (float)outShape[i] / input.Shape[i];
            }
        }
        else
        {
            float[] s = scales!.ToFloatArray();
            for (int i = 0; i < rank; i++)
            {
                scale[i] = s[i];
                outShape[i] = (int)MathF.Floor(input.Shape[i] * s[i]);
            }
        }

        if (outShape[0] != input.Shape[0] || outShape[1] != input.Shape[1])
        {
            throw new NotSupportedException("Resize over batch or channel axes.");
        }

        int h = input.Shape[2], w = input.Shape[3], oh = outShape[2], ow = outShape[3];
        int[] ys = NearestIndices(h, oh, scale[2], transform, nearestMode);
        int[] xs = NearestIndices(w, ow, scale[3], transform, nearestMode);
        Tensor result = Tensor.Float(outShape);
        float[] src = input.FloatArray, dst = result.FloatArray;
        int planes = outShape[0] * outShape[1];
        bool doubling = oh == 2 * h && ow == 2 * w && ys.Select((v, i) => v == i / 2).All(b => b) && xs.Select((v, i) => v == i / 2).All(b => b);
        if (doubling && Vector256.IsHardwareAccelerated)
        {
            // The FPN's nearest 2× upsample: each source row, every pixel twice, written to two rows.
            Parallelism.For(planes, (long)planes * oh * ow, (s, e) =>
            {
                for (int plane = s; plane < e; plane++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        ReadOnlySpan<float> row = src.AsSpan((plane * h * w) + (y * w), w);
                        Span<float> d0 = dst.AsSpan((plane * oh * ow) + (2 * y * ow), ow);
                        int x = 0;
                        for (; x + 8 <= w; x += 8)
                        {
                            Vector256<float> v = Vector256.Create(row.Slice(x, 8));
                            Vector256<float> lo = Vector256.Shuffle(v, Vector256.Create(0, 0, 1, 1, 2, 2, 3, 3));
                            Vector256<float> hi = Vector256.Shuffle(v, Vector256.Create(4, 4, 5, 5, 6, 6, 7, 7));
                            lo.CopyTo(d0.Slice(2 * x, 8));
                            hi.CopyTo(d0.Slice((2 * x) + 8, 8));
                        }

                        for (; x < w; x++)
                        {
                            d0[2 * x] = d0[(2 * x) + 1] = row[x];
                        }

                        d0.CopyTo(dst.AsSpan((plane * oh * ow) + (((2 * y) + 1) * ow), ow));
                    }
                }
            });
            return result;
        }

        Parallelism.For(planes, (long)planes * oh * ow, (s, e) =>
        {
            for (int plane = s; plane < e; plane++)
            {
                for (int oy = 0; oy < oh; oy++)
                {
                    int srcRow = (plane * h * w) + (ys[oy] * w);
                    int dstRow = (plane * oh * ow) + (oy * ow);
                    for (int ox = 0; ox < ow; ox++)
                    {
                        dst[dstRow + ox] = src[srcRow + xs[ox]];
                    }
                }
            }
        });
        return result;
    }

    private static int[] NearestIndices(int inSize, int outSize, float scale, string transform, string nearestMode)
    {
        var idx = new int[outSize];
        for (int o = 0; o < outSize; o++)
        {
            float orig = transform switch
            {
                "asymmetric" => o / scale,
                "align_corners" => outSize == 1 ? 0 : o * (inSize - 1) / (float)(outSize - 1),
                "pytorch_half_pixel" => outSize > 1 ? ((o + 0.5f) / scale) - 0.5f : 0,
                _ => ((o + 0.5f) / scale) - 0.5f,
            };
            float r = nearestMode switch
            {
                "floor" => MathF.Floor(orig),
                "ceil" => MathF.Ceiling(orig),
                "round_prefer_ceil" => MathF.Round(orig, MidpointRounding.AwayFromZero),
                _ => orig == MathF.Floor(orig) + 0.5f ? MathF.Floor(orig) : MathF.Round(orig),
            };
            idx[o] = Math.Clamp((int)r, 0, inSize - 1);
        }

        return idx;
    }
}
