using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using NanoOcr.Core;

namespace NanoOcr.Onnx;

internal static partial class Operators
{
    internal static Tensor Conv(OnnxNode node, List<Tensor?> x)
    {
        Tensor input = x[0]!, weight = x[1]!;
        Tensor? bias = x.Count > 2 ? x[2] : null;
        if (input.Rank == 3)
        {
            // A 1-D convolution is a 2-D one over a height of 1.
            Tensor in2 = input.Reshape([input.Shape[0], input.Shape[1], 1, input.Shape[2]]);
            Tensor w2 = weight.Reshape([weight.Shape[0], weight.Shape[1], 1, weight.Shape[2]]);
            var attrs = new OnnxNode { OpType = "Conv", Name = node.Name };
            attrs.Attributes["group"] = new OnnxAttribute { Name = "group", I = node.Int("group", 1) };
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
            Tensor out2 = Conv(attrs, [in2, w2, bias]);
            Tensor out1 = out2.Reshape([out2.Shape[0], out2.Shape[1], out2.Shape[3]]);
            in2.Release();
            w2.Release();
            out2.Release();
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
        if (group == c && group == m && cg == 1)
        {
            Depthwise(input.FloatArray, weight.FloatArray, b, output.FloatArray, n, p);
        }
        else
        {
            GroupedGemm(input.FloatArray, weight.FloatArray, b, output.FloatArray, n, p);
        }

        return output;
    }

    internal readonly record struct ConvParams(
        int C, int H, int W, int M, int Cg, int Kh, int Kw, int Sh, int Sw, int Dh, int Dw, int Pt, int Pl, int Oh, int Ow, int Group);

    /// <summary>
    /// Convolution as im2col + GEMM, tiled over output pixels so each worker's column buffer stays
    /// cache-sized. A 1×1, stride-1, unpadded convolution reads its input in place.
    /// </summary>
    private static void GroupedGemm(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p)
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
                return;
            }

            Im2Col(input.AsSpan(inBase), p, p0, p1, col!);
            GemmSerial.Multiply(mg, cols, k, wg, k, col, cols, outSpan.Slice(p0), pixels, accumulate: true);
        }

        if (pointwise && p.Group == 1 && n > 1 && pixels < 1024)
        {
            PointwiseBatched(input, weight, bias, output, n, p);
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
    private static void PointwiseBatched(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p)
    {
        int t = p.H * p.W, c = p.C, m = p.M, cols = n * t;
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
                            if (p.Sw == 1 && ix >= 0 && ix + run <= p.W)
                            {
                                src.Slice(ix, run).CopyTo(dst.Slice(i, run));
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

    /// <summary>Depthwise convolution: one filter per channel, accumulated row by row.</summary>
    private static void Depthwise(float[] input, float[] weight, float[]? bias, float[] output, int n, ConvParams p)
    {
        int planes = n * p.C;
        long work = (long)planes * p.Oh * p.Ow * p.Kh * p.Kw;
        Parallelism.For(planes, work, (s, e) =>
        {
            for (int plane = s; plane < e; plane++)
            {
                int ch = plane % p.C;
                ReadOnlySpan<float> src = input.AsSpan(plane * p.H * p.W, p.H * p.W);
                Span<float> dst = output.AsSpan(plane * p.Oh * p.Ow, p.Oh * p.Ow);
                ReadOnlySpan<float> kernel = weight.AsSpan(ch * p.Kh * p.Kw, p.Kh * p.Kw);
                dst.Fill(bias is null ? 0f : bias[ch]);
                for (int oy = 0; oy < p.Oh; oy++)
                {
                    Span<float> outRow = dst.Slice(oy * p.Ow, p.Ow);
                    for (int ky = 0; ky < p.Kh; ky++)
                    {
                        int iy = (oy * p.Sh) - p.Pt + (ky * p.Dh);
                        if ((uint)iy >= (uint)p.H)
                        {
                            continue;
                        }

                        ReadOnlySpan<float> inRow = src.Slice(iy * p.W, p.W);
                        for (int kx = 0; kx < p.Kw; kx++)
                        {
                            float wv = kernel[(ky * p.Kw) + kx];
                            int off = (kx * p.Dw) - p.Pl;

                            // Output columns whose tap lands inside the row.
                            int lo = off >= 0 ? 0 : (-off + p.Sw - 1) / p.Sw;
                            int hi = Math.Min(p.Ow, ((p.W - 1 - off) / p.Sw) + 1);
                            if (hi <= lo)
                            {
                                continue;
                            }

                            if (p.Sw == 1)
                            {
                                TensorPrimitives.MultiplyAdd(inRow.Slice(lo + off, hi - lo), wv, outRow.Slice(lo, hi - lo), outRow.Slice(lo, hi - lo));
                            }
                            else
                            {
                                for (int ox = lo; ox < hi; ox++)
                                {
                                    outRow[ox] = MathF.FusedMultiplyAdd(inRow[(ox * p.Sw) + off], wv, outRow[ox]);
                                }
                            }
                        }
                    }
                }
            }
        });
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
