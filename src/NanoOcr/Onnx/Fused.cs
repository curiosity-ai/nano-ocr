using System.Runtime.Intrinsics;

namespace NanoOcr.Onnx;

internal static partial class Operators
{
    /// <summary>Domain of the operators the fusion pass introduces; not part of any ONNX file.</summary>
    internal const string FusedDomain = "nanoocr";

    private static readonly Dictionary<string, Kernel> FusedTable = new(StringComparer.Ordinal)
    {
        ["SoftmaxPeak"] = (n, x) => SoftmaxPeak(x[0]!),
        ["ConvSoftmaxPeak"] = (n, x) => ConvSoftmaxPeak(x[0]!, x[1]!, x.Count > 2 ? x[2] : null),
    };

    /// <summary>
    /// A 1×1 convolution <c>[B, K, T] → [B, C, T]</c> (the CTC head, C = 1678 classes) followed by
    /// <see cref="SoftmaxPeak"/>, without materializing the logits: they are computed 64 time
    /// steps at a time into a scratch tile that stays in L2, reduced, and dropped. The tensor the
    /// graph form writes and reads back is 26 MB a page. Each logit is the same FMA chain plus
    /// bias as the batched convolution computes, so max and arg-max are unchanged.
    /// </summary>
    internal static Tensor[] ConvSoftmaxPeak(Tensor x, Tensor w, Tensor? bias)
    {
        int b = x.Shape[0], k = x.Shape[1], t = x.Shape[2], c = w.Shape[0];
        const int Chunk = 64;
        Tensor max = Tensor.Float([b, t]);
        Tensor arg = Tensor.Int64([b, t]);
        Tensor sum = Tensor.Float([b, t]);
        float[] src = x.FloatArray, wt = w.FloatArray, mx = max.FloatArray, sm = sum.FloatArray;
        float[]? bs = bias?.FloatArray;
        long[] am = arg.LongArray;
        int chunks = (t + Chunk - 1) / Chunk;
        Core.Parallelism.For(b * chunks, (long)b * c * k * t, (j0, j1) =>
        {
            float[] tile = System.Buffers.ArrayPool<float>.Shared.Rent(c * Chunk);
            for (int j = j0; j < j1; j++)
            {
                int item = j / chunks, t0 = (j % chunks) * Chunk, width = Math.Min(Chunk, t - t0);
                Core.GemmSerial.Multiply(c, width, k, wt, k, src.AsSpan((item * k * t) + t0), t, tile, width);
                if (bs is not null)
                {
                    for (int r = 0; r < c; r++)
                    {
                        Span<float> row = tile.AsSpan(r * width, width);
                        System.Numerics.Tensors.TensorPrimitives.Add(row, bs[r], row);
                    }
                }

                int o = (item * t) + t0;
                SoftmaxPeakItem(tile.AsSpan(0, c * width), c, width, mx.AsSpan(o, width), am.AsSpan(o, width), sm.AsSpan(o, width));
            }

            System.Buffers.ArrayPool<float>.Shared.Return(tile);
        });
        return [max, arg, sum];
    }

    /// <summary>
    /// The CTC head's tail in one operator, over logits <c>X[B, C, T]</c> (classes before time):
    /// what <c>Transpose(perm=[0,2,1]) → ReduceMax(-1) / ArgMax(-1) → Sub(·, max) → Exp →
    /// ReduceSum(-1)</c> computes, as the three outputs <c>max[B,T]</c>, <c>argmax[B,T]</c> (the
    /// first maximal class) and <c>Σ exp(x − max)[B,T]</c>. The graph form transposes 2.9 M
    /// logits and then walks them eight times; here each item's class rows are streamed twice,
    /// lanes running over time steps, with the per-step state in L1. The arg-max and max are
    /// exact; the sum adds the same terms in class order (ONNX Runtime's order differs as well).
    /// </summary>
    internal static Tensor[] SoftmaxPeak(Tensor x)
    {
        int b = x.Shape[0], c = x.Shape[1], t = x.Shape[2];
        Tensor max = Tensor.Float([b, t]);
        Tensor arg = Tensor.Int64([b, t]);
        Tensor sum = Tensor.Float([b, t]);
        float[] src = x.FloatArray, mx = max.FloatArray, sm = sum.FloatArray;
        long[] am = arg.LongArray;
        Core.Parallelism.For(b, (long)b * c * t * 4, (b0, b1) =>
        {
            for (int item = b0; item < b1; item++)
            {
                SoftmaxPeakItem(src.AsSpan(item * c * t, c * t), c, t, mx.AsSpan(item * t, t), am.AsSpan(item * t, t), sm.AsSpan(item * t, t));
            }
        });
        return [max, arg, sum];
    }

    /// <summary>
    /// A vector of terms all below <c>exp(−25) ≈ 1.4e-11</c> is not added. The sum always holds
    /// <c>exp(0) = 1</c>, and even all 1678 classes at that bound come to 2.3e-8, under half an ulp
    /// of 1, so skipping them moves the sum by at most an ulp.
    /// </summary>
    private const float Negligible = -25f;

    /// <summary>
    /// One item: for each vector of time steps, the running max, arg-max and then the sum of
    /// exponentials stay in registers across all class rows (the rows are <c>t</c> floats apart),
    /// and are stored once. The last vector overlaps the one before it when <c>t</c> is not a
    /// multiple of the width; each lane is computed from scratch, so the overlap is harmless.
    /// </summary>
    private static unsafe void SoftmaxPeakItem(ReadOnlySpan<float> rows, int c, int t, Span<float> max, Span<long> arg, Span<float> sum)
    {
        fixed (float* x = rows)
        fixed (float* m = max)
        fixed (long* a = arg)
        fixed (float* s = sum)
        {
            int i = 0;
            if (Core.Gemm.Wide && t >= 16)
            {
                for (int i0 = 0; i0 < t; i0 += 16)
                {
                    i = Math.Min(i0, t - 16);
                    float* p = x + i;
                    Vector512<float> best = Vector512.Load(p);
                    Vector512<int> idx = Vector512<int>.Zero;
                    for (int k = 1; k < c; k++)
                    {
                        Vector512<float> v = Vector512.Load(p + ((long)k * t));
                        Vector512<float> gt = Vector512.GreaterThan(v, best);
                        best = Vector512.ConditionalSelect(gt, v, best);
                        idx = Vector512.ConditionalSelect(gt.AsInt32(), Vector512.Create(k), idx);
                    }

                    Vector512<float> acc = Vector512<float>.Zero;
                    var floor = Vector512.Create(Negligible);
                    for (int k = 0; k < c; k++)
                    {
                        Vector512<float> d = Vector512.Load(p + ((long)k * t)) - best;
                        if (!Vector512.LessThanAll(d, floor))
                        {
                            acc += ExpNonPositive(d);
                        }
                    }

                    best.Store(m + i);
                    acc.Store(s + i);
                    for (int l = 0; l < 16; l++)
                    {
                        a[i + l] = idx.GetElement(l);
                    }
                }

                return;
            }

            if (Vector256.IsHardwareAccelerated && t >= 8)
            {
                for (int i0 = 0; i0 < t; i0 += 8)
                {
                    i = Math.Min(i0, t - 8);
                    float* p = x + i;
                    Vector256<float> best = Vector256.Load(p);
                    Vector256<int> idx = Vector256<int>.Zero;
                    for (int k = 1; k < c; k++)
                    {
                        Vector256<float> v = Vector256.Load(p + ((long)k * t));
                        Vector256<float> gt = Vector256.GreaterThan(v, best);
                        best = Vector256.ConditionalSelect(gt, v, best);
                        idx = Vector256.ConditionalSelect(gt.AsInt32(), Vector256.Create(k), idx);
                    }

                    Vector256<float> acc = Vector256<float>.Zero;
                    var floor = Vector256.Create(Negligible);
                    for (int k = 0; k < c; k++)
                    {
                        Vector256<float> d = Vector256.Load(p + ((long)k * t)) - best;
                        if (!Vector256.LessThanAll(d, floor))
                        {
                            acc += ExpNonPositive(d);
                        }
                    }

                    best.Store(m + i);
                    acc.Store(s + i);
                    for (int l = 0; l < 8; l++)
                    {
                        a[i + l] = idx.GetElement(l);
                    }
                }

                return;
            }

            for (; i < t; i++)
            {
                float bv = x[i];
                int bi = 0;
                for (int k = 1; k < c; k++)
                {
                    if (x[((long)k * t) + i] > bv)
                    {
                        bv = x[((long)k * t) + i];
                        bi = k;
                    }
                }

                float acc = 0f;
                for (int k = 0; k < c; k++)
                {
                    acc += MathF.Exp(x[((long)k * t) + i] - bv);
                }

                m[i] = bv;
                a[i] = bi;
                s[i] = acc;
            }
        }
    }

    /// <summary>
    /// <c>exp(x)</c> for <c>x ≤ 0</c> (Cephes' <c>expf</c>: <c>x = n·ln2 + r</c>, a degree-6
    /// polynomial in <c>r</c>, scaled by <c>2ⁿ</c>), within ~2 ulp of the true value; below −87
    /// it returns 0. About four times cheaper than <see cref="Vector256.Exp(Vector256{float})"/>,
    /// which evaluates in double. It only feeds the softmax peak, whose sum is already in an
    /// order of its own.
    /// </summary>
    private static Vector512<float> ExpNonPositive(Vector512<float> x)
    {
        Vector512<float> under = Vector512.LessThan(x, Vector512.Create(-87.3f));
        x = Vector512.Max(x, Vector512.Create(-87.3f));
        Vector512<float> n = Vector512.Round(x * Vector512.Create(1.44269504088896341f));
        Vector512<float> r = Vector512.FusedMultiplyAdd(n, Vector512.Create(-0.693359375f), x);
        r = Vector512.FusedMultiplyAdd(n, Vector512.Create(2.12194440e-4f), r);
        Vector512<float> p = Vector512.Create(1.9875691500e-4f);
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(1.3981999507e-3f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(8.3334519073e-3f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(4.1665795894e-2f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(1.6666665459e-1f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(5.0000001201e-1f));
        p = Vector512.FusedMultiplyAdd(p * r, r, r) + Vector512<float>.One;
        Vector512<int> scale = (Vector512.ConvertToInt32(n) + Vector512.Create(127)) << 23;
        return Vector512.AndNot(p * scale.AsSingle(), under);
    }

    /// <inheritdoc cref="ExpNonPositive(Vector512{float})"/>
    private static Vector256<float> ExpNonPositive(Vector256<float> x)
    {
        Vector256<float> under = Vector256.LessThan(x, Vector256.Create(-87.3f));
        x = Vector256.Max(x, Vector256.Create(-87.3f));
        Vector256<float> n = Vector256.Round(x * Vector256.Create(1.44269504088896341f));
        Vector256<float> r = Vector256.FusedMultiplyAdd(n, Vector256.Create(-0.693359375f), x);
        r = Vector256.FusedMultiplyAdd(n, Vector256.Create(2.12194440e-4f), r);
        Vector256<float> p = Vector256.Create(1.9875691500e-4f);
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(1.3981999507e-3f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(8.3334519073e-3f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(4.1665795894e-2f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(1.6666665459e-1f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(5.0000001201e-1f));
        p = Vector256.FusedMultiplyAdd(p * r, r, r) + Vector256<float>.One;
        Vector256<int> scale = (Vector256.ConvertToInt32(n) + Vector256.Create(127)) << 23;
        return Vector256.AndNot(p * scale.AsSingle(), under);
    }
}
