using System.Runtime.Intrinsics;

namespace NanoOcr.Onnx;

internal static partial class Operators
{
    /// <summary>Domain of the operators the fusion pass introduces; not part of any ONNX file.</summary>
    internal const string FusedDomain = "nanoocr";

    private static readonly Dictionary<string, Kernel> FusedTable = new(StringComparer.Ordinal)
    {
        ["SoftmaxPeak"] = (n, x) => SoftmaxPeak(x[0]!),
    };

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
            int[] idx = new int[t];
            for (int item = b0; item < b1; item++)
            {
                int o = item * t;
                Span<float> m = mx.AsSpan(o, t), s = sm.AsSpan(o, t);
                ReadOnlySpan<float> rows = src.AsSpan(item * c * t, c * t);
                rows[..t].CopyTo(m);
                Array.Clear(idx);
                for (int k = 1; k < c; k++)
                {
                    MaxRow(rows.Slice(k * t, t), m, idx, k);
                }

                s.Clear();
                for (int k = 0; k < c; k++)
                {
                    ExpSumRow(rows.Slice(k * t, t), m, s);
                }

                for (int i = 0; i < t; i++)
                {
                    am[o + i] = idx[i];
                }
            }
        });
        return [max, arg, sum];
    }

    /// <summary><c>if (row[i] &gt; m[i]) { m[i] = row[i]; idx[i] = k; }</c>: strictly greater, so the first maximum wins.</summary>
    private static void MaxRow(ReadOnlySpan<float> row, Span<float> m, Span<int> idx, int k)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var kv = Vector256.Create(k);
            for (; i + 8 <= row.Length; i += 8)
            {
                var v = Vector256.Create(row.Slice(i, 8));
                var cur = Vector256.Create((ReadOnlySpan<float>)m.Slice(i, 8));
                Vector256<float> gt = Vector256.GreaterThan(v, cur);
                Vector256.ConditionalSelect(gt, v, cur).CopyTo(m.Slice(i, 8));
                Vector256.ConditionalSelect(gt.AsInt32(), kv, Vector256.Create((ReadOnlySpan<int>)idx.Slice(i, 8))).CopyTo(idx.Slice(i, 8));
            }
        }

        for (; i < row.Length; i++)
        {
            if (row[i] > m[i])
            {
                m[i] = row[i];
                idx[i] = k;
            }
        }
    }

    /// <summary><c>s[i] += exp(row[i] − m[i])</c>.</summary>
    private static void ExpSumRow(ReadOnlySpan<float> row, ReadOnlySpan<float> m, Span<float> s)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 8 <= row.Length; i += 8)
            {
                Vector256<float> e = Vector256.Exp(Vector256.Create(row.Slice(i, 8)) - Vector256.Create(m.Slice(i, 8)));
                (Vector256.Create((ReadOnlySpan<float>)s.Slice(i, 8)) + e).CopyTo(s.Slice(i, 8));
            }
        }

        for (; i < row.Length; i++)
        {
            s[i] += MathF.Exp(row[i] - m[i]);
        }
    }
}
