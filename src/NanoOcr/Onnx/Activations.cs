using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NanoOcr.Onnx;

/// <summary>Scalar and vector activation functions the operators share.</summary>
internal static class Activations
{
    /// <summary><c>x · clamp(x/6 + 1/2, 0, 1)</c>, ONNX HardSwish.</summary>
    public static void HardSwish(ReadOnlySpan<float> x, Span<float> z)
    {
        int i = 0;
        if (Core.Gemm.Wide)
        {
            // The same three operations per element on 512-bit lanes: identical results.
            var sixth512 = Vector512.Create(1f / 6f);
            var half512 = Vector512.Create(0.5f);
            for (; i + 16 <= x.Length; i += 16)
            {
                var v = Vector512.Create(x.Slice(i, 16));
                Vector512<float> g = Vector512.Min(Vector512<float>.One, Vector512.Max(Vector512<float>.Zero, (v * sixth512) + half512));
                (v * g).CopyTo(z.Slice(i, 16));
            }
        }

        int vw = Vector<float>.Count;
        var sixth = new Vector<float>(1f / 6f);
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        for (; i + vw <= x.Length; i += vw)
        {
            var v = new Vector<float>(x.Slice(i, vw));
            Vector<float> g = Vector.Min(one, Vector.Max(Vector<float>.Zero, (v * sixth) + half));
            (v * g).CopyTo(z.Slice(i, vw));
        }

        for (; i < x.Length; i++)
        {
            z[i] = HardSwish(x[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HardSwish(float v) => v * MathF.Min(1f, MathF.Max(0f, (v * (1f / 6f)) + 0.5f));

    private static readonly double[] InverseOdd = Enumerable.Range(0, 256).Select(n => 1.0 / ((2 * n) + 1)).ToArray();

    /// <summary>
    /// The error function in single precision, evaluated in double through the everywhere-positive
    /// series <c>erf(x) = 2/√π · e^(−x²) · Σ 2ⁿ x^(2n+1) / (2n+1)!!</c>, which has no cancellation,
    /// then rounded. The series stops once a term is below 1e-10 of the sum — far under half a float
    /// ulp. Beyond |x| = 4, erfc is below half a float ulp of 1, so the result is ±1.
    /// </summary>
    public static float Erf(float xf)
    {
        double x = xf;
        double ax = Math.Abs(x);
        if (ax >= 4.0)
        {
            return x < 0 ? -1f : 1f;
        }

        double x2 = 2.0 * ax * ax;
        double term = ax;
        double sum = ax;
        for (int n = 1; n < InverseOdd.Length; n++)
        {
            term *= x2 * InverseOdd[n];
            sum += term;
            if (term < sum * 1e-10)
            {
                break;
            }
        }

        double r = 1.1283791670955126 * Math.Exp(-ax * ax) * sum;
        return (float)(x < 0 ? -r : r);
    }

    /// <summary><see cref="Erf(float)"/> over a span, four lanes of doubles at a time.</summary>
    public static void Erf(ReadOnlySpan<float> x, Span<float> z)
    {
        int i = 0;
        int vw = Vector<double>.Count;
        if (Vector.IsHardwareAccelerated && vw >= 2)
        {
            Span<double> lane = stackalloc double[vw];
            Span<double> sumOut = stackalloc double[vw];
            var tiny = new Vector<double>(1e-10);
            for (; i + vw <= x.Length; i += vw)
            {
                bool big = false;
                for (int k = 0; k < vw; k++)
                {
                    double v = Math.Abs((double)x[i + k]);
                    big |= v >= 4.0;
                    lane[k] = v;
                }

                if (big)
                {
                    for (int k = 0; k < vw; k++)
                    {
                        z[i + k] = Erf(x[i + k]);
                    }

                    continue;
                }

                var ax = new Vector<double>(lane);
                Vector<double> x2 = 2.0 * ax * ax;
                Vector<double> term = ax, sum = ax;
                for (int n = 1; n < InverseOdd.Length; n++)
                {
                    term *= x2 * InverseOdd[n];
                    sum += term;
                    if (Vector.LessThanAll(term, sum * tiny))
                    {
                        break;
                    }
                }

                sum.CopyTo(sumOut);
                for (int k = 0; k < vw; k++)
                {
                    double a = lane[k];
                    double r = 1.1283791670955126 * Math.Exp(-a * a) * sumOut[k];
                    z[i + k] = (float)(x[i + k] < 0 ? -r : r);
                }
            }
        }

        for (; i < x.Length; i++)
        {
            z[i] = Erf(x[i]);
        }
    }

    /// <summary>
    /// <c>softplus(x)</c>: <c>x + log(1 + e^−x)</c> for <c>x &gt; 0</c>, else <c>log(1 + e^x)</c> —
    /// written as <c>max(x, 0) + log(1 + e^−|x|)</c> (adding +0 is exact) and evaluated in
    /// blocks with the vectorized exp and log.
    /// </summary>
    public static void Softplus(ReadOnlySpan<float> x, Span<float> z)
    {
        Span<float> t = stackalloc float[256];
        for (int i = 0; i < x.Length; i += 256)
        {
            int n = Math.Min(256, x.Length - i);
            ReadOnlySpan<float> xs = x.Slice(i, n);
            Span<float> ts = t[..n];
            TensorPrimitives.Abs(xs, ts);
            TensorPrimitives.Negate(ts, ts);
            TensorPrimitives.Exp(ts, ts);
            TensorPrimitives.Add(ts, 1f, ts);
            TensorPrimitives.Log(ts, ts);
            Span<float> zs = z.Slice(i, n);
            TensorPrimitives.Max(xs, 0f, zs);
            TensorPrimitives.Add(zs, ts, zs);
        }
    }

    /// <summary>
    /// The error function in single precision, Abramowitz &amp; Stegun 7.1.26:
    /// <c>erf(x) = 1 − (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)·e^−x²</c>, <c>t = 1/(1 + p|x|)</c>, odd in x,
    /// absolute error under 1.5e-7 — a few float ulps, which is the class ONNX Runtime's own
    /// polynomial is in, at a tenth of the cost of the double series above. Eight lanes at a time.
    /// </summary>
    public static void ErfFast(ReadOnlySpan<float> x, Span<float> z)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 8 <= x.Length; i += 8)
            {
                Vector256<float> v = Vector256.Create(x.Slice(i, 8));
                Vector256<float> ax = Vector256.Abs(v);
                Vector256<float> t = Vector256<float>.One / Vector256.FusedMultiplyAdd(ax, Vector256.Create(0.3275911f), Vector256<float>.One);
                Vector256<float> p = Vector256.Create(1.061405429f);
                p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(-1.453152027f));
                p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(1.421413741f));
                p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(-0.284496736f));
                p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(0.254829592f));
                Vector256<float> r = Vector256<float>.One - (p * t * ExpNonPositive(-(ax * ax)));
                Vector256.ConditionalSelect(Vector256.LessThan(v, Vector256<float>.Zero), -r, r).CopyTo(z.Slice(i, 8));
            }
        }

        for (; i < x.Length; i++)
        {
            float ax = MathF.Abs(x[i]);
            float t = 1f / MathF.FusedMultiplyAdd(ax, 0.3275911f, 1f);
            float p = 1.061405429f;
            p = MathF.FusedMultiplyAdd(p, t, -1.453152027f);
            p = MathF.FusedMultiplyAdd(p, t, 1.421413741f);
            p = MathF.FusedMultiplyAdd(p, t, -0.284496736f);
            p = MathF.FusedMultiplyAdd(p, t, 0.254829592f);
            float r = 1f - (p * t * MathF.Exp(-(ax * ax)));
            z[i] = x[i] < 0 ? -r : r;
        }
    }

    /// <summary>
    /// <c>exp(x)</c> for <c>x ≤ 0</c> (Cephes' <c>expf</c>: <c>x = n·ln2 + r</c>, a degree-6
    /// polynomial in <c>r</c>, scaled by <c>2ⁿ</c>), within ~2 ulp of the true value; below −87
    /// it returns 0. About four times cheaper than <see cref="Vector256.Exp(Vector256{float})"/>,
    /// which evaluates in double. It only feeds the softmax peak, whose sum is already in an
    /// order of its own.
    /// </summary>
    internal static Vector512<float> ExpNonPositive(Vector512<float> x)
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
    internal static Vector256<float> ExpNonPositive(Vector256<float> x)
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
