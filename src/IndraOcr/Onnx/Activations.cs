using System.Numerics;
using System.Runtime.CompilerServices;

namespace IndraOcr.Onnx;

/// <summary>Scalar and vector activation functions the operators share.</summary>
internal static class Activations
{
    /// <summary><c>x · clamp(x/6 + 1/2, 0, 1)</c>, ONNX HardSwish.</summary>
    public static void HardSwish(ReadOnlySpan<float> x, Span<float> z)
    {
        int i = 0;
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

    /// <summary>
    /// The error function in single precision, evaluated in double through the everywhere-positive
    /// series <c>erf(x) = 2/√π · e^(−x²) · Σ 2ⁿ x^(2n+1) / (2n+1)!!</c>, which has no cancellation,
    /// then rounded. Beyond |x| = 4, erfc is below half a float ulp of 1, so the result is ±1.
    /// </summary>
    public static float Erf(float xf)
    {
        double x = xf;
        double ax = Math.Abs(x);
        if (ax >= 4.0)
        {
            return x < 0 ? -1f : 1f;
        }

        double x2 = ax * ax;
        double term = ax;
        double sum = ax;
        for (int n = 1; n < 200; n++)
        {
            term *= 2.0 * x2 / ((2 * n) + 1);
            sum += term;
            if (term < sum * 1e-17)
            {
                break;
            }
        }

        double r = 1.1283791670955126 * Math.Exp(-x2) * sum;
        return (float)(x < 0 ? -r : r);
    }
}
