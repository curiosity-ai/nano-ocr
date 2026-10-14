using System.Globalization;
using System.Numerics;
using System.Text;

namespace NanoOcr.Pipeline;

/// <summary>
/// Python and numpy semantics the port has to reproduce exactly: banker's rounding, <c>round(x, n)</c>
/// on the exact binary value, numpy's float32 pairwise summation and median, and the Unicode-aware
/// <c>str</c> predicates over code points rather than UTF-16 units.
/// </summary>
internal static class Py
{
    /// <summary>Python's <c>round(x)</c>: nearest integer, ties to even.</summary>
    public static long Round(double x) => (long)Math.Round(x, MidpointRounding.ToEven);

    /// <summary>
    /// Python's <c>round(x, digits)</c>, which rounds the exact binary value of <paramref name="x"/>
    /// (so <c>round(2.675, 2) == 2.67</c>) half to even, then reads the decimal back.
    /// </summary>
    public static double Round(double x, int digits)
    {
        if (double.IsNaN(x) || double.IsInfinity(x) || x == 0)
        {
            return x;
        }

        long bits = BitConverter.DoubleToInt64Bits(x);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0)
        {
            exponent = 1;
        }
        else
        {
            mantissa |= 1L << 52;
        }

        exponent -= 1075;
        BigInteger scaled = new BigInteger(mantissa) * BigInteger.Pow(10, digits);
        BigInteger q;
        if (exponent >= 0)
        {
            q = scaled << exponent;
        }
        else
        {
            BigInteger den = BigInteger.One << -exponent;
            q = BigInteger.DivRem(scaled, den, out BigInteger r);
            BigInteger twice = r * 2;
            if (twice > den || (twice == den && !q.IsEven))
            {
                q += 1;
            }
        }

        double result = double.Parse($"{q}e-{digits}", CultureInfo.InvariantCulture);
        return negative ? -result : result;
    }

    /// <summary>numpy's float32 pairwise summation (<c>np.add.reduce</c> over a contiguous run).</summary>
    public static float PairwiseSum(ReadOnlySpan<float> a)
    {
        int n = a.Length;
        if (n < 8)
        {
            float res = -0f;
            for (int i = 0; i < n; i++)
            {
                res += a[i];
            }

            return n == 0 ? 0f : res;
        }

        if (n <= 128)
        {
            float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
            int i;
            for (i = 8; i < n - (n % 8); i += 8)
            {
                r0 += a[i];
                r1 += a[i + 1];
                r2 += a[i + 2];
                r3 += a[i + 3];
                r4 += a[i + 4];
                r5 += a[i + 5];
                r6 += a[i + 6];
                r7 += a[i + 7];
            }

            float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++)
            {
                res += a[i];
            }

            return res;
        }

        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a[..n2]) + PairwiseSum(a[n2..]);
    }

    /// <summary><c>np.mean</c> of a float32 array: pairwise sum, then divided in float32.</summary>
    public static float Mean(ReadOnlySpan<float> a) => PairwiseSum(a) / a.Length;

    /// <summary><c>np.median</c> of a float32 array (mean of the two middle values in float32).</summary>
    public static float Median(ReadOnlySpan<float> a)
    {
        float[] s = a.ToArray();
        Array.Sort(s);
        int n = s.Length;
        if (n == 0)
        {
            return float.NaN;
        }

        if (n % 2 == 1)
        {
            return s[n / 2];
        }

        // np.mean of the two middle values: (a + b) / 2 in float32.
        return (s[(n / 2) - 1] + s[n / 2]) / 2f;
    }

    /// <summary><c>np.median</c> of a float64 array.</summary>
    public static double Median(ReadOnlySpan<double> a)
    {
        double[] s = a.ToArray();
        Array.Sort(s);
        int n = s.Length;
        return n == 0 ? double.NaN : n % 2 == 1 ? s[n / 2] : (s[(n / 2) - 1] + s[n / 2]) / 2.0;
    }

    /// <summary>The code points of a string (Python indexes strings by code point).</summary>
    public static int[] CodePoints(string s)
    {
        var r = new List<int>(s.Length);
        foreach (Rune rune in s.EnumerateRunes())
        {
            r.Add(rune.Value);
        }

        return [.. r];
    }

    /// <summary>A string from code points.</summary>
    public static string FromCodePoints(IEnumerable<int> cps)
    {
        var sb = new StringBuilder();
        foreach (int c in cps)
        {
            sb.Append(char.ConvertFromUtf32(c));
        }

        return sb.ToString();
    }

    /// <summary>Python's <c>str.isspace()</c> for one code point.</summary>
    public static bool IsSpace(int c) =>
        c is ' ' or '\t' or '\n' or '\r' or '\v' or '\f' or 0x1C or 0x1D or 0x1E or 0x1F or 0x85
        || (c > 0x7F && Rune.IsValid(c) && Rune.GetUnicodeCategory(new Rune(c)) is UnicodeCategory.SpaceSeparator
            or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator);

    /// <summary>Python's <c>str.isalpha()</c> for one code point (categories Lu, Ll, Lt, Lm, Lo).</summary>
    public static bool IsAlpha(int c) => Rune.IsValid(c) && Rune.IsLetter(new Rune(c));

    /// <summary>Python's <c>str.isdigit()</c> for one code point (decimal and other digits, e.g. "²").</summary>
    public static bool IsDigit(int c)
    {
        if (!Rune.IsValid(c))
        {
            return false;
        }

        var r = new Rune(c);
        if (Rune.GetUnicodeCategory(r) == UnicodeCategory.DecimalDigitNumber)
        {
            return true;
        }

        return Rune.GetUnicodeCategory(r) == UnicodeCategory.OtherNumber && CharUnicodeInfo.GetDigitValue(r.ToString(), 0) >= 0;
    }

    /// <summary>Python's <c>str.isupper()</c> for one code point.</summary>
    public static bool IsUpper(int c) => Rune.IsValid(c) && Rune.IsUpper(new Rune(c));

    /// <summary>Python's <c>str.islower()</c> for one code point.</summary>
    public static bool IsLower(int c) => Rune.IsValid(c) && Rune.IsLower(new Rune(c));

    /// <summary>Python's <c>str.strip()</c>.</summary>
    public static string Strip(string s)
    {
        int[] cp = CodePoints(s);
        int a = 0, b = cp.Length;
        while (a < b && IsSpace(cp[a]))
        {
            a++;
        }

        while (b > a && IsSpace(cp[b - 1]))
        {
            b--;
        }

        return a == 0 && b == cp.Length ? s : FromCodePoints(cp[a..b]);
    }

    /// <summary>Python's <c>" ".join(s.split())</c>: whitespace runs collapsed, ends trimmed.</summary>
    public static string CollapseSpaces(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool pending = false;
        foreach (Rune r in s.EnumerateRunes())
        {
            if (IsSpace(r.Value))
            {
                pending = sb.Length > 0;
                continue;
            }

            if (pending)
            {
                sb.Append(' ');
                pending = false;
            }

            sb.Append(r.ToString());
        }

        return sb.ToString();
    }

    /// <summary>Python's <c>html.escape(s, quote=False)</c>.</summary>
    public static string HtmlEscape(string s) => s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

    /// <summary>A stable argsort.</summary>
    public static int[] ArgSort<T>(IReadOnlyList<T> keys)
        where T : IComparable<T>
    {
        int[] idx = Enumerable.Range(0, keys.Count).ToArray();
        return idx.OrderBy(i => keys[i]).ToArray();
    }

    /// <summary><c>np.argmax</c>: the first index of the largest value (NaN wins, as in numpy).</summary>
    public static int ArgMax(ReadOnlySpan<float> a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++)
        {
            if (float.IsNaN(a[best]))
            {
                return best;
            }

            if (a[i] > a[best] || float.IsNaN(a[i]))
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary><c>np.argmax</c> over doubles.</summary>
    public static int ArgMax(ReadOnlySpan<double> a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++)
        {
            if (double.IsNaN(a[best]))
            {
                return best;
            }

            if (a[i] > a[best] || double.IsNaN(a[i]))
            {
                best = i;
            }
        }

        return best;
    }
}

/// <summary>Double-precision counterparts of the numpy helpers.</summary>
internal static class PyDouble
{
    /// <summary>numpy's float64 pairwise summation.</summary>
    public static double PairwiseSum(ReadOnlySpan<double> a)
    {
        int n = a.Length;
        if (n < 8)
        {
            double res = 0;
            for (int i = 0; i < n; i++)
            {
                res += a[i];
            }

            return res;
        }

        if (n <= 128)
        {
            double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
            int i;
            for (i = 8; i < n - (n % 8); i += 8)
            {
                r0 += a[i];
                r1 += a[i + 1];
                r2 += a[i + 2];
                r3 += a[i + 3];
                r4 += a[i + 4];
                r5 += a[i + 5];
                r6 += a[i + 6];
                r7 += a[i + 7];
            }

            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++)
            {
                res += a[i];
            }

            return res;
        }

        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a[..n2]) + PairwiseSum(a[n2..]);
    }

    /// <summary><c>np.mean</c> of a float64 array.</summary>
    public static double Mean(ReadOnlySpan<double> a) => PairwiseSum(a) / a.Length;
}
