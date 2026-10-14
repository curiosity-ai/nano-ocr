using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NanoOcr.Core;

namespace NanoOcr.Imaging;

/// <summary>Interpolation methods of <see cref="CvResize"/>, named after OpenCV's.</summary>
public enum Interpolation
{
    /// <summary><c>cv2.INTER_LINEAR</c>.</summary>
    Linear,

    /// <summary><c>cv2.INTER_AREA</c>.</summary>
    Area,
}

/// <summary>
/// A port of OpenCV's <c>cv::resize</c> for 8-bit single-channel images with
/// <c>INTER_LINEAR</c> and <c>INTER_AREA</c> — the two the Indra pipeline uses — reproducing its
/// integer and float arithmetic so the detector and recognizer see the bytes they were trained on.
/// </summary>
/// <remarks>
/// <para>Reference: <c>opencv/modules/imgproc/src/resize.cpp</c> (<c>cv::hal::resize</c>,
/// <c>HResizeLinear</c>, <c>VResizeLinear&lt;uchar,int,short,…&gt;</c>, <c>ResizeArea_Invoker</c>,
/// <c>resizeAreaFast_Invoker</c>, <c>computeResizeAreaTab</c>).</para>
/// <list type="bullet">
/// <item>Linear on 8 bits is fixed point: 11-bit coefficients, a horizontal pass into <c>int</c>,
/// and a vertical pass <c>((b0·(S0≫4))≫16 + (b1·(S1≫4))≫16 + 2)≫2</c>.</item>
/// <item>Area with a non-integer shrink is a float accumulation over a coverage table; with an
/// integer shrink it is a box average (<c>(a+b+c+d+2)≫2</c> at 2×).</item>
/// <item>Area when enlarging falls back to the linear kernel with area-style coefficients.</item>
/// <item>A linear shrink by exactly 2 is turned into an area shrink, as OpenCV does.</item>
/// </list>
/// </remarks>
public static class CvResize
{
    private const int CoefBits = 11;
    private const int CoefScale = 1 << CoefBits;

    /// <summary>Resizes to <paramref name="width"/> × <paramref name="height"/>.</summary>
    public static GrayImage Resize(GrayImage src, int width, int height, Interpolation interpolation)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (width == src.Width && height == src.Height)
        {
            return src.Clone();
        }

        var dst = new GrayImage(width, height);
        ResizeInto(src, dst, interpolation);
        return dst;
    }

    /// <summary>
    /// <see cref="Resize"/> into an existing image of the target size (a pooled one, say); every
    /// pixel of <paramref name="dst"/> is written.
    /// </summary>
    internal static void ResizeInto(GrayImage src, GrayImage dst, Interpolation interpolation)
    {
        int width = dst.Width, height = dst.Height;
        if (width == src.Width && height == src.Height)
        {
            src.Pixels.AsSpan(0, width * height).CopyTo(dst.Pixels);
            return;
        }

        double invScaleX = (double)width / src.Width;
        double invScaleY = (double)height / src.Height;
        double scaleX = 1.0 / invScaleX, scaleY = 1.0 / invScaleY;
        int iscaleX = RoundInt(scaleX), iscaleY = RoundInt(scaleY);
        const double DblEpsilon = 2.220446049250313e-16; // C's DBL_EPSILON (not .NET's double.Epsilon)
        bool areaFast = Math.Abs(scaleX - iscaleX) < DblEpsilon && Math.Abs(scaleY - iscaleY) < DblEpsilon;
        if (interpolation == Interpolation.Linear && areaFast && iscaleX == 2 && iscaleY == 2)
        {
            interpolation = Interpolation.Area;
        }

        if (interpolation == Interpolation.Area && scaleX >= 1 && scaleY >= 1)
        {
            if (areaFast)
            {
                AreaFast(src, dst, iscaleX, iscaleY);
            }
            else
            {
                Area(src, dst, scaleX, scaleY);
            }

            return;
        }

        Linear(src, dst, scaleX, scaleY, invScaleX, invScaleY, interpolation == Interpolation.Area);
    }

    /// <summary>OpenCV's <c>saturate_cast&lt;int&gt;(double)</c>: round half to even.</summary>
    private static int RoundInt(double v) => (int)Math.Round(v, MidpointRounding.ToEven);

    /// <summary>OpenCV's <c>saturate_cast&lt;uchar&gt;(float)</c>: round half to even, then clamp.</summary>
    internal static byte SaturateByte(float v)
    {
        if (float.IsNaN(v))
        {
            return 0;
        }

        int r = (int)MathF.Round(Math.Clamp(v, -1e9f, 1e9f), MidpointRounding.ToEven);
        return (byte)Math.Clamp(r, 0, 255);
    }

    private static void Linear(GrayImage src, GrayImage dst, double scaleX, double scaleY, double invScaleX, double invScaleY, bool areaMode)
    {
        int sw = src.Width, sh = src.Height, dw = dst.Width, dh = dst.Height;
        var xofs = new int[dw];
        var alpha = new short[dw * 2];
        int xmin = 0, xmax = dw;
        for (int dx = 0; dx < dw; dx++)
        {
            float fx;
            int sx;
            if (!areaMode)
            {
                fx = (float)(((dx + 0.5) * scaleX) - 0.5);
                sx = (int)MathF.Floor(fx);
                fx -= sx;
            }
            else
            {
                sx = (int)Math.Floor(dx * scaleX);
                fx = (float)((dx + 1) - ((sx + 1) * invScaleX));
                fx = fx <= 0 ? 0f : fx - MathF.Floor(fx);
            }

            if (sx < 0)
            {
                xmin = dx + 1;
                fx = 0;
                sx = 0;
            }

            if (sx + 1 >= sw)
            {
                xmax = Math.Min(xmax, dx);
                if (sx >= sw - 1)
                {
                    fx = 0;
                    sx = sw - 1;
                }
            }

            xofs[dx] = sx;
            alpha[dx * 2] = SaturateShort((1f - fx) * CoefScale);
            alpha[(dx * 2) + 1] = SaturateShort(fx * CoefScale);
        }

        var yofs = new int[dh];
        var beta = new short[dh * 2];
        for (int dy = 0; dy < dh; dy++)
        {
            float fy;
            int sy;
            if (!areaMode)
            {
                fy = (float)(((dy + 0.5) * scaleY) - 0.5);
                sy = (int)MathF.Floor(fy);
                fy -= sy;
            }
            else
            {
                sy = (int)Math.Floor(dy * scaleY);
                fy = (float)((dy + 1) - ((sy + 1) * invScaleY));
                fy = fy <= 0 ? 0f : fy - MathF.Floor(fy);
            }

            yofs[dy] = sy;
            beta[dy * 2] = SaturateShort((1f - fy) * CoefScale);
            beta[(dy * 2) + 1] = SaturateShort(fy * CoefScale);
        }

        byte[] s = src.Pixels, d = dst.Pixels;
        Parallelism.For(dh, (long)dw * dh * 4, (y0, y1) =>
        {
            var row0 = new int[dw];
            var row1 = new int[dw];
            for (int dy = y0; dy < y1; dy++)
            {
                int r0 = Clip(yofs[dy], sh), r1 = Clip(yofs[dy] + 1, sh);
                HResize(s.AsSpan(r0 * sw, sw), row0, xofs, alpha, xmax);
                if (r1 == r0)
                {
                    row0.AsSpan().CopyTo(row1);
                }
                else
                {
                    HResize(s.AsSpan(r1 * sw, sw), row1, xofs, alpha, xmax);
                }

                int b0 = beta[dy * 2], b1 = beta[(dy * 2) + 1];
                Span<byte> outRow = d.AsSpan(dy * dw, dw);
                for (int x = 0; x < dw; x++)
                {
                    outRow[x] = (byte)((((b0 * (row0[x] >> 4)) >> 16) + ((b1 * (row1[x] >> 4)) >> 16) + 2) >> 2);
                }
            }
        });
    }

    private static void HResize(ReadOnlySpan<byte> s, int[] d, int[] xofs, short[] alpha, int xmax)
    {
        int dx = 0;
        for (; dx < xmax; dx++)
        {
            int sx = xofs[dx];
            d[dx] = (s[sx] * alpha[dx * 2]) + (s[sx + 1] * alpha[(dx * 2) + 1]);
        }

        for (; dx < d.Length; dx++)
        {
            d[dx] = s[xofs[dx]] * CoefScale;
        }
    }

    private static int Clip(int v, int size) => v < 0 ? 0 : v >= size ? size - 1 : v;

    private static short SaturateShort(float v) => (short)Math.Clamp((int)MathF.Round(v, MidpointRounding.ToEven), short.MinValue, short.MaxValue);

    private readonly record struct Decimate(int Si, int Di, float Alpha);

    private static List<Decimate> AreaTable(int ssize, int dsize, double scale)
    {
        var tab = new List<Decimate>(ssize * 2);
        for (int dx = 0; dx < dsize; dx++)
        {
            double fsx1 = dx * scale;
            double fsx2 = fsx1 + scale;
            double cellWidth = Math.Min(scale, ssize - fsx1);
            int sx1 = (int)Math.Ceiling(fsx1), sx2 = (int)Math.Floor(fsx2);
            sx2 = Math.Min(sx2, ssize - 1);
            sx1 = Math.Min(sx1, sx2);
            if (sx1 - fsx1 > 1e-3)
            {
                tab.Add(new Decimate(sx1 - 1, dx, (float)((sx1 - fsx1) / cellWidth)));
            }

            for (int sx = sx1; sx < sx2; sx++)
            {
                tab.Add(new Decimate(sx, dx, (float)(1.0 / cellWidth)));
            }

            if (fsx2 - sx2 > 1e-3)
            {
                tab.Add(new Decimate(sx2, dx, (float)(Math.Min(Math.Min(fsx2 - sx2, 1.0), cellWidth) / cellWidth)));
            }
        }

        return tab;
    }

    private static void Area(GrayImage src, GrayImage dst, double scaleX, double scaleY)
    {
        Decimate[] xtab = AreaTable(src.Width, dst.Width, scaleX).ToArray();
        Decimate[] ytab = AreaTable(src.Height, dst.Height, scaleY).ToArray();
        var tabofs = new int[dst.Height + 1];
        for (int k = 0, dy = 0; k < ytab.Length; k++)
        {
            if (k == 0 || ytab[k].Di != ytab[k - 1].Di)
            {
                tabofs[dy++] = k;
            }
        }

        tabofs[dst.Height] = ytab.Length;
        int sw = src.Width, dw = dst.Width;

        // The horizontal table as a [taps, dw] grid: output column dx sums its entries in table
        // order; columns with fewer entries are padded with weight-0 taps at the end, which add
        // exactly +0 to a non-negative sum. One output per lane then accumulates in a register
        // (the original accumulated through memory, entry by entry).
        int taps = 1;
        for (int k = 0, run = 0; k < xtab.Length; k++)
        {
            run = k > 0 && xtab[k].Di == xtab[k - 1].Di ? run + 1 : 1;
            taps = Math.Max(taps, run);
        }

        var xi = new int[taps * dw];
        var xa = new float[taps * dw];
        for (int k = 0, slot = 0; k < xtab.Length; k++)
        {
            slot = k > 0 && xtab[k].Di == xtab[k - 1].Di ? slot + 1 : 0;
            xi[(slot * dw) + xtab[k].Di] = xtab[k].Si;
            xa[(slot * dw) + xtab[k].Di] = xtab[k].Alpha;
        }

        byte[] s = src.Pixels, d = dst.Pixels;
        Parallelism.For(dst.Height, (long)src.Width * src.Height * 2, (y0, y1) =>
        {
            float[] rowF = System.Buffers.ArrayPool<float>.Shared.Rent(sw);
            float[] buf = System.Buffers.ArrayPool<float>.Shared.Rent(dw);
            float[] sum = System.Buffers.ArrayPool<float>.Shared.Rent(dw);
            Array.Clear(sum, 0, dw); // the first source row of a chunk accumulates into it
            int jStart = tabofs[y0], jEnd = tabofs[y1];
            int prevDy = ytab[jStart].Di;
            for (int j = jStart; j < jEnd; j++)
            {
                float beta = ytab[j].Alpha;
                int dy = ytab[j].Di;
                Widen(s.AsSpan(ytab[j].Si * sw, sw), rowF);
                AreaRow(rowF, xi, xa, taps, dw, buf);
                if (dy != prevDy)
                {
                    StoreBytes(sum.AsSpan(0, dw), d.AsSpan(prevDy * dw, dw));
                    ScaleInto(buf, beta, sum, dw, accumulate: false);
                    prevDy = dy;
                }
                else
                {
                    ScaleInto(buf, beta, sum, dw, accumulate: true);
                }
            }

            StoreBytes(sum.AsSpan(0, dw), d.AsSpan(prevDy * dw, dw));
            System.Buffers.ArrayPool<float>.Shared.Return(rowF);
            System.Buffers.ArrayPool<float>.Shared.Return(buf);
            System.Buffers.ArrayPool<float>.Shared.Return(sum);
        });
    }

    /// <summary>Bytes to floats (exact).</summary>
    private static void Widen(ReadOnlySpan<byte> row, float[] dst)
    {
        int x = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; x + 16 <= row.Length; x += 16)
            {
                Vector128<byte> v = Vector128.Create(row.Slice(x, 16));
                (Vector128<ushort> lo, Vector128<ushort> hi) = Vector128.Widen(v);
                (Vector128<uint> a, Vector128<uint> b) = Vector128.Widen(lo);
                (Vector128<uint> c, Vector128<uint> e) = Vector128.Widen(hi);
                Vector128.ConvertToSingle(a.AsInt32()).CopyTo(dst, x);
                Vector128.ConvertToSingle(b.AsInt32()).CopyTo(dst, x + 4);
                Vector128.ConvertToSingle(c.AsInt32()).CopyTo(dst, x + 8);
                Vector128.ConvertToSingle(e.AsInt32()).CopyTo(dst, x + 12);
            }
        }

        for (; x < row.Length; x++)
        {
            dst[x] = row[x];
        }
    }

    /// <summary>One source row through the horizontal area table: <c>buf[dx] = Σ row[si]·α</c>, in table order.</summary>
    private static unsafe void AreaRow(float[] row, int[] xi, float[] xa, int taps, int dw, float[] buf)
    {
        int dx = 0;
        fixed (float* r = row)
        fixed (int* pi = xi)
        fixed (float* pa = xa)
        fixed (float* b = buf)
        {
            if (Avx2.IsSupported)
            {
                for (; dx + 8 <= dw; dx += 8)
                {
                    Vector256<float> acc = Vector256<float>.Zero;
                    for (int k = 0; k < taps; k++)
                    {
                        Vector256<int> idx = Avx.LoadVector256(pi + (k * dw) + dx);
                        Vector256<float> p = Avx2.GatherVector256(r, idx, 4);
                        acc = Avx.Add(acc, Avx.Multiply(p, Avx.LoadVector256(pa + (k * dw) + dx)));
                    }

                    Avx.Store(b + dx, acc);
                }
            }

            for (; dx < dw; dx++)
            {
                float acc = 0f;
                for (int k = 0; k < taps; k++)
                {
                    acc += r[pi[(k * dw) + dx]] * pa[(k * dw) + dx];
                }

                b[dx] = acc;
            }
        }
    }

    /// <summary><c>sum = β·buf</c> or <c>sum += β·buf</c>, multiply and add rounded separately.</summary>
    private static void ScaleInto(float[] buf, float beta, float[] sum, int n, bool accumulate)
    {
        int x = 0;
        var vb = new Vector<float>(beta);
        int vw = Vector<float>.Count;
        for (; x + vw <= n; x += vw)
        {
            Vector<float> p = vb * new Vector<float>(buf, x);
            (accumulate ? new Vector<float>(sum, x) + p : p).CopyTo(sum, x);
        }

        for (; x < n; x++)
        {
            float p = beta * buf[x];
            sum[x] = accumulate ? sum[x] + p : p;
        }
    }

    /// <summary><see cref="SaturateByte"/> over a row.</summary>
    private static void StoreBytes(ReadOnlySpan<float> sum, Span<byte> dst)
    {
        for (int x = 0; x < dst.Length; x++)
        {
            dst[x] = SaturateByte(sum[x]);
        }
    }

    private static void AreaFast(GrayImage src, GrayImage dst, int sx, int sy)
    {
        int sw = src.Width, sh = src.Height, dw = dst.Width;
        int area = sx * sy;
        float scale = 1f / area;
        int dwidth1 = sw / sx;
        byte[] s = src.Pixels, d = dst.Pixels;
        bool fast2 = sx == 2 && sy == 2;
        Parallelism.For(dst.Height, (long)sw * sh, (y0, y1) =>
        {
            for (int dy = y0; dy < y1; dy++)
            {
                Span<byte> row = d.AsSpan(dy * dw, dw);
                int sy0 = dy * sy;
                int w = sy0 + sy <= sh ? dwidth1 : 0;
                if (sy0 >= sh)
                {
                    row.Clear();
                    continue;
                }

                int dx = 0;
                for (; dx < w; dx++)
                {
                    int sx0 = dx * sx;
                    if (fast2)
                    {
                        int i0 = (sy0 * sw) + sx0, i1 = i0 + sw;
                        row[dx] = (byte)((s[i0] + s[i0 + 1] + s[i1] + s[i1 + 1] + 2) >> 2);
                        continue;
                    }

                    int sum = 0;
                    for (int yy = 0; yy < sy; yy++)
                    {
                        int b = ((sy0 + yy) * sw) + sx0;
                        for (int xx = 0; xx < sx; xx++)
                        {
                            sum += s[b + xx];
                        }
                    }

                    row[dx] = SaturateByte(sum * scale);
                }

                for (; dx < dw; dx++)
                {
                    int sum = 0, count = 0, sx0 = dx * sx;
                    for (int yy = 0; yy < sy && sy0 + yy < sh; yy++)
                    {
                        for (int xx = 0; xx < sx && sx0 + xx < sw; xx++)
                        {
                            sum += s[((sy0 + yy) * sw) + sx0 + xx];
                            count++;
                        }
                    }

                    row[dx] = count == 0 ? (byte)0 : SaturateByte((float)sum / count);
                }
            }
        });
    }
}
