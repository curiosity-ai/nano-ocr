using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NanoOcr.Core;

namespace NanoOcr.Imaging;

/// <summary>
/// OpenCV's <c>getRotationMatrix2D</c> and bilinear <c>warpAffine</c>, for the deskew.
/// </summary>
/// <remarks>
/// Ported from <c>opencv/modules/imgproc/src/warp_kernels.simd.hpp</c>
/// (<c>warpAffineLinearInvoker_8UC1</c> / <c>_32FC1</c>), the float-coordinate kernels OpenCV 4.11+
/// uses: the matrix is narrowed to float, a source position is <c>fma(M0, x, y·M1 + M2)</c> (the
/// row term unfused),
/// the weights are its fractional parts, and the blend is two horizontal <c>fma</c> lerps then a
/// vertical one, rounded half to even for 8-bit output.
/// </remarks>
public static class CvWarp
{

    /// <summary>The 2×3 matrix rotating by <paramref name="degrees"/> counter-clockwise about (cx, cy).</summary>
    public static double[] RotationMatrix(double cx, double cy, double degrees, double scale = 1.0)
    {
        double angle = degrees * Math.PI / 180.0;
        double alpha = Math.Cos(angle) * scale;
        double beta = Math.Sin(angle) * scale;
        return
        [
            alpha, beta, ((1 - alpha) * cx) - (beta * cy),
            -beta, alpha, (beta * cx) + ((1 - alpha) * cy),
        ];
    }

    /// <summary>Inverts a forward affine map the way <c>warpAffine</c> does without WARP_INVERSE_MAP.</summary>
    public static double[] Invert(double[] m)
    {
        double[] r = (double[])m.Clone();
        double d = (r[0] * r[4]) - (r[1] * r[3]);
        d = d != 0 ? 1.0 / d : 0;
        double a11 = r[4] * d, a22 = r[0] * d;
        r[0] = a11;
        r[1] *= -d;
        r[3] *= -d;
        r[4] = a22;
        double b1 = (-r[0] * r[2]) - (r[1] * r[5]);
        double b2 = (-r[3] * r[2]) - (r[4] * r[5]);
        r[2] = b1;
        r[5] = b2;
        return r;
    }

    /// <summary>
    /// <c>cv2.warpAffine(src, M, (w, h), flags=INTER_LINEAR, borderMode=BORDER_REPLICATE)</c>
    /// with the output the size of the input.
    /// </summary>
    public static GrayImage WarpReplicate(GrayImage src, double[] forward)
    {
        float[] m = Narrow(Invert(forward));
        int w = src.Width, h = src.Height;
        var dst = new GrayImage(w, h);
        byte[] s = src.Pixels, d = dst.Pixels;
        Parallelism.For(h, (long)w * h * 8, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                // The row offset is a plain multiply-add and the per-pixel step a fused one, as
                // OpenCV's vector kernel computes them; anything else moves ~0.006% of pixels by a level.
                float mx = (y * m[1]) + m[2];
                float my = (y * m[4]) + m[5];
                int row = y * w;
                for (int x = 0; x < w; x++)
                {
                    float sx = MathF.FusedMultiplyAdd(m[0], x, mx);
                    float sy = MathF.FusedMultiplyAdd(m[3], x, my);
                    int ix = (int)MathF.Floor(sx), iy = (int)MathF.Floor(sy);
                    float ax = sx - ix, ay = sy - iy;
                    float p00, p01, p10, p11;
                    if ((uint)ix < (uint)(w - 1) && (uint)iy < (uint)(h - 1))
                    {
                        int o = (iy * w) + ix;
                        p00 = s[o];
                        p01 = s[o + 1];
                        p10 = s[o + w];
                        p11 = s[o + w + 1];
                    }
                    else
                    {
                        int x0 = Math.Clamp(ix, 0, w - 1), x1 = Math.Clamp(ix + 1, 0, w - 1);
                        int r0 = Math.Clamp(iy, 0, h - 1) * w, r1 = Math.Clamp(iy + 1, 0, h - 1) * w;
                        p00 = s[r0 + x0];
                        p01 = s[r0 + x1];
                        p10 = s[r1 + x0];
                        p11 = s[r1 + x1];
                    }

                    float v0 = MathF.FusedMultiplyAdd(ax, p01 - p00, p00);
                    float v1 = MathF.FusedMultiplyAdd(ax, p11 - p10, p10);
                    d[row + x] = CvResize.SaturateByte(MathF.FusedMultiplyAdd(ay, v1 - v0, v0));
                }
            }
        });
        return dst;
    }

    /// <summary>
    /// <c>cv2.warpAffine(src, M, (w, h))</c> on a float image: bilinear, constant border 0, output
    /// the size of the input.
    /// </summary>
    public static float[] WarpConstantZero(float[] src, int w, int h, double[] forward, float[]? dst = null)
    {
        float[] m = Narrow(Invert(forward));
        dst ??= new float[w * h];
        float[] d = dst;
        for (int y = 0; y < h; y++)
        {
            float mx = (y * m[1]) + m[2];
            float my = (y * m[4]) + m[5];
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                float sx = MathF.FusedMultiplyAdd(m[0], x, mx);
                float sy = MathF.FusedMultiplyAdd(m[3], x, my);
                int ix = (int)MathF.Floor(sx), iy = (int)MathF.Floor(sy);
                float ax = sx - ix, ay = sy - iy;
                float p00, p01, p10, p11;
                if ((uint)ix < (uint)(w - 1) && (uint)iy < (uint)(h - 1))
                {
                    int o = (iy * w) + ix;
                    p00 = src[o];
                    p01 = src[o + 1];
                    p10 = src[o + w];
                    p11 = src[o + w + 1];
                }
                else
                {
                    if ((uint)(ix + 1) >= (uint)(w + 1) || (uint)(iy + 1) >= (uint)(h + 1))
                    {
                        d[row + x] = 0f;
                        continue;
                    }

                    p00 = Fetch(src, w, h, ix, iy);
                    p01 = Fetch(src, w, h, ix + 1, iy);
                    p10 = Fetch(src, w, h, ix, iy + 1);
                    p11 = Fetch(src, w, h, ix + 1, iy + 1);
                }

                float v0 = MathF.FusedMultiplyAdd(ax, p01 - p00, p00);
                float v1 = MathF.FusedMultiplyAdd(ax, p11 - p10, p10);
                d[row + x] = MathF.FusedMultiplyAdd(ay, v1 - v0, v0);
            }
        }

        return dst;

        static float Fetch(float[] s, int w, int h, int x, int y) =>
            (uint)x < (uint)w && (uint)y < (uint)h ? s[(y * w) + x] : 0f;
    }

    /// <summary>
    /// Row <paramref name="y"/> of <see cref="WarpConstantZero"/> into <paramref name="row"/> — the
    /// same arithmetic, one row at a time, so a caller can reduce each row while it is in cache.
    /// </summary>
    internal static void WarpConstantZeroRow(float[] src, int w, int h, float[] m, int y, Span<float> row)
    {
        float mx = (y * m[1]) + m[2];
        float my = (y * m[4]) + m[5];
        WarpRowRange(src, w, h, m, mx, my, row, 0, w);
    }

    /// <summary>
    /// <see cref="WarpConstantZeroRow"/> that skips what it can prove is zero: the output is cut
    /// into 32-pixel chunks, and a chunk whose bilinear taps all fall on zero source pixels (or
    /// outside the image) is written as zeros without being computed — exactly what the full
    /// computation gives, since every tap term is <c>0</c> then. <paramref name="nonzero"/> is the
    /// summed-area table of <c>src != 0</c> (<see cref="NonzeroTable"/>). Returns whether any
    /// chunk was computed; when none was, the row is all zeros.
    /// </summary>
    internal static bool WarpConstantZeroRowSparse(float[] src, int w, int h, float[] m, int y, Span<float> row, int[] nonzero)
    {
        const int Chunk = 32;
        float mx = (y * m[1]) + m[2];
        float my = (y * m[4]) + m[5];
        bool any = false;
        for (int x0 = 0; x0 < w; x0 += Chunk)
        {
            int x1 = Math.Min(w, x0 + Chunk);

            // The map is affine and the fused step is monotonic in x, so the chunk's source
            // positions lie between its two ends; one pixel of margin covers the second tap.
            float sxa = MathF.FusedMultiplyAdd(m[0], x0, mx), sxb = MathF.FusedMultiplyAdd(m[0], x1 - 1, mx);
            float sya = MathF.FusedMultiplyAdd(m[3], x0, my), syb = MathF.FusedMultiplyAdd(m[3], x1 - 1, my);
            int cx0 = Math.Max(0, (int)MathF.Floor(MathF.Min(sxa, sxb)) - 1);
            int cx1 = Math.Min(w, (int)MathF.Floor(MathF.Max(sxa, sxb)) + 3);
            int cy0 = Math.Max(0, (int)MathF.Floor(MathF.Min(sya, syb)) - 1);
            int cy1 = Math.Min(h, (int)MathF.Floor(MathF.Max(sya, syb)) + 3);
            if (cx0 >= cx1 || cy0 >= cy1 || CountNonzero(nonzero, w, cx0, cy0, cx1, cy1) == 0)
            {
                row.Slice(x0, x1 - x0).Clear();
                continue;
            }

            any = true;
            WarpRowRange(src, w, h, m, mx, my, row, x0, x1);
        }

        return any;
    }

    /// <summary>Summed-area table of <c>src != 0</c>, <c>(w + 1)·(h + 1)</c> entries.</summary>
    internal static int[] NonzeroTable(ReadOnlySpan<float> src, int w, int h)
    {
        var t = new int[(w + 1) * (h + 1)];
        for (int y = 0; y < h; y++)
        {
            int run = 0;
            ReadOnlySpan<float> r = src.Slice(y * w, w);
            int o = (y + 1) * (w + 1), up = y * (w + 1);
            for (int x = 0; x < w; x++)
            {
                run += r[x] != 0f ? 1 : 0;
                t[o + x + 1] = t[up + x + 1] + run;
            }
        }

        return t;
    }

    private static int CountNonzero(int[] t, int w, int x0, int y0, int x1, int y1)
    {
        int s = w + 1;
        return t[(y1 * s) + x1] - t[(y0 * s) + x1] - t[(y1 * s) + x0] + t[(y0 * s) + x0];
    }

    /// <summary>Columns <c>[x0, x1)</c> of one warped row.</summary>
    private static void WarpRowRange(float[] src, int w, int h, float[] m, float mx, float my, Span<float> row, int x0, int x1)
    {
        int x = Avx2.IsSupported && Fma.IsSupported ? GatherRow(src, w, h, m, mx, my, row, x0, x1) : x0;
        for (; x < x1; x++)
        {
            float sx = MathF.FusedMultiplyAdd(m[0], x, mx);
            float sy = MathF.FusedMultiplyAdd(m[3], x, my);
            float fx = MathF.Floor(sx), fy = MathF.Floor(sy);
            int ix = (int)fx, iy = (int)fy;
            float ax = sx - fx, ay = sy - fy;
            float p00, p01, p10, p11;
            if ((uint)ix < (uint)(w - 1) && (uint)iy < (uint)(h - 1))
            {
                int o = (iy * w) + ix;
                p00 = src[o];
                p01 = src[o + 1];
                p10 = src[o + w];
                p11 = src[o + w + 1];
            }
            else
            {
                if ((uint)(ix + 1) >= (uint)(w + 1) || (uint)(iy + 1) >= (uint)(h + 1))
                {
                    row[x] = 0f;
                    continue;
                }

                p00 = Fetch(src, w, h, ix, iy);
                p01 = Fetch(src, w, h, ix + 1, iy);
                p10 = Fetch(src, w, h, ix, iy + 1);
                p11 = Fetch(src, w, h, ix + 1, iy + 1);
            }

            float v0 = MathF.FusedMultiplyAdd(ax, p01 - p00, p00);
            float v1 = MathF.FusedMultiplyAdd(ax, p11 - p10, p10);
            row[x] = MathF.FusedMultiplyAdd(ay, v1 - v0, v0);
        }

        static float Fetch(float[] s, int w, int h, int x, int y) =>
            (uint)x < (uint)w && (uint)y < (uint)h ? s[(y * w) + x] : 0f;
    }

    /// <summary>
    /// Eight pixels at a time with AVX2 gathers, for the stretches where all four taps of all eight
    /// lie inside the image; returns the first column it did not do. The arithmetic per lane is the
    /// scalar path's (the same fused multiply-adds), so the output is identical.
    /// </summary>
    private static unsafe int GatherRow(float[] src, int w, int h, float[] m, float mx, float my, Span<float> row, int x0, int x1)
    {
        var m0 = Vector256.Create(m[0]);
        var m3 = Vector256.Create(m[3]);
        var vmx = Vector256.Create(mx);
        var vmy = Vector256.Create(my);
        var lanes = Vector256.Create(0f, 1, 2, 3, 4, 5, 6, 7);
        var wv = Vector256.Create(w);
        var limitX = Vector256.Create(w - 1);
        var limitY = Vector256.Create(h - 1);
        int x = x0;
        fixed (float* s = src)
        fixed (float* d = row)
        {
            for (; x + 8 <= x1; x += 8)
            {
                Vector256<float> xs = Vector256.Create((float)x) + lanes;
                Vector256<float> sx = Fma.MultiplyAdd(m0, xs, vmx);
                Vector256<float> sy = Fma.MultiplyAdd(m3, xs, vmy);
                Vector256<float> fx = Avx.Floor(sx), fy = Avx.Floor(sy);
                Vector256<int> ix = Avx.ConvertToVector256Int32WithTruncation(fx);
                Vector256<int> iy = Avx.ConvertToVector256Int32WithTruncation(fy);

                // (uint)ix < w - 1 and (uint)iy < h - 1 for every lane.
                Vector256<int> inX = Avx2.AndNot(Avx2.CompareGreaterThan(Vector256<int>.Zero, ix), Avx2.CompareGreaterThan(limitX, ix));
                Vector256<int> inY = Avx2.AndNot(Avx2.CompareGreaterThan(Vector256<int>.Zero, iy), Avx2.CompareGreaterThan(limitY, iy));
                if (Avx.MoveMask(Avx2.And(inX, inY).AsSingle()) != 0xFF)
                {
                    // Scalar for this group, then carry on vectorized.
                    for (int k = 0; k < 8; k++)
                    {
                        d[x + k] = ScalarPixel(src, w, h, m, mx, my, x + k);
                    }

                    continue;
                }

                Vector256<float> ax = sx - fx, ay = sy - fy;
                Vector256<int> o = Avx2.Add(Avx2.MultiplyLow(iy, wv), ix);
                Vector256<float> p00 = Avx2.GatherVector256(s, o, 4);
                Vector256<float> p01 = Avx2.GatherVector256(s + 1, o, 4);
                Vector256<float> p10 = Avx2.GatherVector256(s + w, o, 4);
                Vector256<float> p11 = Avx2.GatherVector256(s + w + 1, o, 4);
                Vector256<float> v0 = Fma.MultiplyAdd(ax, p01 - p00, p00);
                Vector256<float> v1 = Fma.MultiplyAdd(ax, p11 - p10, p10);
                Fma.MultiplyAdd(ay, v1 - v0, v0).Store(d + x);
            }
        }

        return x;
    }

    private static float ScalarPixel(float[] src, int w, int h, float[] m, float mx, float my, int x)
    {
        float sx = MathF.FusedMultiplyAdd(m[0], x, mx);
        float sy = MathF.FusedMultiplyAdd(m[3], x, my);
        float fx = MathF.Floor(sx), fy = MathF.Floor(sy);
        int ix = (int)fx, iy = (int)fy;
        float ax = sx - fx, ay = sy - fy;
        if ((uint)(ix + 1) >= (uint)(w + 1) || (uint)(iy + 1) >= (uint)(h + 1))
        {
            return 0f;
        }

        float p00 = Fetch(src, w, h, ix, iy), p01 = Fetch(src, w, h, ix + 1, iy);
        float p10 = Fetch(src, w, h, ix, iy + 1), p11 = Fetch(src, w, h, ix + 1, iy + 1);
        float v0 = MathF.FusedMultiplyAdd(ax, p01 - p00, p00);
        float v1 = MathF.FusedMultiplyAdd(ax, p11 - p10, p10);
        return MathF.FusedMultiplyAdd(ay, v1 - v0, v0);

        static float Fetch(float[] s, int w, int h, int x, int y) =>
            (uint)x < (uint)w && (uint)y < (uint)h ? s[(y * w) + x] : 0f;
    }

    /// <summary>The inverse map, narrowed to float, as the warp kernels use it.</summary>
    internal static float[] InverseFloat(double[] forward) => Narrow(Invert(forward));

    private static float[] Narrow(double[] m) => m.Select(v => (float)v).ToArray();
}
