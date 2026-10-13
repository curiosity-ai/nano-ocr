using IndraOcr.Core;

namespace IndraOcr.Imaging;

/// <summary>
/// OpenCV's <c>getRotationMatrix2D</c> and bilinear <c>warpAffine</c>, for the deskew.
/// </summary>
/// <remarks>
/// Ported from <c>opencv/modules/imgproc/src/warp_kernels.simd.hpp</c>
/// (<c>warpAffineLinearInvoker_8UC1</c> / <c>_32FC1</c>), the float-coordinate kernels OpenCV 4.11+
/// uses: the matrix is narrowed to float, a source position is <c>fma(M0, x, y·M1 + M2)</c>,
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
                float mx = MathF.FusedMultiplyAdd(y, m[1], m[2]);
                float my = MathF.FusedMultiplyAdd(y, m[4], m[5]);
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
            float mx = MathF.FusedMultiplyAdd(y, m[1], m[2]);
            float my = MathF.FusedMultiplyAdd(y, m[4], m[5]);
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

    private static float[] Narrow(double[] m) => m.Select(v => (float)v).ToArray();
}
