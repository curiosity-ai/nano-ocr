using System.Numerics;
using System.Numerics.Tensors;
using IndraOcr.Vlm.Core;
using CoreGemm = IndraOcr.Core.Gemm;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>The activations EfficientViT's <c>ConvNormAct</c> layers use.</summary>
public enum Activation
{
    /// <summary><c>nn.Identity</c>.</summary>
    None,

    /// <summary><c>nn.Hardswish</c>: <c>x · min(max(x + 3, 0), 6) / 6</c>.</summary>
    Hardswish,

    /// <summary><c>nn.ReLU</c>.</summary>
    Relu,
}

/// <summary>
/// The float32 kernels behind the detector, on channels-last maps: convolution (im2col and the
/// packed GEMM of <c>IndraOcr.Core</c>), depthwise and grouped convolution, the per-channel affine
/// a frozen batch norm reduces to, the activations, and <c>F.interpolate(mode="bilinear",
/// align_corners=False)</c>.
/// </summary>
internal static class ConvKernels
{
    /// <summary>Floats of im2col scratch one band of output rows may use (16 MiB).</summary>
    private const int Im2ColBudget = 1 << 22;

    /// <summary>
    /// <c>nn.Conv2d(groups=1)</c>: <paramref name="weight"/> is <c>[k·k·Cin, Cout]</c>, ordered
    /// <c>(ky, kx, ci)</c> along the reduction to match the im2col rows.
    /// </summary>
    public static FeatureMap Convolution(FeatureMap x, float[] weight, int outChannels, int kernel, int stride, int padding)
    {
        int outH = ((x.Height + (2 * padding) - kernel) / stride) + 1;
        int outW = ((x.Width + (2 * padding) - kernel) / stride) + 1;
        FeatureMap y = FeatureMap.Rent(outH, outW, outChannels);
        int cin = x.Channels;

        if (kernel == 1 && stride == 1 && padding == 0)
        {
            CoreGemm.Multiply(x.Pixels, outChannels, cin, x.Span, cin, weight, outChannels, y.Span, outChannels);
            return y;
        }

        int k = kernel * kernel * cin;
        int bandRows = Math.Clamp(Im2ColBudget / Math.Max(1, outW * k), 1, outH);
        float[] columns = TensorPool.RentArray(bandRows * outW * k);
        try
        {
            for (int r0 = 0; r0 < outH; r0 += bandRows)
            {
                int rows = Math.Min(bandRows, outH - r0);
                Im2Col(x, kernel, stride, padding, outW, r0, rows, columns);
                CoreGemm.Multiply(
                    rows * outW, outChannels, k,
                    columns.AsSpan(0, rows * outW * k), k,
                    weight, outChannels,
                    y.Array.AsSpan(r0 * outW * outChannels, rows * outW * outChannels), outChannels);
            }
        }
        finally
        {
            TensorPool.Return(columns);
        }

        return y;
    }

    private static void Im2Col(FeatureMap x, int kernel, int stride, int padding, int outW, int r0, int rows, float[] columns)
    {
        int cin = x.Channels;
        int k = kernel * kernel * cin;
        int inH = x.Height;
        int inW = x.Width;
        float[] input = x.Array;
        Parallel.For(0, rows, Parallelism.Options, r =>
        {
            int oy = r0 + r;
            Span<float> rowColumns = columns.AsSpan(r * outW * k, outW * k);
            for (int ox = 0; ox < outW; ox++)
            {
                Span<float> dst = rowColumns.Slice(ox * k, k);
                int t = 0;
                for (int ky = 0; ky < kernel; ky++)
                {
                    int iy = (oy * stride) - padding + ky;
                    for (int kx = 0; kx < kernel; kx++, t += cin)
                    {
                        int ix = (ox * stride) - padding + kx;
                        if ((uint)iy < (uint)inH && (uint)ix < (uint)inW)
                        {
                            input.AsSpan(((iy * inW) + ix) * cin, cin).CopyTo(dst.Slice(t, cin));
                        }
                        else
                        {
                            dst.Slice(t, cin).Clear();
                        }
                    }
                }
            }
        });
    }

    /// <summary>
    /// <c>nn.Conv2d(groups=C)</c> with a <c>[k·k, C]</c> weight, accumulating each output over the
    /// window in a register across a vector of channels.
    /// </summary>
    public static FeatureMap Depthwise(FeatureMap x, float[] weight, int kernel, int stride, int padding)
    {
        int c = x.Channels;
        int outH = ((x.Height + (2 * padding) - kernel) / stride) + 1;
        int outW = ((x.Width + (2 * padding) - kernel) / stride) + 1;
        FeatureMap y = FeatureMap.Rent(outH, outW, c);
        int inH = x.Height;
        int inW = x.Width;
        float[] input = x.Array;
        float[] output = y.Array;
        int vw = Vector<float>.Count;

        Parallel.For(0, outH, Parallelism.Options, oy =>
        {
            int ky0 = Math.Max(0, padding - (oy * stride));
            int ky1 = Math.Min(kernel, inH + padding - (oy * stride));
            for (int ox = 0; ox < outW; ox++)
            {
                int kx0 = Math.Max(0, padding - (ox * stride));
                int kx1 = Math.Min(kernel, inW + padding - (ox * stride));
                Span<float> dst = output.AsSpan(((oy * outW) + ox) * c, c);
                int ch = 0;
                for (; ch + vw <= c; ch += vw)
                {
                    Vector<float> acc = Vector<float>.Zero;
                    for (int ky = ky0; ky < ky1; ky++)
                    {
                        int iy = (oy * stride) - padding + ky;
                        for (int kx = kx0; kx < kx1; kx++)
                        {
                            int ix = (ox * stride) - padding + kx;
                            var v = new Vector<float>(input, (((iy * inW) + ix) * c) + ch);
                            var w = new Vector<float>(weight, (((ky * kernel) + kx) * c) + ch);
                            acc += v * w;
                        }
                    }

                    acc.CopyTo(dst.Slice(ch));
                }

                for (; ch < c; ch++)
                {
                    float acc = 0f;
                    for (int ky = ky0; ky < ky1; ky++)
                    {
                        int iy = (oy * stride) - padding + ky;
                        for (int kx = kx0; kx < kx1; kx++)
                        {
                            int ix = (ox * stride) - padding + kx;
                            acc += input[(((iy * inW) + ix) * c) + ch] * weight[(((ky * kernel) + kx) * c) + ch];
                        }
                    }

                    dst[ch] = acc;
                }
            }
        });

        return y;
    }

    /// <summary>
    /// A 1×1 <c>nn.Conv2d</c> with <paramref name="groups"/> groups: each group's
    /// <c>[pixels, Cin/g]</c> slice times its <c>[Cin/g, Cout/g]</c> block of
    /// <paramref name="weight"/> (stored <c>[g, Cin/g, Cout/g]</c>).
    /// </summary>
    public static FeatureMap GroupedPointwise(FeatureMap x, float[] weight, int outChannels, int groups)
    {
        int cin = x.Channels;
        int gin = cin / groups;
        int gout = outChannels / groups;
        FeatureMap y = FeatureMap.Rent(x.Height, x.Width, outChannels);
        int pixels = x.Pixels;
        float[] input = x.Array;
        float[] output = y.Array;
        Parallel.For(0, groups, Parallelism.Options, g =>
        {
            ReadOnlySpan<float> a = input.AsSpan(g * gin, ((pixels - 1) * cin) + gin);
            ReadOnlySpan<float> b = weight.AsSpan(g * gin * gout, gin * gout);
            Span<float> o = output.AsSpan(g * gout, ((pixels - 1) * outChannels) + gout);
            using (IndraOcr.Core.Parallelism.Use(1))
            {
                CoreGemm.Multiply(pixels, gout, gin, a, cin, b, gout, o, outChannels);
            }
        });
        return y;
    }

    /// <summary>
    /// <c>y = act(x · scale[c] + shift[c])</c> in place — a frozen <c>BatchNorm2d</c> as PyTorch's
    /// CPU kernel evaluates it, or a convolution bias when <paramref name="scale"/> is null.
    /// </summary>
    public static void Affine(FeatureMap x, float[]? scale, float[]? shift, Activation activation)
    {
        if (scale is null && shift is null && activation == Activation.None)
        {
            return;
        }

        int c = x.Channels;
        int pixels = x.Pixels;
        float[] data = x.Array;
        int chunk = Math.Max(1, (1 << 16) / c);
        int chunks = (pixels + chunk - 1) / chunk;
        Parallel.For(0, chunks, Parallelism.Options, i =>
        {
            int p0 = i * chunk;
            int p1 = Math.Min(pixels, p0 + chunk);
            for (int p = p0; p < p1; p++)
            {
                Span<float> row = data.AsSpan(p * c, c);
                if (scale is not null)
                {
                    TensorPrimitives.MultiplyAdd(row, scale, shift!, row);
                }
                else if (shift is not null)
                {
                    TensorPrimitives.Add(row, shift, row);
                }

                Activate(row, activation);
            }
        });
    }

    /// <summary>Applies <paramref name="activation"/> in place.</summary>
    public static void Activate(Span<float> values, Activation activation)
    {
        switch (activation)
        {
            case Activation.Relu:
                TensorPrimitives.Max(values, 0f, values);
                break;
            case Activation.Hardswish:
                Hardswish(values);
                break;
        }
    }

    /// <summary>PyTorch's <c>hardswish</c>: <c>x · min(max(x + 3, 0), 6) / 6</c>.</summary>
    public static void Hardswish(Span<float> values)
    {
        int vw = Vector<float>.Count;
        var three = new Vector<float>(3f);
        var six = new Vector<float>(6f);
        int i = 0;
        for (; i + vw <= values.Length; i += vw)
        {
            var x = new Vector<float>(values.Slice(i));
            Vector<float> gate = Vector.Min(Vector.Max(x + three, Vector<float>.Zero), six);
            (x * gate / six).CopyTo(values.Slice(i));
        }

        for (; i < values.Length; i++)
        {
            float x = values[i];
            values[i] = x * MathF.Min(MathF.Max(x + 3f, 0f), 6f) / 6f;
        }
    }

    /// <summary>
    /// Source index and weights of <c>upsample_bilinear2d(align_corners=False)</c> along one axis
    /// (ATen's <c>compute_indices_weights_linear</c>): <c>src = max(scale·(dst + ½) − ½, 0)</c> in
    /// float32 with <c>scale = in / out</c>.
    /// </summary>
    public static (int[] Index0, int[] Index1, float[] Lambda0, float[] Lambda1) LinearTaps(int inSize, int outSize)
    {
        var i0 = new int[outSize];
        var i1 = new int[outSize];
        var l0 = new float[outSize];
        var l1 = new float[outSize];
        float scale = (float)inSize / outSize;
        for (int d = 0; d < outSize; d++)
        {
            float src = (scale * (d + 0.5f)) - 0.5f;
            if (src < 0f)
            {
                src = 0f;
            }

            int index = Math.Min((int)MathF.Floor(src), inSize - 1);
            float lambda = Math.Min(Math.Max(src - index, 0f), 1f);
            i0[d] = index;
            i1[d] = index + (index < inSize - 1 ? 1 : 0);
            l1[d] = lambda;
            l0[d] = 1f - lambda;
        }

        return (i0, i1, l0, l1);
    }

    /// <summary>
    /// Bilinear resize of one channel plane, <c>F.interpolate(mode="bilinear",
    /// align_corners=False)</c>: <c>h0·(w0·a + w1·b) + h1·(w0·c + w1·d)</c>.
    /// </summary>
    public static void ResizePlane(float[] input, int offset, int inH, int inW, float[] output, int outH, int outW)
    {
        (int[] y0, int[] y1, float[] h0, float[] h1) = LinearTaps(inH, outH);
        (int[] x0, int[] x1, float[] w0, float[] w1) = LinearTaps(inW, outW);
        Parallel.For(0, outH, Parallelism.Options, y =>
        {
            ReadOnlySpan<float> top = input.AsSpan(offset + (y0[y] * inW), inW);
            ReadOnlySpan<float> bottom = input.AsSpan(offset + (y1[y] * inW), inW);
            Span<float> row = output.AsSpan(y * outW, outW);
            float a = h0[y];
            float b = h1[y];
            for (int x = 0; x < outW; x++)
            {
                row[x] = (a * ((w0[x] * top[x0[x]]) + (w1[x] * top[x1[x]])))
                    + (b * ((w0[x] * bottom[x0[x]]) + (w1[x] * bottom[x1[x]])));
            }
        });
    }
}
