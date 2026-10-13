namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>
/// <c>torch.nn.functional.interpolate(x, size, mode="bilinear"|"bicubic", align_corners=False,
/// antialias=True)</c> on a float32 CPU tensor — ATen's <c>_upsample_bilinear2d_aa</c> and
/// <c>_upsample_bicubic2d_aa</c>, i.e. the generic separable path of
/// <c>aten/src/ATen/native/cpu/UpSampleKernel.cpp</c>.
/// </summary>
/// <remarks>
/// <para>
/// RF-DETR uses both: <c>torchvision.transforms.functional.resize</c> on the normalised float
/// image (bilinear — torchvision's default — with <c>antialias=True</c>), and DINOv2's
/// <c>interpolate_pos_encoding</c> (bicubic, antialiased) to fit the 44×44 position grid to the
/// patch grid. Neither is Pillow, and the uint8 path <c>PilResize.ResizeTorchBicubic</c> ports is a
/// different kernel again: this is the float one.
/// </para>
/// <para>
/// What makes it reproducible to the bit: the taps and weights are computed in <c>float</c> exactly
/// as <c>HelperInterpBase::_compute_indices_min_size_weights_aa</c> does (with its mix of double
/// and float intermediates), the width is resampled first into a float temporary and the height
/// second, and every output is the sum <c>Σ src[j]·w[j]</c> of <c>interpolate_separable_1d</c> with
/// the rounding sequence of the compiled kernel (see <see cref="Accumulate"/>). Bilinear is pinned
/// to the bit by <c>SuryaLayoutParityTests</c>; the bicubic filter polynomial itself may be
/// contracted differently by the compiler, so the position table is equal to float tolerance.
/// </para>
/// </remarks>
public static class TorchResize
{
    /// <summary>The reconstruction filter.</summary>
    public enum Mode
    {
        /// <summary>Triangle filter, support 1 (scaled by the downsampling factor).</summary>
        Bilinear,

        /// <summary>Keys cubic with <c>a = −0.5</c>, support 2 (scaled by the downsampling factor).</summary>
        Bicubic,
    }

    /// <summary>
    /// Resizes <paramref name="channels"/> planes of <paramref name="inH"/>×<paramref name="inW"/>
    /// floats (CHW) to <paramref name="outH"/>×<paramref name="outW"/>.
    /// </summary>
    public static float[] Resize(
        ReadOnlySpan<float> source, int channels, int inH, int inW, int outH, int outW, Mode mode)
    {
        float[] current = source[..(channels * inH * inW)].ToArray();
        if (outW != inW)
        {
            // Horizontal first: the contiguous dimension (UpSampleKernel.cpp's
            // separable_upsample_generic_Nd_kernel_impl).
            (int[] start, int[] size, float[] weights, int taps) = Weights(inW, outW, mode);
            float[] next = new float[channels * inH * outW];
            Parallel.For(0, channels * inH, Core.Parallelism.Options, row =>
            {
                ReadOnlySpan<float> src = current.AsSpan(row * inW, inW);
                Span<float> dst = next.AsSpan(row * outW, outW);
                for (int x = 0; x < outW; x++)
                {
                    dst[x] = Accumulate(src.Slice(start[x]), 1, weights.AsSpan(x * taps, size[x]));
                }
            });
            current = next;
        }

        if (outH != inH)
        {
            (int[] start, int[] size, float[] weights, int taps) = Weights(inH, outH, mode);
            float[] next = new float[channels * outH * outW];
            Parallel.For(0, channels * outH, Core.Parallelism.Options, index =>
            {
                int c = index / outH, y = index % outH;
                ReadOnlySpan<float> plane = current.AsSpan(c * inH * outW, inH * outW);
                Span<float> dst = next.AsSpan(index * outW, outW);
                ReadOnlySpan<float> w = weights.AsSpan(y * taps, size[y]);
                ReadOnlySpan<float> rows = plane.Slice(start[y] * outW);
                for (int x = 0; x < outW; x++)
                {
                    dst[x] = Accumulate(rows.Slice(x), outW, w);
                }
            });
            current = next;
        }

        return current;
    }

    /// <summary>
    /// <c>interpolate_separable_1d</c>: <c>Σ src[j·stride]·w[j]</c>, with the exact operation sequence
    /// of the shipped float kernel.
    /// </summary>
    /// <remarks>
    /// The source is a plain left-to-right <c>output += t * wts</c>, but what decides the bits is the
    /// machine code GCC made of it (read from <c>libtorch_cpu.so</c> 2.14's AVX512 build, and
    /// confirmed against <c>F.interpolate</c> on random data in both directions): the first product
    /// is rounded; with five or more taps the next taps go four at a time as a vector multiply whose
    /// rounded products are added in order; whatever is left — every tap after the first when there
    /// are at most four — is a fused multiply-add chain.
    /// </remarks>
    private static float Accumulate(ReadOnlySpan<float> src, int stride, ReadOnlySpan<float> w)
    {
        int n = w.Length;
        float acc = src[0] * w[0];
        int j = 1;
        if (n > 4)
        {
            int grouped = ((n - 1) & ~3) + 1;
            for (; j < grouped; j++)
            {
                float product = src[j * stride] * w[j];
                acc += product;
            }
        }

        for (; j < n; j++)
        {
            acc = MathF.FusedMultiplyAdd(src[j * stride], w[j], acc);
        }

        return acc;
    }

    /// <summary>
    /// <c>HelperInterpBase::_compute_index_ranges_weights</c> with <c>antialias=True</c>,
    /// <c>align_corners=False</c> and <c>scalar_t = float</c>.
    /// </summary>
    private static (int[] Start, int[] Size, float[] Weights, int Taps) Weights(int inSize, int outSize, Mode mode)
    {
        int interpSize = mode == Mode.Bilinear ? 2 : 4;

        // area_pixel_compute_scale<float>: static_cast<float>(input_size) / output_size.
        float scale = (float)inSize / outSize;
        float support = scale >= 1f ? (float)(interpSize * 0.5 * scale) : (float)(interpSize * 0.5);
        int taps = ((int)Math.Ceiling(support) * 2) + 1;
        float invscale = scale >= 1f ? (float)(1.0 / scale) : 1f;

        int[] start = new int[outSize];
        int[] size = new int[outSize];
        float[] weights = new float[outSize * taps];
        for (int i = 0; i < outSize; i++)
        {
            float center = (float)(scale * (i + 0.5));
            long xmin = Math.Max((long)(center - support + 0.5), 0);
            long xsize = Math.Min((long)(center + support + 0.5), inSize) - xmin;
            xsize = Math.Clamp(xsize, 0, taps);

            Span<float> w = weights.AsSpan(i * taps, taps);
            float total = 0f;
            for (int j = 0; j < xsize; j++)
            {
                float arg = (float)((((j + xmin) - center) + 0.5) * invscale);
                float value = mode == Mode.Bilinear ? Triangle(arg) : Cubic(arg);
                w[j] = value;
                total += value;
            }

            if (total != 0f)
            {
                for (int j = 0; j < xsize; j++)
                {
                    w[j] /= total;
                }
            }

            start[i] = (int)xmin;
            size[i] = (int)xsize;
        }

        return (start, size, weights, taps);
    }

    /// <summary><c>HelperInterpLinear::aa_filter</c>.</summary>
    private static float Triangle(float x)
    {
        x = Math.Abs(x);
        return x < 1f ? (float)(1.0 - x) : 0f;
    }

    /// <summary><c>HelperInterpCubic::aa_filter</c> (<c>a = −0.5</c>, float arithmetic).</summary>
    private static float Cubic(float x)
    {
        const float A = -0.5f;
        x = Math.Abs(x);
        if (x < 1f)
        {
            return (((A + 2f) * x) - (A + 3f)) * x * x + 1f;
        }

        if (x < 2f)
        {
            return ((((A * x) - (5f * A)) * x) + (8f * A)) * x - (4f * A);
        }

        return 0f;
    }
}
