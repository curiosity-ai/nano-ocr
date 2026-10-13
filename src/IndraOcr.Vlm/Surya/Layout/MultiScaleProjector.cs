using IndraOcr.Vlm.Core;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>
/// <c>MultiScaleProjector</c> of <c>surya/common/rfdetr/models/backbone/projector.py</c> at the one
/// scale the layout model uses (<c>projector_scale = ["P4"]</c>, factor 1.0): the four backbone maps
/// are concatenated on channels (1536), run through <c>C2f(1536, 256, n=3, layer_norm=True)</c> and a
/// final channel <c>LayerNorm</c>.
/// </summary>
/// <remarks>
/// Everything is channels-last <c>[H·W, C]</c>, where a 1×1 convolution is a plain GEMM and the
/// projector's channel <c>LayerNorm</c> (<c>F.layer_norm</c> over permuted channels, eps 1e-6) is an
/// ordinary row norm. The 3×3 convolutions of the bottlenecks (stride 1, padding 1, no bias) are a
/// GEMM over im2col rows whose columns follow the weight's <c>[c_in, ky, kx]</c> order.
/// </remarks>
internal sealed class MultiScaleProjector
{
    private const string Prefix = "backbone.0.projector.stages.0.";
    private readonly ConvX _cv1, _cv2;
    private readonly (ConvX A, ConvX B)[] _blocks;
    private readonly LayerNormLayer _norm;
    private readonly int _hidden;

    public MultiScaleProjector(StateDict state)
    {
        _cv1 = new ConvX(state, Prefix + "0.cv1", 1);
        _cv2 = new ConvX(state, Prefix + "0.cv2", 1);
        int n = 0;
        while (state.Contains($"{Prefix}0.m.{n}.cv1.conv.weight"))
        {
            n++;
        }

        _blocks = new (ConvX, ConvX)[n];
        for (int i = 0; i < n; i++)
        {
            _blocks[i] = (new ConvX(state, $"{Prefix}0.m.{i}.cv1", 3), new ConvX(state, $"{Prefix}0.m.{i}.cv2", 3));
        }

        _hidden = _cv1.Out / 2;
        _norm = state.Norm(Prefix + "1", 1e-6f);
    }

    /// <summary>Output channels.</summary>
    public int Channels => _cv2.Out;

    /// <summary>Projects the backbone's maps (each <c>[h·w, c]</c>) to one <c>[h·w, 256]</c> map.</summary>
    public float[] Forward(float[][] maps, int h, int w)
    {
        int pixels = h * w;
        int inC = maps.Sum(m => m.Length / pixels);
        float[] fused = new float[pixels * inC];
        for (int p = 0; p < pixels; p++)
        {
            int offset = 0;
            foreach (float[] m in maps)
            {
                int c = m.Length / pixels;
                m.AsSpan(p * c, c).CopyTo(fused.AsSpan((p * inC) + offset, c));
                offset += c;
            }
        }

        // C2f.forward: y = cv1(x).split((c, c)); y += [m(y[-1]) for m]; cv2(cat(y)).
        float[] y = _cv1.Forward(fused, h, w);
        int c2 = _cv1.Out;
        int total = (2 + _blocks.Length) * _hidden;
        float[] cat = new float[pixels * total];
        float[] last = new float[pixels * _hidden];
        for (int p = 0; p < pixels; p++)
        {
            y.AsSpan(p * c2, c2).CopyTo(cat.AsSpan(p * total, c2));
            y.AsSpan((p * c2) + _hidden, _hidden).CopyTo(last.AsSpan(p * _hidden, _hidden));
        }

        for (int b = 0; b < _blocks.Length; b++)
        {
            // Bottleneck with shortcut=False (C2f's default): cv2(cv1(x)).
            last = _blocks[b].B.Forward(_blocks[b].A.Forward(last, h, w), h, w);
            int offset = (2 + b) * _hidden;
            for (int p = 0; p < pixels; p++)
            {
                last.AsSpan(p * _hidden, _hidden).CopyTo(cat.AsSpan((p * total) + offset, _hidden));
            }
        }

        float[] output = _cv2.Forward(cat, h, w);
        _norm.Apply(output);
        return output;
    }

    /// <summary><c>ConvX</c>: convolution (no bias), channel <c>LayerNorm</c>, SiLU.</summary>
    private sealed class ConvX
    {
        private readonly Dense _conv;
        private readonly LayerNormLayer _norm;
        private readonly int _kernel;

        public ConvX(StateDict state, string prefix, int kernel)
        {
            _conv = state.Dense(prefix + ".conv", bias: false);
            _norm = state.Norm(prefix + ".bn", 1e-6f);
            _kernel = kernel;
        }

        public int Out => _conv.Out;

        public float[] Forward(float[] x, int h, int w)
        {
            int pixels = h * w;
            float[] y;
            if (_kernel == 1)
            {
                y = _conv.Forward(x, pixels);
            }
            else
            {
                int c = x.Length / pixels;
                int k = _kernel, pad = _kernel / 2, cols = c * k * k;
                float[] im2col = new float[pixels * cols];
                Parallel.For(0, pixels, Parallelism.Options, p =>
                {
                    int py = p / w, px = p % w;
                    Span<float> row = im2col.AsSpan(p * cols, cols);
                    for (int ky = 0; ky < k; ky++)
                    {
                        int sy = py + ky - pad;
                        if ((uint)sy >= (uint)h)
                        {
                            continue;
                        }

                        for (int kx = 0; kx < k; kx++)
                        {
                            int sx = px + kx - pad;
                            if ((uint)sx >= (uint)w)
                            {
                                continue;
                            }

                            ReadOnlySpan<float> src = x.AsSpan(((sy * w) + sx) * c, c);
                            int tap = (ky * k) + kx;
                            for (int ci = 0; ci < c; ci++)
                            {
                                row[(ci * k * k) + tap] = src[ci];
                            }
                        }
                    }
                });
                y = _conv.Forward(im2col, pixels);
            }

            _norm.Apply(y);
            Kernels.SiluParallel(y);
            return y;
        }
    }
}
