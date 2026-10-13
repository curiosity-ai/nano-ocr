using IndraOcr.Vlm.Core;
using IndraOcr.Vlm.Models;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>
/// RF-DETR's encoder: DINOv2-small (12 layers, width 384, 6 heads, no registers) with windowed
/// attention — <c>WindowedDinov2WithRegistersBackbone</c> of
/// <c>surya/common/rfdetr/models/backbone/dinov2_with_windowed_attn.py</c>, as configured by
/// <c>DinoV2.__init__</c> in <c>dinov2.py</c>.
/// </summary>
/// <remarks>
/// <para>
/// The patch grid is cut into <c>num_windows × num_windows</c> windows, each of which carries its
/// own copy of the class token (<c>WindowedDinov2WithRegistersEmbeddings.forward</c>). Layers whose
/// index is an output index (3, 6, 9 — 12 is past the end) attend across all windows at once; the
/// others attend within a window. The hidden states after layers 3, 6, 9 and 12 go through the
/// shared final <c>layernorm</c>, lose their class tokens and are stitched back into one
/// <c>[H·W, 384]</c> map each (<c>WindowedDinov2WithRegistersBackbone.forward</c>).
/// </para>
/// <para>
/// The checkpoint's position table is a 44×44 grid (<c>positional_encoding_size</c>, i.e. a 704-pixel
/// input at patch 16); <c>interpolate_pos_encoding</c> resizes it to the actual patch grid with
/// antialiased bicubic interpolation, <see cref="TorchResize"/>.
/// </para>
/// </remarks>
internal sealed class WindowedDinoV2
{
    private const string Prefix = "backbone.0.encoder.encoder.";
    private const int Width = 384;
    private const int Heads = 6;
    private const int Patch = 16;
    private const float Eps = 1e-6f;

    private readonly Dense _patch;
    private readonly float[] _cls;
    private readonly float[] _positions;
    private readonly int _positionGrid;
    private readonly Layer[] _layers;
    private readonly LayerNormLayer _norm;
    private readonly int[] _outIndices = [3, 6, 9, 12];
    private readonly int _windows;
    private readonly Dictionary<(int, int), float[]> _interpolated = [];

    public WindowedDinoV2(StateDict state, int windows)
    {
        _windows = windows;
        _patch = state.Dense(Prefix + "embeddings.patch_embeddings.projection");
        _cls = state.Floats(Prefix + "embeddings.cls_token");
        _positions = state.Floats(Prefix + "embeddings.position_embeddings");
        _positionGrid = (int)Math.Sqrt((_positions.Length / Width) - 1);
        int count = 0;
        while (state.Contains($"{Prefix}encoder.layer.{count}.norm1.weight"))
        {
            count++;
        }

        _layers = new Layer[count];
        for (int i = 0; i < count; i++)
        {
            _layers[i] = new Layer(state, $"{Prefix}encoder.layer.{i}.");
        }

        _norm = state.Norm(Prefix + "layernorm", Eps);
    }

    /// <summary>Feature channels of each output map.</summary>
    public static int Channels => Width;

    /// <summary>
    /// Runs the backbone over a normalised <c>[3, H, W]</c> image; returns the four feature maps,
    /// each channels-last <c>[H/16 · W/16, 384]</c>.
    /// </summary>
    public float[][] Forward(float[] pixels, int height, int width, LayoutTrace? trace)
    {
        int gh = height / Patch, gw = width / Patch;
        int tokens = gh * gw;
        if (gh % _windows != 0 || gw % _windows != 0)
        {
            throw new ArgumentException($"The patch grid {gh}x{gw} is not divisible into {_windows}x{_windows} windows.");
        }

        // Patch embedding: a 16x16 stride-16 convolution, posed as a GEMM over im2col rows whose
        // columns follow the weight's [c, ky, kx] order.
        int k = 3 * Patch * Patch;
        float[] columns = new float[tokens * k];
        Parallel.For(0, tokens, Parallelism.Options, t =>
        {
            int py = t / gw, px = t % gw;
            Span<float> row = columns.AsSpan(t * k, k);
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < Patch; y++)
                {
                    pixels.AsSpan((c * height * width) + (((py * Patch) + y) * width) + (px * Patch), Patch)
                        .CopyTo(row.Slice((c * Patch * Patch) + (y * Patch), Patch));
                }
            }
        });
        float[] patches = _patch.Forward(columns, tokens);

        float[] pos = PositionEmbedding(gh, gw);
        trace?.Invoke("bb.pos_embed", pos);
        LayoutMath.Add(patches, pos);

        // Windows: [windows², 1 + tokens/windows², C], the class token (plus its position) first.
        int windows2 = _windows * _windows;
        int wh = gh / _windows, ww = gw / _windows;
        int perWindow = 1 + (wh * ww);
        float[] hidden = new float[windows2 * perWindow * Width];
        for (int w = 0; w < windows2; w++)
        {
            Span<float> cls = hidden.AsSpan(w * perWindow * Width, Width);
            for (int c = 0; c < Width; c++)
            {
                cls[c] = _cls[c] + _positions[c];
            }

            int wr = w / _windows, wc = w % _windows;
            for (int r = 0; r < wh; r++)
            {
                for (int col = 0; col < ww; col++)
                {
                    int source = (((wr * wh) + r) * gw) + (wc * ww) + col;
                    patches.AsSpan(source * Width, Width)
                        .CopyTo(hidden.AsSpan(((w * perWindow) + 1 + (r * ww) + col) * Width, Width));
                }
            }
        }

        trace?.Invoke("bb.embed", hidden);

        var outputs = new float[_outIndices.Length][];
        for (int i = 0; i < _layers.Length; i++)
        {
            // window_block_indexes = range(last out index + 1) minus the out indexes.
            bool full = Array.IndexOf(_outIndices, i) >= 0;
            _layers[i].Forward(hidden, full ? 1 : windows2, full ? windows2 * perWindow : perWindow);
            trace?.Invoke($"bb.layer{i + 1}", hidden);

            int stage = Array.IndexOf(_outIndices, i + 1);
            if (stage >= 0)
            {
                float[] normed = (float[])hidden.Clone();
                _norm.Apply(normed);
                float[] map = new float[tokens * Width];
                for (int w = 0; w < windows2; w++)
                {
                    int wr = w / _windows, wc = w % _windows;
                    for (int r = 0; r < wh; r++)
                    {
                        for (int col = 0; col < ww; col++)
                        {
                            int target = (((wr * wh) + r) * gw) + (wc * ww) + col;
                            normed.AsSpan(((w * perWindow) + 1 + (r * ww) + col) * Width, Width)
                                .CopyTo(map.AsSpan(target * Width, Width));
                        }
                    }
                }

                outputs[stage] = map;
                if (trace is not null)
                {
                    trace($"bb.stage{stage}", LayoutMath.Transpose(map, tokens, Width));
                }
            }
        }

        return outputs;
    }

    /// <summary>
    /// <c>WindowedDinov2WithRegistersEmbeddings.interpolate_pos_encoding</c>: the patch part of the
    /// table resized to the patch grid, <c>[gh·gw, C]</c>. Cached per grid.
    /// </summary>
    public float[] PositionEmbedding(int gh, int gw)
    {
        lock (_interpolated)
        {
            if (_interpolated.TryGetValue((gh, gw), out float[]? cached))
            {
                return cached;
            }

            int n = _positionGrid;
            float[] result;
            if (n * n == gh * gw && gh == gw)
            {
                result = _positions.AsSpan(Width).ToArray();
            }
            else
            {
                // [n*n, C] -> [C, n, n] -> bicubic AA -> [C, gh, gw] -> [gh*gw, C].
                float[] planar = LayoutMath.Transpose(_positions.AsSpan(Width), n * n, Width);
                float[] resized = TorchResize.Resize(planar, Width, n, n, gh, gw, TorchResize.Mode.Bicubic);
                result = LayoutMath.Transpose(resized, Width, gh * gw);
            }

            _interpolated[(gh, gw)] = result;
            return result;
        }
    }

    /// <summary><c>WindowedDinov2WithRegistersLayer</c>: pre-norm attention and MLP with layer scale.</summary>
    private sealed class Layer
    {
        private readonly LayerNormLayer _norm1, _norm2;
        private readonly Dense _qkv, _o, _fc1, _fc2;
        private readonly float[] _scale1, _scale2;

        public Layer(StateDict state, string p)
        {
            _norm1 = state.Norm(p + "norm1", Eps);
            _norm2 = state.Norm(p + "norm2", Eps);
            // query, key and value as one [3·384, 384] product.
            string a = p + "attention.attention.";
            byte[] weight = [.. state.Get(a + "query.weight").Bytes.Span, .. state.Get(a + "key.weight").Bytes.Span, .. state.Get(a + "value.weight").Bytes.Span];
            float[] bias = [.. state.Floats(a + "query.bias"), .. state.Floats(a + "key.bias"), .. state.Floats(a + "value.bias")];
            _qkv = new Dense(WeightMatrix.Create(weight, DType.Float32, 3 * Width, Width), bias);
            _o = state.Dense(p + "attention.output.dense");
            _fc1 = state.Dense(p + "mlp.fc1");
            _fc2 = state.Dense(p + "mlp.fc2");
            _scale1 = state.Floats(p + "layer_scale1.lambda1");
            _scale2 = state.Floats(p + "layer_scale2.lambda1");
        }

        /// <summary>Updates <paramref name="hidden"/> (<c>groups × tokens</c> rows) in place.</summary>
        public void Forward(float[] hidden, int groups, int tokens)
        {
            int rows = groups * tokens;
            float[] x = (float[])hidden.Clone();
            _norm1.Apply(x);
            float[] qkv = _qkv.Forward(x, rows);

            int headDim = Width / Heads;
            float[] context = new float[rows * Width];
            float[] qh = new float[tokens * Width], kh = new float[tokens * Width], vh = new float[tokens * Width], oh = new float[tokens * Width];
            for (int g = 0; g < groups; g++)
            {
                int offset = g * tokens * Width;
                ReadOnlySpan<float> group = qkv.AsSpan(3 * offset, 3 * tokens * Width);
                ToHeads(group, 0, tokens, headDim, qh);
                ToHeads(group, Width, tokens, headDim, kh);
                ToHeads(group, 2 * Width, tokens, headDim, vh);
                Attention.Bidirectional(qh, kh, vh, oh, Heads, tokens, headDim, 1f / MathF.Sqrt(headDim));
                FromHeads(oh, tokens, headDim, context.AsSpan(offset, tokens * Width));
            }

            float[] attn = _o.Forward(context, rows);
            ScaleAdd(hidden, attn, _scale1, rows);

            x = (float[])hidden.Clone();
            _norm2.Apply(x);
            float[] h = _fc1.Forward(x, rows);
            Kernels.GeluErfParallel(h);
            float[] mlp = _fc2.Forward(h, rows);
            ScaleAdd(hidden, mlp, _scale2, rows);
        }

        /// <summary><c>hidden += layer_scale(x)</c>, i.e. <c>drop_path(x · λ) + shortcut</c>.</summary>
        private static void ScaleAdd(float[] hidden, float[] x, float[] lambda, int rows)
        {
            for (int r = 0; r < rows; r++)
            {
                Span<float> xr = x.AsSpan(r * Width, Width);
                System.Numerics.Tensors.TensorPrimitives.Multiply(xr, lambda, xr);
                LayoutMath.Add(hidden.AsSpan(r * Width, Width), xr);
            }
        }

        /// <summary>Columns <c>[column, column + 384)</c> of <c>[tokens, 3·384]</c> rows → <c>[heads][tokens][headDim]</c>.</summary>
        private static void ToHeads(ReadOnlySpan<float> x, int column, int tokens, int headDim, Span<float> heads)
        {
            for (int t = 0; t < tokens; t++)
            {
                for (int h = 0; h < Heads; h++)
                {
                    x.Slice((t * 3 * Width) + column + (h * headDim), headDim).CopyTo(heads.Slice(((h * tokens) + t) * headDim, headDim));
                }
            }
        }

        private static void FromHeads(ReadOnlySpan<float> heads, int tokens, int headDim, Span<float> x)
        {
            for (int t = 0; t < tokens; t++)
            {
                for (int h = 0; h < Heads; h++)
                {
                    heads.Slice(((h * tokens) + t) * headDim, headDim).CopyTo(x.Slice((t * Width) + (h * headDim), headDim));
                }
            }
        }
    }
}
