using IndraOcr.Vlm.Core;

namespace IndraOcr.Vlm.Models;

/// <summary>
/// <c>Qwen3_5TextModel</c>: a hybrid decoder of gated-delta-net layers and gated full-attention
/// layers (three to one in both checkpoints), each followed by a SwiGLU MLP
/// (<c>modeling_qwen3_5.py</c>, 139-906 and 1212-1300).
/// </summary>
/// <remarks>
/// <para>
/// The full-attention layers: <c>q_proj</c> produces a query and an output gate per head;
/// queries and keys get a zero-centred RMSNorm (<c>x · (1 + w)</c>) over the head; the rotary
/// embedding covers the first quarter of the head and is interleaved M-RoPE (frequency <c>i</c>
/// takes the row position when <c>i % 3 == 1</c>, the column when <c>i % 3 == 2</c>, within
/// <c>mrope_section</c>, and the temporal one otherwise); the attention output is multiplied by
/// <c>sigmoid(gate)</c> before <c>o_proj</c>.
/// </para>
/// <para>
/// The gated delta net keeps a <c>[key_dim, value_dim]</c> state per value head. Each token:
/// <c>S ← S · exp(g)</c>, <c>δ = (v − Sᵀk) · β</c>, <c>S ← S + k δᵀ</c>, <c>o = Sᵀq</c>, after a
/// causal depthwise convolution (kernel 4, SiLU) over the packed q/k/v projection and L2
/// normalisation of q and k. The reference runs prompts through the chunked form of the same
/// recurrence (<c>torch_chunk_gated_delta_rule</c>) and single tokens through the recurrent form;
/// both are this, up to float32 rounding, and this runs the recurrent form throughout.
/// </para>
/// </remarks>
public sealed class TextDecoder
{
    private readonly Qwen35Config _config;
    private readonly WeightMatrix _embeddings;
    private readonly WeightMatrix _lmHead;
    private readonly float[] _finalNorm;
    private readonly Layer[] _layers;
    private readonly float[] _inverseFrequency;
    private readonly int[] _frequencyAxis;

    /// <summary>Binds the decoder's weights.</summary>
    public TextDecoder(WeightStore weights, Qwen35Config config, string prefix = "model.language_model.")
    {
        _config = config;
        _embeddings = weights.Matrix(prefix + "embed_tokens.weight");
        _lmHead = config.TieWordEmbeddings || !weights.Contains("lm_head.weight")
            ? _embeddings
            : weights.Matrix("lm_head.weight");
        _finalNorm = OnePlus(weights.Vector(prefix + "norm.weight"));

        _layers = new Layer[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            string l = $"{prefix}layers.{i}.";
            var layer = new Layer
            {
                IsLinear = config.LayerTypes[i] != "full_attention",
                InputNorm = OnePlus(weights.Vector(l + "input_layernorm.weight")),
                PostNorm = OnePlus(weights.Vector(l + "post_attention_layernorm.weight")),
                Gate = weights.Matrix(l + "mlp.gate_proj.weight"),
                Up = weights.Matrix(l + "mlp.up_proj.weight"),
                Down = weights.Matrix(l + "mlp.down_proj.weight"),
            };

            if (layer.IsLinear)
            {
                string a = l + "linear_attn.";
                layer.InProjQkv = weights.Matrix(a + "in_proj_qkv.weight");
                layer.InProjZ = weights.Matrix(a + "in_proj_z.weight");
                layer.InProjB = weights.Matrix(a + "in_proj_b.weight");
                layer.InProjA = weights.Matrix(a + "in_proj_a.weight");
                layer.ConvWeight = weights.Vector(a + "conv1d.weight");
                layer.NegativeA = weights.Vector(a + "A_log").Select(x => -MathF.Exp(x)).ToArray();
                layer.DtBias = weights.Vector(a + "dt_bias");
                layer.GatedNorm = weights.Vector(a + "norm.weight");
                layer.OutProj = weights.Matrix(a + "out_proj.weight");
            }
            else
            {
                string a = l + "self_attn.";
                layer.QProj = weights.Matrix(a + "q_proj.weight");
                layer.KProj = weights.Matrix(a + "k_proj.weight");
                layer.VProj = weights.Matrix(a + "v_proj.weight");
                layer.OProj = weights.Matrix(a + "o_proj.weight");
                layer.QNorm = OnePlus(weights.Vector(a + "q_norm.weight"));
                layer.KNorm = OnePlus(weights.Vector(a + "k_norm.weight"));
            }

            _layers[i] = layer;
        }

        // compute_default_rope_parameters, then recomposition_frequencies' interleaving.
        int rotary = config.RotaryDim;
        _inverseFrequency = new float[rotary / 2];
        _frequencyAxis = new int[rotary / 2];
        for (int i = 0; i < _inverseFrequency.Length; i++)
        {
            _inverseFrequency[i] = 1f / (float)Math.Pow(config.RopeTheta, (2 * i) / (float)rotary);
            _frequencyAxis[i] = (i % 3 == 1 && i < 3 * config.MropeSection[1]) ? 1
                : (i % 3 == 2 && i < 3 * config.MropeSection[2]) ? 2
                : 0;
        }
    }

    /// <summary>The configuration.</summary>
    public Qwen35Config Config => _config;

    /// <summary>A fresh per-sequence state.</summary>
    public DecoderState CreateState(int capacity = 1024) => new(_config, capacity);

    /// <summary>Copies token embeddings into <paramref name="destination"/>, <c>[tokens, hidden]</c>.</summary>
    public void Embed(ReadOnlySpan<int> tokens, Span<float> destination)
    {
        int width = _config.HiddenSize;
        for (int i = 0; i < tokens.Length; i++)
        {
            _embeddings.CopyRow(tokens[i], destination.Slice(i * width, width));
        }
    }

    /// <summary>Output-head logits for <paramref name="rows"/> final hidden states.</summary>
    public void Logits(ReadOnlyMemory<float> hidden, int rows, Memory<float> logits) =>
        Gemm.Linear(hidden, rows, _config.HiddenSize, _lmHead, default, logits, _config.VocabSize);

    /// <summary>
    /// Runs <paramref name="count"/> tokens through every layer, appending them to
    /// <paramref name="state"/>, and leaves the final-normed hidden states in <paramref name="hidden"/>.
    /// </summary>
    /// <param name="hidden"><c>[count, hidden]</c> input embeddings, overwritten with the output.</param>
    /// <param name="count">Tokens.</param>
    /// <param name="positions"><c>[3, count]</c> M-RoPE positions (temporal, row, column).</param>
    /// <param name="state">The sequence's caches.</param>
    /// <param name="trace">Receives per-layer tensors.</param>
    /// <param name="profile">Receives stage timings.</param>
    public void Forward(
        Memory<float> hidden,
        int count,
        ReadOnlySpan<int> positions,
        DecoderState state,
        TraceSink? trace = null,
        StageProfile? profile = null)
    {
        int width = _config.HiddenSize;
        int rotary = _config.RotaryDim;
        state.Reserve(count);

        float[] cos = new float[count * rotary];
        float[] sin = new float[count * rotary];
        int half = rotary / 2;
        for (int t = 0; t < count; t++)
        {
            for (int i = 0; i < half; i++)
            {
                float angle = positions[(_frequencyAxis[i] * count) + t] * _inverseFrequency[i];
                cos[(t * rotary) + i] = cos[(t * rotary) + i + half] = MathF.Cos(angle);
                sin[(t * rotary) + i] = sin[(t * rotary) + i + half] = MathF.Sin(angle);
            }
        }

        using PooledBuffer normed = TensorPool.Rent(count * width);
        using PooledBuffer mixed = TensorPool.Rent(count * width);
        using PooledBuffer gate = TensorPool.Rent(count * _config.IntermediateSize);
        using PooledBuffer up = TensorPool.Rent(count * _config.IntermediateSize);

        for (int index = 0; index < _layers.Length; index++)
        {
            Layer layer = _layers[index];
            using (profile?.Measure("norm"))
            {
                hidden.Span[..(count * width)].CopyTo(normed.Span);
                Norms.RmsNormParallel(normed.Memory[..(count * width)], width, layer.InputNorm, _config.RmsNormEps);
            }

            if (layer.IsLinear)
            {
                LinearAttention(layer, index, normed.Memory, count, state, mixed.Memory, trace, profile);
            }
            else
            {
                FullAttention(layer, index, normed.Memory, count, state, cos, sin, mixed.Memory, trace, profile);
            }

            Kernels.AddInPlace(hidden.Span[..(count * width)], mixed.Span[..(count * width)]);

            using (profile?.Measure("norm"))
            {
                hidden.Span[..(count * width)].CopyTo(normed.Span);
                Norms.RmsNormParallel(normed.Memory[..(count * width)], width, layer.PostNorm, _config.RmsNormEps);
            }

            int inner = _config.IntermediateSize;
            using (profile?.Measure("mlp gemm"))
            {
                Gemm.Linear(normed.Memory, count, width, layer.Gate, default, gate.Memory, inner);
                Gemm.Linear(normed.Memory, count, width, layer.Up, default, up.Memory, inner);
            }

            using (profile?.Measure("silu"))
            {
                Kernels.SiluParallel(gate.Memory[..(count * inner)]);
                Kernels.MultiplyInPlace(gate.Span[..(count * inner)], up.Span[..(count * inner)]);
            }

            using (profile?.Measure("mlp gemm"))
            {
                Gemm.Linear(gate.Memory, count, inner, layer.Down, default, mixed.Memory, width);
            }

            Kernels.AddInPlace(hidden.Span[..(count * width)], mixed.Span[..(count * width)]);
            trace?.Invoke($"txt.layer{index}", hidden.Span[..(count * width)], width);
        }

        state.Length += count;
        Norms.RmsNormParallel(hidden[..(count * width)], width, _finalNorm, _config.RmsNormEps);
        trace?.Invoke("txt.norm", hidden.Span[..(count * width)], width);
    }

    private void LinearAttention(
        Layer layer,
        int index,
        ReadOnlyMemory<float> normed,
        int count,
        DecoderState state,
        Memory<float> output,
        TraceSink? trace,
        StageProfile? profile)
    {
        int width = _config.HiddenSize;
        int keyDim = _config.LinearKeyDim;
        int valueDim = _config.LinearValueDim;
        int convDim = (2 * keyDim) + valueDim;
        int valueHeads = _config.LinearValueHeads;
        int keyHeadDim = _config.LinearKeyHeadDim;
        int valueHeadDim = _config.LinearValueHeadDim;
        int repeat = valueHeads / _config.LinearKeyHeads;

        using PooledBuffer qkv = TensorPool.Rent(count * convDim);
        using PooledBuffer z = TensorPool.Rent(count * valueDim);
        using PooledBuffer b = TensorPool.Rent(count * valueHeads);
        using PooledBuffer a = TensorPool.Rent(count * valueHeads);
        using PooledBuffer convolved = TensorPool.Rent(count * convDim);
        using PooledBuffer core = TensorPool.Rent(count * valueDim);

        using (profile?.Measure("linear gemm"))
        {
            Gemm.Linear(normed, count, width, layer.InProjQkv, default, qkv.Memory, convDim);
            Gemm.Linear(normed, count, width, layer.InProjZ, default, z.Memory, valueDim);
            Gemm.Linear(normed, count, width, layer.InProjB, default, b.Memory, valueHeads);
            Gemm.Linear(normed, count, width, layer.InProjA, default, a.Memory, valueHeads);
        }

        using (profile?.Measure("conv"))
        {
            CausalConv(layer.ConvWeight, state.ConvHistory[index], qkv.Memory, convolved.Memory, count, convDim);
        }

        using (profile?.Measure("delta rule"))
        {
            float[] recurrent = state.Recurrent[index];
            float scale = 1f / MathF.Sqrt(keyHeadDim);
            ReadOnlyMemory<float> conv = convolved.Memory;
            ReadOnlyMemory<float> bMemory = b.Memory;
            ReadOnlyMemory<float> aMemory = a.Memory;
            Memory<float> coreMemory = core.Memory;
            Parallel.For(0, valueHeads, Parallelism.Options, head =>
                DeltaRuleHead(
                    layer, head, head / repeat, count, conv.Span, convDim, keyDim, keyHeadDim, valueHeadDim,
                    bMemory.Span, aMemory.Span, valueHeads, scale,
                    recurrent.AsSpan(head * keyHeadDim * valueHeadDim, keyHeadDim * valueHeadDim),
                    coreMemory.Span, valueDim));
        }

        if (index == 0 && trace is not null)
        {
            trace("txt.l0.core", core.Span[..(count * valueDim)], valueHeadDim);
            trace("txt.l0.gate", z.Span[..(count * valueDim)], valueHeadDim);
        }

        using (profile?.Measure("gated norm"))
        {
            GatedNorm(core.Span[..(count * valueDim)], z.Span[..(count * valueDim)], layer.GatedNorm, _config.RmsNormEps);
        }

        using (profile?.Measure("linear gemm"))
        {
            Gemm.Linear(core.Memory, count, valueDim, layer.OutProj, default, output, width);
        }

        if (index == 0)
        {
            trace?.Invoke("txt.l0.out", output.Span[..(count * width)], width);
        }
    }

    /// <summary>
    /// <c>causal_conv1d_fn</c> / <c>causal_conv1d_update</c>: a depthwise convolution over time,
    /// left-padded by the history the state carries, then SiLU.
    /// </summary>
    private static void CausalConv(
        float[] weight,
        float[] history,
        ReadOnlyMemory<float> input,
        Memory<float> output,
        int count,
        int channels)
    {
        int kernel = weight.Length / channels;
        int past = kernel - 1;
        Parallelism.Chunked(channels, (start, end) =>
        {
            ReadOnlySpan<float> x = input.Span;
            Span<float> y = output.Span;
            for (int c = start; c < end; c++)
            {
                ReadOnlySpan<float> w = weight.AsSpan(c * kernel, kernel);
                for (int t = 0; t < count; t++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kernel; j++)
                    {
                        int source = t - past + j;
                        float value = source >= 0 ? x[(source * channels) + c] : history[((past + source) * channels) + c];
                        sum += w[j] * value;
                    }

                    y[(t * channels) + c] = sum / (1f + MathF.Exp(-sum));
                }

                // The history becomes the last `past` inputs.
                for (int j = 0; j < past; j++)
                {
                    int source = count - past + j;
                    history[(j * channels) + c] = source >= 0
                        ? x[(source * channels) + c]
                        : history[((past + source) * channels) + c];
                }
            }
        });
    }

    private static void DeltaRuleHead(
        Layer layer,
        int head,
        int keyHead,
        int count,
        ReadOnlySpan<float> conv,
        int convDim,
        int keyDim,
        int keyHeadDim,
        int valueHeadDim,
        ReadOnlySpan<float> bAll,
        ReadOnlySpan<float> aAll,
        int valueHeads,
        float scale,
        Span<float> state,
        Span<float> core,
        int valueDim)
    {
        Span<float> q = stackalloc float[keyHeadDim];
        Span<float> k = stackalloc float[keyHeadDim];
        Span<float> kv = stackalloc float[valueHeadDim];
        float negativeA = layer.NegativeA[head];
        float dtBias = layer.DtBias[head];

        for (int t = 0; t < count; t++)
        {
            int row = t * convDim;
            conv.Slice(row + (keyHead * keyHeadDim), keyHeadDim).CopyTo(q);
            conv.Slice(row + keyDim + (keyHead * keyHeadDim), keyHeadDim).CopyTo(k);
            ReadOnlySpan<float> v = conv.Slice(row + (2 * keyDim) + (head * valueHeadDim), valueHeadDim);

            // l2norm (eps 1e-6), then the query's 1/sqrt(d_k).
            float qInv = 1f / MathF.Sqrt(Norms.SumOfSquares(q) + 1e-6f);
            float kInv = 1f / MathF.Sqrt(Norms.SumOfSquares(k) + 1e-6f);
            for (int i = 0; i < keyHeadDim; i++)
            {
                q[i] = q[i] * qInv * scale;
                k[i] *= kInv;
            }

            float beta = 1f / (1f + MathF.Exp(-bAll[(t * valueHeads) + head]));
            float x = aAll[(t * valueHeads) + head] + dtBias;
            float softplus = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
            float decay = MathF.Exp(negativeA * softplus);

            // Pass 1: decay the state and read it with the key.
            kv.Clear();
            for (int i = 0; i < keyHeadDim; i++)
            {
                Span<float> rowState = state.Slice(i * valueHeadDim, valueHeadDim);
                Kernels.Scale(rowState, decay);
                Kernels.AddScaled(kv, rowState, k[i]);
            }

            // δ = (v − kv) · β
            for (int j = 0; j < valueHeadDim; j++)
            {
                kv[j] = (v[j] - kv[j]) * beta;
            }

            // Pass 2: write k δᵀ and read the result with the query.
            Span<float> output = core.Slice((t * valueDim) + (head * valueHeadDim), valueHeadDim);
            output.Clear();
            for (int i = 0; i < keyHeadDim; i++)
            {
                Span<float> rowState = state.Slice(i * valueHeadDim, valueHeadDim);
                Kernels.AddScaled(rowState, kv, k[i]);
                Kernels.AddScaled(output, rowState, q[i]);
            }
        }
    }

    /// <summary>
    /// <c>Qwen3_5RMSNormGated</c>: an RMSNorm over each value head (plain weight), times
    /// <c>silu(gate)</c>.
    /// </summary>
    private static void GatedNorm(Span<float> values, ReadOnlySpan<float> gate, float[] weight, float epsilon)
    {
        int width = weight.Length;
        for (int start = 0; start < values.Length; start += width)
        {
            Span<float> row = values.Slice(start, width);
            float inverse = 1f / MathF.Sqrt((Norms.SumOfSquares(row) / width) + epsilon);
            for (int i = 0; i < width; i++)
            {
                float g = gate[start + i];
                row[i] = weight[i] * (row[i] * inverse) * (g / (1f + MathF.Exp(-g)));
            }
        }
    }

    private void FullAttention(
        Layer layer,
        int index,
        ReadOnlyMemory<float> normed,
        int count,
        DecoderState state,
        float[] cos,
        float[] sin,
        Memory<float> output,
        TraceSink? trace,
        StageProfile? profile)
    {
        int width = _config.HiddenSize;
        int heads = _config.NumHeads;
        int kvHeads = _config.NumKeyValueHeads;
        int headDim = _config.HeadDim;
        int rotary = _config.RotaryDim;
        int queryWidth = heads * headDim;
        int kvWidth = kvHeads * headDim;
        int past = state.Length;

        using PooledBuffer queryGate = TensorPool.Rent(count * 2 * queryWidth);
        using PooledBuffer queries = TensorPool.Rent(count * queryWidth);
        using PooledBuffer attention = TensorPool.Rent(count * queryWidth);
        float[] keys = state.Keys[index];
        float[] values = state.Values[index];

        using (profile?.Measure("attn gemm"))
        {
            Gemm.Linear(normed, count, width, layer.QProj!.Value, default, queryGate.Memory, 2 * queryWidth);
            Gemm.Linear(normed, count, width, layer.KProj!.Value, default, keys.AsMemory(past * kvWidth), kvWidth);
            Gemm.Linear(normed, count, width, layer.VProj!.Value, default, values.AsMemory(past * kvWidth), kvWidth);
        }

        using (profile?.Measure("qk norm+rope"))
        {
            float eps = _config.RmsNormEps;
            for (int t = 0; t < count; t++)
            {
                ReadOnlySpan<float> c = cos.AsSpan(t * rotary, rotary);
                ReadOnlySpan<float> s = sin.AsSpan(t * rotary, rotary);
                for (int h = 0; h < heads; h++)
                {
                    Span<float> q = queries.Span.Slice((t * queryWidth) + (h * headDim), headDim);
                    queryGate.Span.Slice((t * 2 * queryWidth) + (h * 2 * headDim), headDim).CopyTo(q);
                    Norms.RmsNorm(q, headDim, layer.QNorm, eps);
                    Rotary.ApplyPartial(q, c, s);
                }

                for (int h = 0; h < kvHeads; h++)
                {
                    Span<float> k = keys.AsSpan(((past + t) * kvWidth) + (h * headDim), headDim);
                    Norms.RmsNorm(k, headDim, layer.KNorm, eps);
                    Rotary.ApplyPartial(k, c, s);
                }
            }
        }

        using (profile?.Measure("attention"))
        {
            CausalAttention(queries.Memory, keys, values, past, count, heads, kvHeads, headDim, attention.Memory);
        }

        using (profile?.Measure("attn gate"))
        {
            Span<float> att = attention.Span;
            ReadOnlySpan<float> qg = queryGate.Span;
            for (int t = 0; t < count; t++)
            {
                for (int h = 0; h < heads; h++)
                {
                    Span<float> row = att.Slice((t * queryWidth) + (h * headDim), headDim);
                    ReadOnlySpan<float> g = qg.Slice((t * 2 * queryWidth) + (h * 2 * headDim) + headDim, headDim);
                    for (int i = 0; i < headDim; i++)
                    {
                        row[i] *= 1f / (1f + MathF.Exp(-g[i]));
                    }
                }
            }
        }

        using (profile?.Measure("attn gemm"))
        {
            Gemm.Linear(attention.Memory, count, queryWidth, layer.OProj!.Value, default, output, width);
        }

        if (index == 3)
        {
            trace?.Invoke("txt.l3.attn", output.Span[..(count * width)], width);
        }
    }

    /// <summary>
    /// Causal grouped-query attention of <paramref name="count"/> new queries against the cache.
    /// Long runs go through the blocked kernels (keys transposed once per key/value head); a
    /// handful of decode rows dot against the cache directly.
    /// </summary>
    private void CausalAttention(
        ReadOnlyMemory<float> queries,
        float[] keys,
        float[] values,
        int past,
        int count,
        int heads,
        int kvHeads,
        int headDim,
        Memory<float> output)
    {
        int queryWidth = heads * headDim;
        int kvWidth = kvHeads * headDim;
        int visible = past + count;
        int groups = heads / kvHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        if (count < 8)
        {
            Parallel.For(0, heads * count, Parallelism.Options, item =>
            {
                int h = item % heads;
                int t = item / heads;
                int g = h / groups;
                int limit = past + t + 1;
                using PooledBuffer scoreBuffer = TensorPool.Rent(limit);
                Span<float> scores = scoreBuffer.Span[..limit];
                ReadOnlySpan<float> q = queries.Span.Slice((t * queryWidth) + (h * headDim), headDim);
                for (int j = 0; j < limit; j++)
                {
                    scores[j] = Gemm.Dot(q, keys.AsSpan((j * kvWidth) + (g * headDim), headDim));
                }

                Kernels.ScaledSoftmax(scores, scale);
                Span<float> target = output.Span.Slice((t * queryWidth) + (h * headDim), headDim);
                target.Clear();
                for (int j = 0; j < limit; j++)
                {
                    Kernels.AddScaled(target, values.AsSpan((j * kvWidth) + (g * headDim), headDim), scores[j]);
                }
            });
            return;
        }

        const int Block = 16;
        int blocks = (count + Block - 1) / Block;

        // Per key/value head: keys as [headDim, visible], values as [visible, headDim].
        float[][] keysT = new float[kvHeads][];
        float[][] valueRows = new float[kvHeads][];
        Parallel.For(0, kvHeads, Parallelism.Options, g =>
        {
            float[] kt = TensorPool.RentArray(headDim * visible);
            float[] vr = TensorPool.RentArray(visible * headDim);
            for (int t0 = 0; t0 < visible; t0 += 32)
            {
                int tEnd = Math.Min(t0 + 32, visible);
                for (int d = 0; d < headDim; d++)
                {
                    for (int t = t0; t < tEnd; t++)
                    {
                        kt[(d * visible) + t] = keys[(t * kvWidth) + (g * headDim) + d];
                    }
                }
            }

            for (int t = 0; t < visible; t++)
            {
                values.AsSpan((t * kvWidth) + (g * headDim), headDim).CopyTo(vr.AsSpan(t * headDim, headDim));
            }

            keysT[g] = kt;
            valueRows[g] = vr;
        });

        try
        {
            Parallel.For(0, heads * blocks, Parallelism.Options, item =>
            {
                int h = item / blocks;
                int block = item % blocks;
                int g = h / groups;
                int start = block * Block;
                int rows = Math.Min(Block, count - start);
                int limit = past + start + rows;

                using PooledBuffer q = TensorPool.Rent(rows * headDim);
                using PooledBuffer scores = TensorPool.Rent(rows * limit);
                using PooledBuffer result = TensorPool.Rent(rows * headDim);
                for (int r = 0; r < rows; r++)
                {
                    queries.Span.Slice(((start + r) * queryWidth) + (h * headDim), headDim)
                        .CopyTo(q.Span.Slice(r * headDim, headDim));
                }

                AttentionKernels.Scores(q.Span, rows, headDim, keysT[g], visible, limit, scores.Span);
                for (int r = 0; r < rows; r++)
                {
                    int own = past + start + r + 1;
                    Span<float> row = scores.Span.Slice(r * limit, limit);
                    Kernels.ScaledSoftmax(row[..own], scale);
                    row[own..].Clear();
                }

                AttentionKernels.Values(scores.Span, rows, limit, valueRows[g], headDim, result.Span);
                for (int r = 0; r < rows; r++)
                {
                    result.Span.Slice(r * headDim, headDim)
                        .CopyTo(output.Span.Slice(((start + r) * queryWidth) + (h * headDim), headDim));
                }
            });
        }
        finally
        {
            for (int g = 0; g < kvHeads; g++)
            {
                TensorPool.Return(keysT[g]);
                TensorPool.Return(valueRows[g]);
            }
        }
    }

    private static float[] OnePlus(float[] weight) => weight.Select(w => 1f + w).ToArray();

    private sealed class Layer
    {
        public bool IsLinear;
        public required float[] InputNorm;
        public required float[] PostNorm;
        public required WeightMatrix Gate;
        public required WeightMatrix Up;
        public required WeightMatrix Down;

        // gated delta net
        public WeightMatrix InProjQkv;
        public WeightMatrix InProjZ;
        public WeightMatrix InProjB;
        public WeightMatrix InProjA;
        public float[] ConvWeight = [];
        public float[] NegativeA = [];
        public float[] DtBias = [];
        public float[] GatedNorm = [];
        public WeightMatrix OutProj;

        // full attention
        public WeightMatrix? QProj;
        public WeightMatrix? KProj;
        public WeightMatrix? VProj;
        public WeightMatrix? OProj;
        public float[] QNorm = [];
        public float[] KNorm = [];
    }
}
