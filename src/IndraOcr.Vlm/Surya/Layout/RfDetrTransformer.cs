using IndraOcr.Vlm.Core;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>
/// RF-DETR's two-stage deformable decoder — <c>Transformer.forward</c>,
/// <c>TransformerDecoder.forward</c> and <c>TransformerDecoderLayer.forward_post</c> of
/// <c>surya/common/rfdetr/models/transformer.py</c>, at inference (one query group), with the
/// configuration <c>rfdetr/predictor.py</c> builds: <c>two_stage</c>, <c>bbox_reparam</c>,
/// <c>lite_refpoint_refine</c>, one feature level, 4 layers, 8 self-attention and 16 cross-attention
/// heads, 2 sampling points.
/// </summary>
/// <remarks>
/// <para>
/// Unpadded input means the padding masks are all false and every valid ratio is 1, so
/// <c>gen_encoder_output_proposals</c> reduces to a grid of <c>((x+0.5)/W, (y+0.5)/H, 0.05, 0.05)</c>
/// boxes (all inside its (0.01, 0.99) validity band at the layout model's 28×28 grid; the masking is
/// still applied).
/// </para>
/// <para>
/// The sine position embedding the backbone computes for the memory is passed to the decoder
/// layers as <c>pos</c> and never read by <c>forward_post</c>, so it is not computed here.
/// </para>
/// </remarks>
internal sealed class RfDetrTransformer
{
    private const int D = 256;
    private const int SelfHeads = 8;
    private const int CrossHeads = 16;
    private const int Points = 2;

    private readonly Dense _encOutput;
    private readonly LayerNormLayer _encNorm;
    private readonly Dense _encClass;
    private readonly Mlp _encBox;
    private readonly Mlp _refPointHead;
    private readonly DecoderLayer[] _layers;
    private readonly LayerNormLayer _norm;
    private readonly float[] _refpointEmbed;
    private readonly float[] _queryFeat;

    public RfDetrTransformer(StateDict state, int queries)
    {
        Queries = queries;
        _encOutput = state.Dense("transformer.enc_output.0");
        _encNorm = state.Norm("transformer.enc_output_norm.0", 1e-5f);
        _encClass = state.Dense("transformer.enc_out_class_embed.0");
        _encBox = Mlp.Load(state, "transformer.enc_out_bbox_embed.0", 3);
        _refPointHead = Mlp.Load(state, "transformer.decoder.ref_point_head", 2);
        int count = 0;
        while (state.Contains($"transformer.decoder.layers.{count}.norm1.weight"))
        {
            count++;
        }

        _layers = Enumerable.Range(0, count).Select(i => new DecoderLayer(state, $"transformer.decoder.layers.{i}.")).ToArray();
        _norm = state.Norm("transformer.decoder.norm", 1e-5f);

        // Only the first group of the group-DETR queries is used at inference.
        _refpointEmbed = state.Floats("refpoint_embed.weight").AsSpan(0, queries * 4).ToArray();
        _queryFeat = state.Floats("query_feat.weight").AsSpan(0, queries * D).ToArray();
    }

    /// <summary>Number of decoder queries (<c>num_queries</c>).</summary>
    public int Queries { get; }

    /// <summary>
    /// Replaces the proposal ranking (for parity tests): receives the port's top-k and returns
    /// the order to use. The decoder's queries are positional — query <c>i</c> takes the learned
    /// <c>refpoint_embed[i]</c> and <c>query_feat[i]</c> — so two proposals whose scores differ by a
    /// few ulps and swap ranks give two queries different seeds.
    /// </summary>
    internal Func<int[], int[]>? ProposalOverride { get; set; }

    /// <summary>
    /// Runs the two-stage head and the decoder over <paramref name="memory"/> (<c>[h·w, 256]</c>);
    /// returns the normed last-layer states <c>[Q, 256]</c> and the (unsigmoided, re-parameterised)
    /// reference boxes they refine, <c>[Q, 4]</c>.
    /// </summary>
    public (float[] Hidden, float[] References) Forward(float[] memory, int h, int w, LayoutTrace? trace)
    {
        int tokens = h * w;

        // gen_encoder_output_proposals(memory, mask, shapes, unsigmoid=False).
        float[] proposals = new float[tokens * 4];
        bool[] valid = new bool[tokens];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int t = (y * w) + x;
                Span<float> p = proposals.AsSpan(t * 4, 4);
                p[0] = (x + 0.5f) / w;
                p[1] = (y + 0.5f) / h;
                p[2] = 0.05f;
                p[3] = 0.05f;
                valid[t] = p[0] > 0.01f && p[0] < 0.99f && p[1] > 0.01f && p[1] < 0.99f && p[2] > 0.01f && p[2] < 0.99f;
                if (!valid[t])
                {
                    p.Clear();
                }
            }
        }

        float[] outputMemory = (float[])memory.Clone();
        for (int t = 0; t < tokens; t++)
        {
            if (!valid[t])
            {
                outputMemory.AsSpan(t * D, D).Clear();
            }
        }

        float[] encMemory = _encOutput.Forward(outputMemory, tokens);
        _encNorm.Apply(encMemory);
        trace?.Invoke("enc.memory", encMemory);
        float[] encClass = _encClass.Forward(encMemory, tokens);
        trace?.Invoke("enc.class", encClass);
        float[] delta = _encBox.Forward(encMemory, tokens);
        float[] encCoord = new float[tokens * 4];
        for (int t = 0; t < tokens; t++)
        {
            Reparam(delta.AsSpan(t * 4, 4), proposals.AsSpan(t * 4, 4), encCoord.AsSpan(t * 4, 4));
        }

        trace?.Invoke("enc.coord", encCoord);

        // topk(enc_class.max(-1)) — sorted descending, ties by index.
        int classes = _encClass.Out;
        float[] best = new float[tokens];
        for (int t = 0; t < tokens; t++)
        {
            best[t] = System.Numerics.Tensors.TensorPrimitives.Max(encClass.AsSpan(t * classes, classes));
        }

        int topk = Math.Min(Queries, tokens);
        int[] order = TopK(best, topk);
        if (ProposalOverride is not null)
        {
            order = ProposalOverride(order);
        }
        if (trace is not null)
        {
            trace("enc.topk", order.Select(i => (float)i).ToArray());
        }

        float[] refTs = new float[topk * 4];
        for (int i = 0; i < topk; i++)
        {
            encCoord.AsSpan(order[i] * 4, 4).CopyTo(refTs.AsSpan(i * 4, 4));
        }

        trace?.Invoke("enc.refpoints", refTs);

        // Decoder inputs: the learned queries, and the learned reference boxes re-parameterised
        // around the selected proposals (the first ts_len of them; the rest stay as learned).
        float[] tgt = (float[])_queryFeat.Clone();
        float[] refs = (float[])_refpointEmbed.Clone();
        for (int i = 0; i < topk; i++)
        {
            float[] learned = refs.AsSpan(i * 4, 4).ToArray();
            Reparam(learned, refTs.AsSpan(i * 4, 4), refs.AsSpan(i * 4, 4));
        }

        trace?.Invoke("dec.refpoints", refs);

        // lite_refpoint_refine: the reference is computed once, from the initial boxes
        // (valid ratios are 1, so refpoints_input is the box itself).
        float[] sine = SineEmbedding(refs, Queries, D / 2);
        float[] queryPos = _refPointHead.Forward(sine, Queries);
        trace?.Invoke("dec.query_pos", queryPos);

        for (int l = 0; l < _layers.Length; l++)
        {
            tgt = _layers[l].Forward(tgt, queryPos, refs, Queries, memory, h, w);
            trace?.Invoke($"dec.layer{l}", tgt);
        }

        _norm.Apply(tgt);
        trace?.Invoke("dec.hs", tgt);
        return (tgt, refs);
    }

    /// <summary>
    /// The <c>bbox_reparam</c> update: <c>cxcy = δ[:2]·ref[2:] + ref[:2]</c>, <c>wh = exp(δ[2:])·ref[2:]</c>.
    /// </summary>
    public static void Reparam(ReadOnlySpan<float> delta, ReadOnlySpan<float> reference, Span<float> result)
    {
        float cx = (delta[0] * reference[2]) + reference[0];
        float cy = (delta[1] * reference[3]) + reference[1];
        float bw = MathF.Exp(delta[2]) * reference[2];
        float bh = MathF.Exp(delta[3]) * reference[3];
        result[0] = cx;
        result[1] = cy;
        result[2] = bw;
        result[3] = bh;
    }

    /// <summary>Indices of the <paramref name="k"/> largest values, largest first, ties by index.</summary>
    public static int[] TopK(ReadOnlySpan<float> values, int k)
    {
        int[] index = Enumerable.Range(0, values.Length).ToArray();
        float[] keys = values.ToArray();
        Array.Sort(index, (a, b) =>
        {
            int c = keys[b].CompareTo(keys[a]);
            return c != 0 ? c : a.CompareTo(b);
        });
        return index[..k];
    }

    /// <summary>
    /// <c>gen_sineembed_for_position(pos, dim=128)</c> for 4-d boxes: <c>[y, x, w, h]</c> blocks of
    /// interleaved <c>sin</c> / <c>cos</c> of <c>v·2π / 10000^(2⌊i/2⌋/128)</c>.
    /// </summary>
    public static float[] SineEmbedding(ReadOnlySpan<float> boxes, int count, int dim)
    {
        float[] dimT = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            // 10000 ** (2 * (dim_t // 2) / dim), in float32.
            float exponent = 2f * MathF.Floor(i / 2f) / dim;
            dimT[i] = MathF.Pow(10000f, exponent);
        }

        const float Scale = 2f * MathF.PI;
        int[] blocks = [1, 0, 2, 3]; // pos_y, pos_x, pos_w, pos_h
        float[] result = new float[count * 4 * dim];
        for (int q = 0; q < count; q++)
        {
            for (int b = 0; b < 4; b++)
            {
                float embed = boxes[(q * 4) + blocks[b]] * Scale;
                Span<float> o = result.AsSpan((q * 4 * dim) + (b * dim), dim);
                for (int i = 0; i < dim; i += 2)
                {
                    o[i] = MathF.Sin(embed / dimT[i]);
                    o[i + 1] = MathF.Cos(embed / dimT[i + 1]);
                }
            }
        }

        return result;
    }

    /// <summary><c>TransformerDecoderLayer.forward_post</c>.</summary>
    private sealed class DecoderLayer
    {
        private readonly MultiHeadAttention _selfAttn;
        private readonly LayerNormLayer _norm1, _norm2, _norm3;
        private readonly Dense _offsets, _weights, _valueProj, _outputProj, _linear1, _linear2;

        public DecoderLayer(StateDict state, string p)
        {
            _selfAttn = new MultiHeadAttention(state, p + "self_attn", D, SelfHeads);
            _norm1 = state.Norm(p + "norm1", 1e-5f);
            _norm2 = state.Norm(p + "norm2", 1e-5f);
            _norm3 = state.Norm(p + "norm3", 1e-5f);
            _offsets = state.Dense(p + "cross_attn.sampling_offsets");
            _weights = state.Dense(p + "cross_attn.attention_weights");
            _valueProj = state.Dense(p + "cross_attn.value_proj");
            _outputProj = state.Dense(p + "cross_attn.output_proj");
            _linear1 = state.Dense(p + "linear1");
            _linear2 = state.Dense(p + "linear2");
        }

        public float[] Forward(float[] tgt, float[] queryPos, float[] refs, int queries, float[] memory, int h, int w)
        {
            // Self-attention: q = k = tgt + query_pos, v = tgt.
            float[] qk = LayoutMath.Sum(tgt, queryPos);
            float[] q = _selfAttn.ProjectQuery(qk, queries);
            float[] k = _selfAttn.ProjectKey(qk, queries);
            float[] v = _selfAttn.ProjectValue(tgt, queries);
            float[] sa = _selfAttn.Output(_selfAttn.Attend(q, queries, k, v, queries), queries);
            LayoutMath.Add(tgt, sa);
            _norm1.Apply(tgt);

            // Deformable cross-attention over the memory.
            float[] query = LayoutMath.Sum(tgt, queryPos);
            float[] ca = DeformableAttention(query, refs, queries, memory, h, w);
            LayoutMath.Add(tgt, ca);
            _norm2.Apply(tgt);

            float[] ff = _linear1.Forward(tgt, queries);
            LayoutMath.Relu(ff);
            float[] ff2 = _linear2.Forward(ff, queries);
            LayoutMath.Add(tgt, ff2);
            _norm3.Apply(tgt);
            return tgt;
        }

        /// <summary>
        /// <c>MSDeformAttn.forward</c> with 4-d reference boxes and one level, sampling through
        /// <c>ms_deform_attn_core_pytorch</c> — <c>F.grid_sample(bilinear, zeros, align_corners=False)</c>.
        /// </summary>
        private float[] DeformableAttention(float[] query, float[] refs, int queries, float[] memory, int h, int w)
        {
            int tokens = h * w;
            int headDim = D / CrossHeads;
            float[] value = _valueProj.Forward(memory, tokens);
            float[] offsets = _offsets.Forward(query, queries); // [Q, heads, levels=1, points, 2]
            float[] weights = _weights.Forward(query, queries); // [Q, heads, levels*points]
            float[] output = new float[queries * D];

            // GridSamplerKernel.cpp's ComputeLocationBase<float, align_corners=false>.
            float scaleX = w / 2f, scaleY = h / 2f;
            Parallel.For(0, queries, Parallelism.Options, qi =>
            {
                ReadOnlySpan<float> r = refs.AsSpan(qi * 4, 4);
                Span<float> sampled = stackalloc float[Points * 16];
                for (int head = 0; head < CrossHeads; head++)
                {
                    Span<float> a = weights.AsSpan((qi * CrossHeads * Points) + (head * Points), Points);
                    Kernels.Softmax(a);
                    for (int p = 0; p < Points; p++)
                    {
                        int o = (((qi * CrossHeads) + head) * Points + p) * 2;
                        // reference[:2] + offsets / n_points * reference[2:] * 0.5
                        float lx = r[0] + (offsets[o] / Points * r[2] * 0.5f);
                        float ly = r[1] + (offsets[o + 1] / Points * r[3] * 0.5f);
                        float gx = (2f * lx) - 1f;
                        float gy = (2f * ly) - 1f;
                        float x = ((gx + 1f) * scaleX) - 0.5f;
                        float y = ((gy + 1f) * scaleY) - 0.5f;
                        Sample(value, h, w, head * headDim, headDim, x, y, sampled.Slice(p * headDim, headDim));
                    }

                    Span<float> dst = output.AsSpan((qi * D) + (head * headDim), headDim);
                    for (int c = 0; c < headDim; c++)
                    {
                        float acc = sampled[c] * a[0];
                        for (int p = 1; p < Points; p++)
                        {
                            acc += sampled[(p * headDim) + c] * a[p];
                        }

                        dst[c] = acc;
                    }
                }
            });

            return _outputProj.Forward(output, queries);
        }

        /// <summary><c>ApplyGridSample&lt;bilinear, zeros&gt;</c> for one location, every channel of one head.</summary>
        private static void Sample(float[] value, int h, int w, int channel, int count, float x, float y, Span<float> result)
        {
            float xw = MathF.Floor(x), yn = MathF.Floor(y);
            float fw = x - xw, fe = 1f - fw;
            float fn = y - yn, fs = 1f - fn;
            float nw = fs * fe, ne = fs * fw, sw = fn * fe, se = fn * fw;
            int x0 = (int)xw, y0 = (int)yn, x1 = x0 + 1, y1 = y0 + 1;
            bool vx0 = (uint)x0 < (uint)w, vx1 = (uint)x1 < (uint)w, vy0 = (uint)y0 < (uint)h, vy1 = (uint)y1 < (uint)h;
            for (int c = 0; c < count; c++)
            {
                float vnw = vx0 && vy0 ? value[(((y0 * w) + x0) * D) + channel + c] : 0f;
                float vne = vx1 && vy0 ? value[(((y0 * w) + x1) * D) + channel + c] : 0f;
                float vsw = vx0 && vy1 ? value[(((y1 * w) + x0) * D) + channel + c] : 0f;
                float vse = vx1 && vy1 ? value[(((y1 * w) + x1) * D) + channel + c] : 0f;
                result[c] = (vnw * nw) + (vne * ne) + (vsw * sw) + (vse * se);
            }
        }
    }
}
