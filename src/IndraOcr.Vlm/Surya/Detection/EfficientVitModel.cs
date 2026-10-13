using IndraOcr.Vlm.Core;
using IndraOcr.Vlm.Models;
using CoreGemm = IndraOcr.Core.Gemm;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>The model's sigmoid maps, <c>[labels, height, width]</c>.</summary>
/// <param name="Maps">The maps, label-major.</param>
/// <param name="Labels">Number of maps.</param>
/// <param name="Height">Rows of each map.</param>
/// <param name="Width">Columns of each map.</param>
public readonly record struct SegmentationOutput(float[] Maps, int Labels, int Height, int Width);

/// <summary>
/// Surya's text-line detector, <c>EfficientViTForSemanticSegmentation</c>
/// (<c>surya/detection/model/encoderdecoder.py</c>): an EfficientViT-L backbone (a convolutional
/// stem, three convolutional stages and one with LiteMLA linear attention) and a SegFormer-style
/// decode head, followed by a sigmoid.
/// </summary>
/// <remarks>
/// Float32 throughout, as Surya runs on CPU; the fp16 checkpoint is widened once at load (it is
/// 38 M parameters). Batch norms are kept as PyTorch evaluates them in inference — the per-channel
/// <c>x · (γ / √(var + ε)) + (β − mean · γ / √(var + ε))</c> of <c>batch_norm_cpu</c> — rather than
/// folded into the convolution weights.
/// </remarks>
public sealed class EfficientVitModel
{
    private readonly ConvNormAct _stemIn;
    private readonly IBlock[] _stem;
    private readonly IBlock[][] _stages;
    private readonly DecodeHead _head;

    private EfficientVitModel(DetectionConfig config, ConvNormAct stemIn, IBlock[] stem, IBlock[][] stages, DecodeHead head)
    {
        Config = config;
        _stemIn = stemIn;
        _stem = stem;
        _stages = stages;
        _head = head;
    }

    /// <summary>The configuration the model was built from.</summary>
    public DetectionConfig Config { get; }

    /// <summary>Builds the model from a checkpoint directory (<c>model.safetensors</c>).</summary>
    public static EfficientVitModel Load(string directory, DetectionConfig? config = null)
    {
        config ??= DetectionConfig.Load(directory);
        using WeightStore weights = WeightStore.Open(directory);
        var loader = new Loader(weights, config.NormEps);

        // EfficientVitLarge.__init__: Stem(block_type="large") then the four stages.
        ConvNormAct stemIn = loader.ConvNormAct(
            "vit.stem.in_conv", config.NumChannels, config.Widths[0], config.Strides[0] + 1, config.Strides[0],
            bias: false, norm: true, Activation.Hardswish);
        var stem = new IBlock[config.Depths[0]];
        for (int i = 0; i < stem.Length; i++)
        {
            string p = $"vit.stem.res{i}.main";
            stem[i] = new Residual(new ConvBlock(
                loader.ConvNormAct($"{p}.conv1", config.Widths[0], config.Widths[0], 3, 1, false, true, Activation.Hardswish),
                loader.ConvNormAct($"{p}.conv2", config.Widths[0], config.Widths[0], 3, 1, false, true, Activation.None)));
        }

        var stages = new IBlock[config.Widths.Length - 1][];
        int inChannels = config.Widths[0];
        for (int s = 0; s < stages.Length; s++)
        {
            int width = config.Widths[s + 1];
            int depth = config.Depths[s + 1];
            int stride = config.Strides[s + 1];
            bool vitStage = s >= 3;
            bool fewerNorm = s >= 2;
            var blocks = new List<IBlock>();
            string prefix = $"vit.stages.{s}.blocks";

            // EfficientVitLargeStage: the downsampling block, then `depth` blocks.
            blocks.Add(loader.LocalBlock(
                $"{prefix}.0.main", inChannels, width, stride, stride + 1, vitStage ? 24 : 16,
                fewerNorm: vitStage || fewerNorm, fused: !fewerNorm));
            for (int d = 0; d < depth; d++)
            {
                string p = $"{prefix}.{d + 1}";
                if (vitStage)
                {
                    blocks.Add(new EfficientVitBlock(
                        new Residual(loader.LiteMla($"{p}.context_module.main", width, config.HeadDim)),
                        new Residual(loader.LocalBlock($"{p}.local_module.main", width, width, 1, 3, 6, fewerNorm: true, fused: false))));
                }
                else
                {
                    blocks.Add(new Residual(loader.LocalBlock($"{p}.main", width, width, 1, 3, 4, fewerNorm, fused: !fewerNorm)));
                }
            }

            stages[s] = blocks.ToArray();
            inChannels = width;
        }

        var head = DecodeHead.Load(weights, config);
        return new EfficientVitModel(config, stemIn, stem, stages, head);
    }

    /// <summary>
    /// Runs the model on one processor output (<c>[3, H, W]</c>, CHW float32) and returns the
    /// sigmoid maps at stage 0's resolution (<c>[num_labels, H/4, W/4]</c>).
    /// </summary>
    /// <param name="pixels">The normalised image, channels first.</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="observe">Receives each stage's output (channels-last) under its reference name.</param>
    public SegmentationOutput Forward(ReadOnlySpan<float> pixels, int height, int width, Action<string, FeatureMap>? observe = null)
    {
        FeatureMap x = FeatureMap.Rent(height, width, Config.NumChannels);
        Span<float> hwc = x.Span;
        int plane = height * width;
        for (int c = 0; c < Config.NumChannels; c++)
        {
            ReadOnlySpan<float> src = pixels.Slice(c * plane, plane);
            for (int p = 0; p < plane; p++)
            {
                hwc[(p * Config.NumChannels) + c] = src[p];
            }
        }

        x = Replace(x, _stemIn.Forward(x));
        foreach (IBlock block in _stem)
        {
            x = Replace(x, block.Forward(x));
        }

        observe?.Invoke("model.stem", x);

        var hidden = new FeatureMap[_stages.Length];
        try
        {
            for (int s = 0; s < _stages.Length; s++)
            {
                for (int b = 0; b < _stages[s].Length; b++)
                {
                    IBlock block = _stages[s][b];
                    if (observe is not null && block is EfficientVitBlock vit && s == 3 && b == 1)
                    {
                        x = Replace(x, vit.Context.Forward(x));
                        observe("model.stage3.block1.context", x);
                        x = Replace(x, vit.Local.Forward(x));
                        observe("model.stage3.block1.local", x);
                        continue;
                    }

                    x = s > 0 && b == 0 ? block.Forward(x) : Replace(x, block.Forward(x));
                }

                hidden[s] = x;
                observe?.Invoke($"model.stage{s}", x);
            }

            float[] maps = _head.Forward(hidden, observe);
            return new SegmentationOutput(maps, Config.NumLabels, hidden[0].Height, hidden[0].Width);
        }
        finally
        {
            foreach (FeatureMap h in hidden)
            {
                h?.Dispose();
            }
        }
    }

    private static FeatureMap Replace(FeatureMap old, FeatureMap next)
    {
        if (!ReferenceEquals(old, next))
        {
            old.Dispose();
        }

        return next;
    }

    /// <summary>A module that maps one activation to the next.</summary>
    private interface IBlock
    {
        /// <summary>The block's output; the input is left untouched.</summary>
        FeatureMap Forward(FeatureMap x);
    }

    /// <summary><c>ConvNormAct</c>: a convolution, an optional frozen batch norm (or bias), an activation.</summary>
    private sealed class ConvNormAct(
        int inChannels, int outChannels, int kernel, int stride, int groups, float[] weight, float[]? scale, float[]? shift, Activation activation)
        : IBlock
    {
        /// <summary><c>get_padding(kernel, stride)</c> = <c>((stride − 1) + (kernel − 1)) // 2</c>.</summary>
        private readonly int _padding = ((stride - 1) + (kernel - 1)) / 2;

        public FeatureMap Forward(FeatureMap x)
        {
            FeatureMap y = groups == 1
                ? ConvKernels.Convolution(x, weight, outChannels, kernel, stride, _padding)
                : groups == inChannels && groups == outChannels
                    ? ConvKernels.Depthwise(x, weight, kernel, stride, _padding)
                    : throw new NotSupportedException("grouped convolution");
            ConvKernels.Affine(y, scale, shift, activation);
            return y;
        }
    }

    /// <summary><c>ResidualBlock(main, nn.Identity())</c>.</summary>
    private sealed class Residual(IBlock main) : IBlock
    {
        public FeatureMap Forward(FeatureMap x)
        {
            FeatureMap y = main.Forward(x);
            System.Numerics.Tensors.TensorPrimitives.Add(y.Span, x.Span, y.Span);
            return y;
        }
    }

    /// <summary><c>ConvBlock</c>, <c>FusedMBConv</c> (two convolutions) and <c>MBConv</c> (three).</summary>
    private sealed class ConvBlock(params ConvNormAct[] layers) : IBlock
    {
        public FeatureMap Forward(FeatureMap x)
        {
            FeatureMap y = layers[0].Forward(x);
            for (int i = 1; i < layers.Length; i++)
            {
                y = Replace(y, layers[i].Forward(y));
            }

            return y;
        }
    }

    /// <summary><c>EfficientVitBlock</c>: a residual LiteMLA, then a residual MBConv.</summary>
    private sealed class EfficientVitBlock(Residual context, Residual local) : IBlock
    {
        public Residual Context => context;

        public Residual Local => local;

        public FeatureMap Forward(FeatureMap x)
        {
            FeatureMap y = context.Forward(x);
            return Replace(y, local.Forward(y));
        }
    }

    /// <summary>
    /// <c>LiteMLA</c>, lightweight multi-scale linear attention: a 1×1 qkv projection, a 5×5
    /// depthwise plus grouped 1×1 aggregation of it, then per head of <c>dim</c> channels
    /// <c>out = relu(q) · (relu(k)ᵀ · [v, 1])</c>, divided by its last column plus ε.
    /// </summary>
    private sealed class LiteMla(
        int dim, int totalDim, float[] qkv, float[] aggregateDepthwise, float[] aggregatePointwise, int heads, ConvNormAct proj)
        : IBlock
    {
        private const float Eps = 1e-5f;

        public FeatureMap Forward(FeatureMap x)
        {
            int qkvChannels = 3 * totalDim;
            using FeatureMap q = ConvKernels.Convolution(x, qkv, qkvChannels, 1, 1, 0);
            using FeatureMap depthwise = ConvKernels.Depthwise(q, aggregateDepthwise, 5, 1, 2);
            using FeatureMap aggregated = ConvKernels.GroupedPointwise(depthwise, aggregatePointwise, qkvChannels, 3 * heads);

            // multi_scale_qkv = cat([qkv, aggregated]).reshape(B, -1, 3·dim, H·W): head h reads
            // channels [h·3·dim, (h+1)·3·dim) of the concatenation.
            int scaleHeads = 2 * qkvChannels / (3 * dim);
            int pixels = x.Pixels;
            using FeatureMap attended = FeatureMap.Rent(x.Height, x.Width, scaleHeads * dim);
            float[] output = attended.Array;
            int outChannels = scaleHeads * dim;
            Parallel.For(0, scaleHeads, Parallelism.Options, h =>
            {
                FeatureMap source = h < scaleHeads / 2 ? q : aggregated;
                int offset = (h % (scaleHeads / 2)) * 3 * dim;
                float[] data = source.Array;
                int stride = qkvChannels;

                // kv[i, j] = Σ_p relu(k[p, i]) · v'[p, j], v' = [v, 1].
                int cols = dim + 1;
                var kv = new float[dim * cols];
                var kRow = new float[dim];
                for (int p = 0; p < pixels; p++)
                {
                    int row = (p * stride) + offset;
                    for (int i = 0; i < dim; i++)
                    {
                        kRow[i] = MathF.Max(data[row + dim + i], 0f);
                    }

                    ReadOnlySpan<float> v = data.AsSpan(row + (2 * dim), dim);
                    for (int i = 0; i < dim; i++)
                    {
                        float ki = kRow[i];
                        Span<float> kvRow = kv.AsSpan(i * cols, cols);
                        System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(v, ki, kvRow[..dim], kvRow[..dim]);
                        kvRow[dim] += ki;
                    }
                }

                // out[p, j] = Σ_i relu(q[p, i]) · kv[i, j]; result = out[:, :dim] / (out[:, dim] + ε).
                var o = new float[cols];
                for (int p = 0; p < pixels; p++)
                {
                    int row = (p * stride) + offset;
                    Array.Clear(o);
                    for (int i = 0; i < dim; i++)
                    {
                        float qi = MathF.Max(data[row + i], 0f);
                        System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(kv.AsSpan(i * cols, cols), qi, o, o);
                    }

                    float denominator = o[dim] + Eps;
                    Span<float> dst = output.AsSpan((p * outChannels) + (h * dim), dim);
                    for (int j = 0; j < dim; j++)
                    {
                        dst[j] = o[j] / denominator;
                    }
                }
            });

            return proj.Forward(attended);
        }
    }

    /// <summary>Reads the checkpoint's tensors into the layouts the kernels take.</summary>
    private sealed class Loader(WeightStore weights, float normEps)
    {
        /// <summary>A <c>ConvNormAct</c> with a dense (groups = 1) convolution.</summary>
        public ConvNormAct ConvNormAct(
            string prefix, int inChannels, int outChannels, int kernel, int stride, bool bias, bool norm, Activation activation) =>
            Build(prefix, inChannels, outChannels, kernel, stride, 1, bias, norm, activation);

        /// <summary>A <c>ConvNormAct</c> with a depthwise convolution.</summary>
        public ConvNormAct DepthwiseNormAct(
            string prefix, int channels, int kernel, int stride, bool bias, bool norm, Activation activation) =>
            Build(prefix, channels, channels, kernel, stride, channels, bias, norm, activation);

        /// <summary>
        /// <c>build_local_block</c> with an expand ratio other than 1: <c>FusedMBConv</c> when
        /// <paramref name="fused"/>, else <c>MBConv</c>.
        /// </summary>
        public ConvBlock LocalBlock(
            string prefix, int inChannels, int outChannels, int stride, int kernel, int expandRatio, bool fewerNorm, bool fused)
        {
            int mid = (int)Math.Round(inChannels * (double)expandRatio, MidpointRounding.ToEven);
            if (fused)
            {
                // use_bias=(True, False) if fewer_norm else False; norm=(None, BN) if fewer_norm else BN.
                return new ConvBlock(
                    ConvNormAct($"{prefix}.spatial_conv", inChannels, mid, kernel, stride, fewerNorm, !fewerNorm, Activation.Hardswish),
                    ConvNormAct($"{prefix}.point_conv", mid, outChannels, 1, 1, false, true, Activation.None));
            }

            // use_bias=(True, True, False) if fewer_norm else False; norm=(None, None, BN) if fewer_norm else BN.
            return new ConvBlock(
                ConvNormAct($"{prefix}.inverted_conv", inChannels, mid, 1, 1, fewerNorm, !fewerNorm, Activation.Hardswish),
                DepthwiseNormAct($"{prefix}.depth_conv", mid, kernel, stride, fewerNorm, !fewerNorm, Activation.Hardswish),
                ConvNormAct($"{prefix}.point_conv", mid, outChannels, 1, 1, false, true, Activation.None));
        }

        /// <summary><c>LiteMLA(in, in, heads_ratio=1, dim=head_dim, norm=(None, BN))</c>, scales (5,).</summary>
        public LiteMla LiteMla(string prefix, int channels, int dim)
        {
            int heads = channels / dim;
            int totalDim = heads * dim;
            int qkvChannels = 3 * totalDim;
            float[] qkv = DenseWeight($"{prefix}.qkv.conv.weight", channels, qkvChannels, 1);
            float[] depthwise = DepthwiseWeight($"{prefix}.aggreg.0.0.weight", qkvChannels, 5);

            // Grouped 1×1, groups = 3·heads: [Cout, Cin/g] → per group [Cin/g, Cout/g].
            int groups = 3 * heads;
            int g = qkvChannels / groups;
            float[] raw = weights.Tensor($"{prefix}.aggreg.0.1.weight").ToFloats();
            var grouped = new float[groups * g * g];
            for (int group = 0; group < groups; group++)
            {
                for (int o = 0; o < g; o++)
                {
                    for (int i = 0; i < g; i++)
                    {
                        grouped[(group * g * g) + (i * g) + o] = raw[(((group * g) + o) * g) + i];
                    }
                }
            }

            ConvNormAct proj = ConvNormAct($"{prefix}.proj", 2 * totalDim, channels, 1, 1, false, true, Activation.None);
            return new LiteMla(dim, totalDim, qkv, depthwise, grouped, heads, proj);
        }

        private ConvNormAct Build(
            string prefix, int inChannels, int outChannels, int kernel, int stride, int groups, bool bias, bool norm, Activation activation)
        {
            float[] weight = groups == 1
                ? DenseWeight($"{prefix}.conv.weight", inChannels, outChannels, kernel)
                : DepthwiseWeight($"{prefix}.conv.weight", outChannels, kernel);
            float[]? scale = null;
            float[]? shift = bias ? weights.Tensor($"{prefix}.conv.bias").ToFloats() : null;
            if (norm)
            {
                (scale, float[] normShift) = BatchNorm(weights, $"{prefix}.norm", normEps);
                if (shift is not null)
                {
                    // Bias before a batch norm: (x + b)·α + β = x·α + (b·α + β).
                    for (int c = 0; c < outChannels; c++)
                    {
                        normShift[c] += shift[c] * scale[c];
                    }
                }

                shift = normShift;
            }

            return new ConvNormAct(inChannels, outChannels, kernel, stride, groups, weight, scale, shift, activation);
        }

        /// <summary><c>[Cout, Cin, k, k]</c> → <c>[(ky, kx, ci), Cout]</c>.</summary>
        private float[] DenseWeight(string name, int inChannels, int outChannels, int kernel)
        {
            float[] raw = weights.Tensor(name).ToFloats();
            int taps = kernel * kernel;
            var result = new float[taps * inChannels * outChannels];
            for (int o = 0; o < outChannels; o++)
            {
                for (int i = 0; i < inChannels; i++)
                {
                    for (int t = 0; t < taps; t++)
                    {
                        result[(((t * inChannels) + i) * outChannels) + o] = raw[(((o * inChannels) + i) * taps) + t];
                    }
                }
            }

            return result;
        }

        /// <summary><c>[C, 1, k, k]</c> → <c>[k·k, C]</c>.</summary>
        private float[] DepthwiseWeight(string name, int channels, int kernel)
        {
            float[] raw = weights.Tensor(name).ToFloats();
            int taps = kernel * kernel;
            var result = new float[taps * channels];
            for (int c = 0; c < channels; c++)
            {
                for (int t = 0; t < taps; t++)
                {
                    result[(t * channels) + c] = raw[(c * taps) + t];
                }
            }

            return result;
        }
    }

    /// <summary>
    /// A frozen <c>BatchNorm2d</c> as <c>batch_norm_cpu_collect_linear_and_constant_terms</c>
    /// reduces it: <c>α = γ / √(var + ε)</c>, <c>β = bias − mean · α</c>.
    /// </summary>
    internal static (float[] Scale, float[] Shift) BatchNorm(WeightStore weights, string prefix, float eps)
    {
        float[] gamma = weights.Tensor($"{prefix}.weight").ToFloats();
        float[] beta = weights.Tensor($"{prefix}.bias").ToFloats();
        float[] mean = weights.Tensor($"{prefix}.running_mean").ToFloats();
        float[] variance = weights.Tensor($"{prefix}.running_var").ToFloats();
        var scale = new float[gamma.Length];
        var shift = new float[gamma.Length];
        for (int c = 0; c < gamma.Length; c++)
        {
            float invStd = 1f / MathF.Sqrt(variance[c] + eps);
            scale[c] = invStd * gamma[c];
            shift[c] = beta[c] - (mean[c] * scale[c]);
        }

        return (scale, shift);
    }

    /// <summary>
    /// <c>DecodeHead</c>: each stage projected to <c>decoder_layer_hidden_size</c> by its
    /// <c>linear_c</c>, upsampled bilinearly to stage 0's resolution, concatenated deepest first,
    /// fused by a 1×1 convolution, batch norm and ReLU, and classified by a 1×1 convolution; then
    /// <c>torch.special.expit</c>.
    /// </summary>
    private sealed class DecodeHead
    {
        private readonly float[][] _linearWeights;
        private readonly float[][] _linearBias;
        private readonly float[] _fuse;
        private readonly float[] _fuseScale;
        private readonly float[] _fuseShift;
        private readonly float[] _classifier;
        private readonly float[] _classifierBias;
        private readonly int _hidden;
        private readonly int _layerHidden;
        private readonly int _labels;

        private DecodeHead(
            float[][] linearWeights, float[][] linearBias, float[] fuse, float[] fuseScale, float[] fuseShift,
            float[] classifier, float[] classifierBias, int hidden, int layerHidden, int labels)
        {
            _linearWeights = linearWeights;
            _linearBias = linearBias;
            _fuse = fuse;
            _fuseScale = fuseScale;
            _fuseShift = fuseShift;
            _classifier = classifier;
            _classifierBias = classifierBias;
            _hidden = hidden;
            _layerHidden = layerHidden;
            _labels = labels;
        }

        public static DecodeHead Load(WeightStore weights, DetectionConfig config)
        {
            int stages = config.Widths.Length - 1;
            var linearWeights = new float[stages][];
            var linearBias = new float[stages][];
            for (int i = 0; i < stages; i++)
            {
                linearWeights[i] = Transpose(weights.Tensor($"decode_head.linear_c.{i}.proj.weight").ToFloats(), config.DecoderLayerHiddenSize, config.Widths[i + 1]);
                linearBias[i] = weights.Tensor($"decode_head.linear_c.{i}.proj.bias").ToFloats();
            }

            int fuseIn = config.DecoderLayerHiddenSize * config.NumStages;
            float[] fuse = Transpose(weights.Tensor("decode_head.linear_fuse.weight").ToFloats(), config.DecoderHiddenSize, fuseIn);

            // DecodeHead's BatchNorm2d keeps nn.BatchNorm2d's default eps of 1e-5.
            (float[] scale, float[] shift) = BatchNorm(weights, "decode_head.batch_norm", 1e-5f);
            float[] classifier = Transpose(weights.Tensor("decode_head.classifier.weight").ToFloats(), config.NumLabels, config.DecoderHiddenSize);
            float[] classifierBias = weights.Tensor("decode_head.classifier.bias").ToFloats();
            return new DecodeHead(
                linearWeights, linearBias, fuse, scale, shift, classifier, classifierBias,
                config.DecoderHiddenSize, config.DecoderLayerHiddenSize, config.NumLabels);
        }

        /// <summary><c>[rows, cols]</c> → <c>[cols, rows]</c>.</summary>
        private static float[] Transpose(float[] source, int rows, int cols)
        {
            var result = new float[rows * cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    result[(c * rows) + r] = source[(r * cols) + c];
                }
            }

            return result;
        }

        public float[] Forward(FeatureMap[] hidden, Action<string, FeatureMap>? observe)
        {
            int height = hidden[0].Height;
            int width = hidden[0].Width;
            int stages = hidden.Length;
            int layer = _layerHidden;

            // linear_c at each stage's own resolution.
            var projected = new FeatureMap[stages];
            var taps = new ((int[] I0, int[] I1, float[] L0, float[] L1) Y, (int[] I0, int[] I1, float[] L0, float[] L1) X)[stages];
            try
            {
                for (int i = 0; i < stages; i++)
                {
                    projected[i] = ConvKernels.Convolution(hidden[i], _linearWeights[i], layer, 1, 1, 0);
                    ConvKernels.Affine(projected[i], null, _linearBias[i], Activation.None);
                    taps[i] = (ConvKernels.LinearTaps(hidden[i].Height, height), ConvKernels.LinearTaps(hidden[i].Width, width));
                }

                if (observe is not null)
                {
                    foreach (int i in new[] { 0, stages - 1 })
                    {
                        using FeatureMap up = FeatureMap.Rent(height, width, layer);
                        float[] dst = up.Array;
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                Upsample(projected[i], taps[i], y, x, dst.AsSpan(((y * width) + x) * layer, layer));
                            }
                        }

                        observe($"model.head.c{i}", up);
                    }
                }

                int fuseIn = layer * stages;
                int pixels = height * width;
                var logits = new float[_labels * pixels];
                int bandRows = Math.Max(1, (1 << 22) / (width * Math.Max(fuseIn, _hidden)));
                float[] cat = TensorPool.RentArray(bandRows * width * fuseIn);
                float[] fused = TensorPool.RentArray(bandRows * width * _hidden);
                float[] classes = TensorPool.RentArray(bandRows * width * _labels);
                FeatureMap? fullLogits = observe is null ? null : FeatureMap.Rent(height, width, _labels);
                try
                {
                    for (int r0 = 0; r0 < height; r0 += bandRows)
                    {
                        int rows = Math.Min(bandRows, height - r0);
                        int bandPixels = rows * width;

                        // torch.cat(all_hidden_states[::-1], dim=1): deepest stage first.
                        Parallel.For(0, rows, Parallelism.Options, r =>
                        {
                            int y = r0 + r;
                            for (int x = 0; x < width; x++)
                            {
                                Span<float> dst = cat.AsSpan(((r * width) + x) * fuseIn, fuseIn);
                                for (int i = 0; i < stages; i++)
                                {
                                    Upsample(projected[i], taps[i], y, x, dst.Slice((stages - 1 - i) * layer, layer));
                                }
                            }
                        });

                        CoreGemm.Multiply(bandPixels, _hidden, fuseIn, cat.AsSpan(0, bandPixels * fuseIn), fuseIn, _fuse, _hidden, fused.AsSpan(0, bandPixels * _hidden), _hidden);
                        Parallel.For(0, rows, Parallelism.Options, r =>
                        {
                            for (int x = 0; x < width; x++)
                            {
                                Span<float> v = fused.AsSpan(((r * width) + x) * _hidden, _hidden);
                                System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(v, _fuseScale, _fuseShift, v);
                                System.Numerics.Tensors.TensorPrimitives.Max(v, 0f, v);
                            }
                        });
                        CoreGemm.Multiply(bandPixels, _labels, _hidden, fused.AsSpan(0, bandPixels * _hidden), _hidden, _classifier, _labels, classes.AsSpan(0, bandPixels * _labels), _labels);
                        for (int p = 0; p < bandPixels; p++)
                        {
                            for (int c = 0; c < _labels; c++)
                            {
                                float z = classes[(p * _labels) + c] + _classifierBias[c];
                                if (fullLogits is not null)
                                {
                                    fullLogits.Array[(((r0 * width) + p) * _labels) + c] = z;
                                }

                                logits[(c * pixels) + (r0 * width) + p] = 1f / (1f + MathF.Exp(-z));
                            }
                        }
                    }

                    if (fullLogits is not null)
                    {
                        observe!("model.head.logits", fullLogits);
                    }
                }
                finally
                {
                    TensorPool.Return(cat);
                    TensorPool.Return(fused);
                    TensorPool.Return(classes);
                    fullLogits?.Dispose();
                }

                return logits;
            }
            finally
            {
                foreach (FeatureMap p in projected)
                {
                    p?.Dispose();
                }
            }
        }

        private static void Upsample(
            FeatureMap source,
            ((int[] I0, int[] I1, float[] L0, float[] L1) Y, (int[] I0, int[] I1, float[] L0, float[] L1) X) taps,
            int y,
            int x,
            Span<float> destination)
        {
            int c = source.Channels;
            int w = source.Width;
            float[] data = source.Array;
            float h0 = taps.Y.L0[y], h1 = taps.Y.L1[y], w0 = taps.X.L0[x], w1 = taps.X.L1[x];
            ReadOnlySpan<float> a = data.AsSpan(((taps.Y.I0[y] * w) + taps.X.I0[x]) * c, c);
            ReadOnlySpan<float> b = data.AsSpan(((taps.Y.I0[y] * w) + taps.X.I1[x]) * c, c);
            ReadOnlySpan<float> cc = data.AsSpan(((taps.Y.I1[y] * w) + taps.X.I0[x]) * c, c);
            ReadOnlySpan<float> d = data.AsSpan(((taps.Y.I1[y] * w) + taps.X.I1[x]) * c, c);
            for (int i = 0; i < c; i++)
            {
                destination[i] = (h0 * ((w0 * a[i]) + (w1 * b[i]))) + (h1 * ((w0 * cc[i]) + (w1 * d[i])));
            }
        }
    }
}
