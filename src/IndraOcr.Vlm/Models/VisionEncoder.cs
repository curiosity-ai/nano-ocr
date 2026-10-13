using IndraOcr.Vlm.Core;

namespace IndraOcr.Vlm.Models;

/// <summary>
/// <c>Qwen3_5VisionModel</c>: a ViT over native-resolution patches, followed by the 2×2 patch
/// merger that projects into the decoder's width (<c>modeling_qwen3_5.py</c>, 938-1200).
/// </summary>
/// <remarks>
/// <list type="bullet">
///   <item>The patch embedding is a <c>Conv3d</c> whose kernel equals its stride, so it is one
///   GEMM over the flattened <c>[3, temporal, patch, patch]</c> patches.</item>
///   <item>The learned <c>48×48</c> position table is resampled to the image's patch grid
///   bilinearly with <c>align_corners=True</c> (<c>vision_utils.get_vision_interpolation_indices_and_weights</c>),
///   four taps per patch, in float32.</item>
///   <item>The rotary embedding is axial: <c>head_dim/4</c> frequencies indexed by the patch row,
///   the same by the column, concatenated and duplicated to the full head, applied
///   <c>rotate_half</c>-style.</item>
///   <item>Attention is bidirectional within an image (the reference packs images and passes
///   <c>cu_seqlens</c>); each image here is encoded on its own, which is the same thing.</item>
///   <item>The patches arrive in spatial-merge-block order, so the merger's 2×2 merge is a
///   reshape: four consecutive rows become one.</item>
/// </list>
/// </remarks>
public sealed class VisionEncoder
{
    private readonly Qwen35Config _config;
    private readonly WeightMatrix _patchEmbed;
    private readonly float[] _patchBias;
    private readonly float[] _positionTable;
    private readonly int _gridSide;
    private readonly float[] _inverseFrequency;
    private readonly Block[] _blocks;
    private readonly float[] _mergerNormWeight;
    private readonly float[] _mergerNormBias;
    private readonly WeightMatrix _mergerFc1;
    private readonly float[] _mergerFc1Bias;
    private readonly WeightMatrix _mergerFc2;
    private readonly float[] _mergerFc2Bias;

    /// <summary>Binds the tower's weights.</summary>
    public VisionEncoder(WeightStore weights, Qwen35Config config, string prefix = "model.visual.")
    {
        _config = config;
        _patchEmbed = weights.MatrixFlattened(prefix + "patch_embed.proj.weight");
        _patchBias = weights.Vector(prefix + "patch_embed.proj.bias");
        _positionTable = weights.Vector(prefix + "pos_embed.weight");
        _gridSide = (int)Math.Sqrt(config.VisionPositionEmbeddings);

        // compute_axial_rope_parameters: 1 / theta ** (arange(0, head_dim / 2, 2) / (head_dim / 2)).
        int spatial = config.VisionHeadDim / 2;
        _inverseFrequency = new float[spatial / 2];
        for (int i = 0; i < _inverseFrequency.Length; i++)
        {
            float exponent = (2 * i) / (float)spatial;
            _inverseFrequency[i] = 1f / (float)Math.Pow(config.VisionRopeTheta, exponent);
        }

        _blocks = new Block[config.VisionDepth];
        for (int i = 0; i < config.VisionDepth; i++)
        {
            string b = $"{prefix}blocks.{i}.";
            _blocks[i] = new Block(
                weights.Vector(b + "norm1.weight"), weights.Vector(b + "norm1.bias"),
                weights.Vector(b + "norm2.weight"), weights.Vector(b + "norm2.bias"),
                weights.Matrix(b + "attn.qkv.weight"), weights.Vector(b + "attn.qkv.bias"),
                weights.Matrix(b + "attn.proj.weight"), weights.Vector(b + "attn.proj.bias"),
                weights.Matrix(b + "mlp.linear_fc1.weight"), weights.Vector(b + "mlp.linear_fc1.bias"),
                weights.Matrix(b + "mlp.linear_fc2.weight"), weights.Vector(b + "mlp.linear_fc2.bias"));
        }

        _mergerNormWeight = weights.Vector(prefix + "merger.norm.weight");
        _mergerNormBias = weights.Vector(prefix + "merger.norm.bias");
        _mergerFc1 = weights.Matrix(prefix + "merger.linear_fc1.weight");
        _mergerFc1Bias = weights.Vector(prefix + "merger.linear_fc1.bias");
        _mergerFc2 = weights.Matrix(prefix + "merger.linear_fc2.weight");
        _mergerFc2Bias = weights.Vector(prefix + "merger.linear_fc2.bias");
    }

    /// <summary>Decoder tokens one image produces.</summary>
    public int MergedTokens(PreprocessedImage image) =>
        image.PatchCount / (_config.SpatialMergeSize * _config.SpatialMergeSize);

    /// <summary>
    /// Encodes one image to <c>[MergedTokens, VisionOutHiddenSize]</c> decoder embeddings.
    /// </summary>
    /// <param name="image">The processed image.</param>
    /// <param name="trace">Receives intermediate tensors (parity tests).</param>
    /// <param name="profile">Receives stage timings.</param>
    public float[] Encode(PreprocessedImage image, TraceSink? trace = null, StageProfile? profile = null)
    {
        int tokens = image.PatchCount;
        int width = _config.VisionHiddenSize;
        int heads = _config.VisionHeads;
        int headDim = _config.VisionHeadDim;
        int inner = _config.VisionIntermediateSize;

        var hidden = new float[tokens * width];
        using (profile?.Measure("patch embed"))
        {
            Gemm.Linear(image.Pixels, tokens, _patchEmbed.Cols, _patchEmbed, _patchBias, hidden, width);
        }

        trace?.Invoke("vis.patch", hidden, width);

        AddPositionEmbedding(hidden, image);
        trace?.Invoke("vis.embed", hidden, width);

        using PooledBuffer cos = TensorPool.Rent(tokens * headDim);
        using PooledBuffer sin = TensorPool.Rent(tokens * headDim);
        FillRotary(image, cos.Span, sin.Span);

        using PooledBuffer normed = TensorPool.Rent(tokens * width);
        using PooledBuffer qkv = TensorPool.Rent(tokens * 3 * width);
        using PooledBuffer queries = TensorPool.Rent(tokens * width);
        using PooledBuffer keys = TensorPool.Rent(tokens * width);
        using PooledBuffer values = TensorPool.Rent(tokens * width);
        using PooledBuffer attention = TensorPool.Rent(tokens * width);
        using PooledBuffer packed = TensorPool.Rent(tokens * width);
        using PooledBuffer intermediate = TensorPool.Rent(tokens * inner);
        float scale = 1f / MathF.Sqrt(headDim);

        for (int index = 0; index < _blocks.Length; index++)
        {
            Block block = _blocks[index];

            using (profile?.Measure("norm"))
            {
                hidden.AsSpan().CopyTo(normed.Span);
                Norms.LayerNormParallel(normed.Memory, width, block.Norm1Weight, block.Norm1Bias, 1e-6f);
            }

            using (profile?.Measure("qkv gemm"))
            {
                Gemm.Linear(normed.Memory, tokens, width, block.Qkv, block.QkvBias, qkv.Memory, 3 * width);
            }

            using (profile?.Measure("rope+split"))
            {
                SplitHeads(qkv.Memory, queries.Memory, keys.Memory, values.Memory, tokens, heads, headDim, cos.Memory, sin.Memory);
            }

            using (profile?.Measure("attention"))
            {
                Attention.Bidirectional(
                    queries.Memory, keys.Memory, values.Memory, attention.Memory, heads, tokens, headDim, scale, profile);
            }

            using (profile?.Measure("rope+split"))
            {
                MergeHeads(attention.Memory, packed.Memory, tokens, heads, headDim);
            }

            using (profile?.Measure("out gemm"))
            {
                Gemm.Linear(packed.Memory, tokens, width, block.Proj, block.ProjBias, normed.Memory, width);
            }

            Kernels.AddInPlace(hidden, normed.Span[..(tokens * width)]);

            using (profile?.Measure("norm"))
            {
                hidden.AsSpan().CopyTo(normed.Span);
                Norms.LayerNormParallel(normed.Memory, width, block.Norm2Weight, block.Norm2Bias, 1e-6f);
            }

            using (profile?.Measure("mlp gemm"))
            {
                Gemm.Linear(normed.Memory, tokens, width, block.Fc1, block.Fc1Bias, intermediate.Memory, inner);
            }

            using (profile?.Measure("gelu"))
            {
                Kernels.GeluTanhParallel(intermediate.Memory[..(tokens * inner)]);
            }

            using (profile?.Measure("mlp gemm"))
            {
                Gemm.Linear(intermediate.Memory, tokens, inner, block.Fc2, block.Fc2Bias, normed.Memory, width);
            }

            Kernels.AddInPlace(hidden, normed.Span[..(tokens * width)]);
            trace?.Invoke($"vis.block{index}", hidden, width);
        }

        using (profile?.Measure("merger"))
        {
            return Merge(hidden, tokens, trace);
        }
    }

    private float[] Merge(float[] hidden, int tokens, TraceSink? trace)
    {
        int width = _config.VisionHiddenSize;
        int unit = _config.SpatialMergeSize * _config.SpatialMergeSize;
        int merged = tokens / unit;
        int mergedWidth = width * unit;
        int outWidth = _config.VisionOutHiddenSize;

        // Qwen3_5VisionPatchMerger (use_postshuffle_norm=False): LayerNorm over the patch width,
        // then view(-1, 4 · width) — four consecutive patches, which is one 2×2 block.
        using PooledBuffer normed = TensorPool.Rent(tokens * width);
        hidden.AsSpan().CopyTo(normed.Span);
        Norms.LayerNormParallel(normed.Memory[..(tokens * width)], width, _mergerNormWeight, _mergerNormBias, 1e-6f);

        using PooledBuffer projected = TensorPool.Rent(merged * mergedWidth);
        Gemm.Linear(normed.Memory, merged, mergedWidth, _mergerFc1, _mergerFc1Bias, projected.Memory, mergedWidth);
        Kernels.GeluErfParallel(projected.Memory[..(merged * mergedWidth)]);

        var output = new float[merged * outWidth];
        Gemm.Linear(projected.Memory, merged, mergedWidth, _mergerFc2, _mergerFc2Bias, output, outWidth);
        trace?.Invoke("vis.merged", output, outWidth);
        return output;
    }

    /// <summary>Patch (row, column) in spatial-merge-block order.</summary>
    private (int Row, int Col) PatchPosition(int index, int gridW)
    {
        int m = _config.SpatialMergeSize;
        int blocksW = gridW / m;
        int inCol = index % m;
        int inRow = (index / m) % m;
        int blockCol = (index / (m * m)) % blocksW;
        int blockRow = index / (m * m * blocksW);
        return ((blockRow * m) + inRow, (blockCol * m) + inCol);
    }

    private void AddPositionEmbedding(float[] hidden, PreprocessedImage image)
    {
        int width = _config.VisionHiddenSize;
        int side = _gridSide;

        Parallel.For(0, image.PatchCount, Parallelism.Options, index =>
        {
            (int row, int col) = PatchPosition(index, image.GridW);
            (int h0, int h1, float hw0, float hw1) = Taps(row, image.GridH, side);
            (int w0, int w1, float ww0, float ww1) = Taps(col, image.GridW, side);

            // (pos_embed(indices) * weights[:, :, None]).sum(1), taps in (h, w) row-major order.
            ReadOnlySpan<float> table = _positionTable;
            ReadOnlySpan<float> t00 = table.Slice(((h0 * side) + w0) * width, width);
            ReadOnlySpan<float> t01 = table.Slice(((h0 * side) + w1) * width, width);
            ReadOnlySpan<float> t10 = table.Slice(((h1 * side) + w0) * width, width);
            ReadOnlySpan<float> t11 = table.Slice(((h1 * side) + w1) * width, width);
            float a = hw0 * ww0, b = hw0 * ww1, c = hw1 * ww0, d = hw1 * ww1;
            Span<float> target = hidden.AsSpan(index * width, width);
            for (int i = 0; i < width; i++)
            {
                float position = (((t00[i] * a) + (t01[i] * b)) + (t10[i] * c)) + (t11[i] * d);
                target[i] += position;
            }
        });
    }

    /// <summary>
    /// <c>_interpolation_axis_taps_weights(mode="bilinear", align_corners=True)</c>, in float32.
    /// </summary>
    private static (int Tap0, int Tap1, float Weight0, float Weight1) Taps(int index, int size, int side)
    {
        float source = index * (float)(side - 1) / Math.Max(size - 1, 1);
        float floor = MathF.Floor(source);
        int tap = (int)floor;
        float fraction = source - floor;
        float weight0 = Math.Max(1f - MathF.Abs(fraction), 0f);
        float weight1 = Math.Max(1f - MathF.Abs(fraction - 1f), 0f);
        return (Math.Clamp(tap, 0, side - 1), Math.Clamp(tap + 1, 0, side - 1), weight0, weight1);
    }

    private void FillRotary(PreprocessedImage image, Span<float> cos, Span<float> sin)
    {
        int headDim = _config.VisionHeadDim;
        int quarter = _inverseFrequency.Length;
        for (int index = 0; index < image.PatchCount; index++)
        {
            (int row, int col) = PatchPosition(index, image.GridW);
            Span<float> c = cos.Slice(index * headDim, headDim);
            Span<float> s = sin.Slice(index * headDim, headDim);
            for (int i = 0; i < quarter; i++)
            {
                float hAngle = row * _inverseFrequency[i];
                float wAngle = col * _inverseFrequency[i];
                c[i] = c[i + (2 * quarter)] = MathF.Cos(hAngle);
                s[i] = s[i + (2 * quarter)] = MathF.Sin(hAngle);
                c[i + quarter] = c[i + (3 * quarter)] = MathF.Cos(wAngle);
                s[i + quarter] = s[i + (3 * quarter)] = MathF.Sin(wAngle);
            }
        }
    }

    private static void SplitHeads(
        ReadOnlyMemory<float> qkv,
        Memory<float> queries,
        Memory<float> keys,
        Memory<float> values,
        int tokens,
        int heads,
        int headDim,
        ReadOnlyMemory<float> cos,
        ReadOnlyMemory<float> sin)
    {
        int width = heads * headDim;
        Parallel.For(0, heads, Parallelism.Options, head =>
        {
            ReadOnlySpan<float> source = qkv.Span;
            Span<float> q = queries.Span;
            Span<float> k = keys.Span;
            Span<float> v = values.Span;
            for (int token = 0; token < tokens; token++)
            {
                int target = ((head * tokens) + token) * headDim;
                int row = token * 3 * width;
                ReadOnlySpan<float> c = cos.Span.Slice(token * headDim, headDim);
                ReadOnlySpan<float> s = sin.Span.Slice(token * headDim, headDim);
                Rotary.Apply(source.Slice(row + (head * headDim), headDim), c, s, q.Slice(target, headDim));
                Rotary.Apply(source.Slice(row + width + (head * headDim), headDim), c, s, k.Slice(target, headDim));
                source.Slice(row + (2 * width) + (head * headDim), headDim).CopyTo(v.Slice(target, headDim));
            }
        });
    }

    private static void MergeHeads(ReadOnlyMemory<float> perHead, Memory<float> destination, int tokens, int heads, int headDim)
    {
        int width = heads * headDim;
        Parallel.For(0, heads, Parallelism.Options, head =>
        {
            ReadOnlySpan<float> source = perHead.Span;
            Span<float> target = destination.Span;
            for (int token = 0; token < tokens; token++)
            {
                source.Slice(((head * tokens) + token) * headDim, headDim)
                    .CopyTo(target.Slice((token * width) + (head * headDim), headDim));
            }
        });
    }

    private sealed record Block(
        float[] Norm1Weight,
        float[] Norm1Bias,
        float[] Norm2Weight,
        float[] Norm2Bias,
        WeightMatrix Qkv,
        float[] QkvBias,
        WeightMatrix Proj,
        float[] ProjBias,
        WeightMatrix Fc1,
        float[] Fc1Bias,
        WeightMatrix Fc2,
        float[] Fc2Bias);
}

/// <summary>Rotary embedding application, <c>x · cos + rotate_half(x) · sin</c>.</summary>
internal static class Rotary
{
    /// <summary>Rotates the whole of <paramref name="x"/> into <paramref name="destination"/>.</summary>
    public static void Apply(ReadOnlySpan<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, Span<float> destination)
    {
        int half = x.Length / 2;
        for (int i = 0; i < half; i++)
        {
            float a = x[i];
            float b = x[i + half];
            destination[i] = (a * cos[i]) + (-b * sin[i]);
            destination[i + half] = (b * cos[i + half]) + (a * sin[i + half]);
        }
    }

    /// <summary>Rotates the first <c>cos.Length</c> elements in place, leaving the rest.</summary>
    public static void ApplyPartial(Span<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = cos.Length / 2;
        for (int i = 0; i < half; i++)
        {
            float a = x[i];
            float b = x[i + half];
            x[i] = (a * cos[i]) + (-b * sin[i]);
            x[i + half] = (b * cos[i + half]) + (a * sin[i + half]);
        }
    }
}
