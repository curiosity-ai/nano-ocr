using System.Runtime.InteropServices;
using IndraOcr.Vlm.Core;
using IndraOcr.Vlm.Formats;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>Receives an intermediate tensor, flattened row-major, in the reference dump's layout.</summary>
/// <param name="name">The dump's name for it (see <c>tools/reference/dump_surya_layout.py</c>).</param>
/// <param name="values">The values.</param>
public delegate void LayoutTrace(string name, ReadOnlySpan<float> values);

/// <summary>A torch state dict, by parameter name, with typed accessors for the layers below.</summary>
internal sealed class StateDict(Dictionary<string, WeightTensor> tensors)
{
    public bool Contains(string name) => tensors.ContainsKey(name);

    public WeightTensor Get(string name) =>
        tensors.TryGetValue(name, out WeightTensor t) ? t : throw new KeyNotFoundException($"Checkpoint has no '{name}'.");

    public float[] Floats(string name) => Get(name).ToFloats();

    public int[] Shape(string name) => Get(name).Shape;

    /// <summary>A float32 tensor viewed as a <c>[rows, cols]</c> matrix, whatever its rank.</summary>
    public WeightMatrix Matrix(string name, int rows, int cols)
    {
        WeightTensor t = Get(name);
        return t.Dtype == DType.Float32
            ? WeightMatrix.Create(t.Bytes, DType.Float32, rows, cols)
            : WeightMatrix.Create(MemoryMarshal.AsBytes(t.ToFloats().AsSpan()).ToArray(), DType.Float32, rows, cols);
    }

    public Dense Dense(string prefix, bool bias = true)
    {
        int[] shape = Shape(prefix + ".weight");
        int cols = 1;
        for (int i = 1; i < shape.Length; i++)
        {
            cols *= shape[i];
        }

        return new Dense(Matrix(prefix + ".weight", shape[0], cols), bias ? Floats(prefix + ".bias") : null);
    }

    public LayerNormLayer Norm(string prefix, float epsilon) =>
        new(Floats(prefix + ".weight"), Floats(prefix + ".bias"), epsilon);
}

/// <summary><c>nn.Linear</c> (and any convolution posed as one): <c>y = x · Wᵀ + b</c>.</summary>
/// <remarks>
/// The weight is transposed once, at load, to <c>[in, out]</c>, so the product runs on the packed
/// register-tiled kernel of <see cref="IndraOcr.Core.Gemm"/> (an outer-product micro-kernel) rather
/// than on dot products. These models are small, so the transposed copy simply replaces the
/// checkpoint's layout.
/// </remarks>
internal sealed class Dense
{
    private readonly float[] _transposed;

    public Dense(WeightMatrix weight, float[]? bias)
    {
        Out = weight.Rows;
        In = weight.Cols;
        float[] w = new float[Out * In];
        weight.CopyTo(w);
        _transposed = new float[In * Out];
        IndraOcr.Core.Gemm.Transpose(w, Out, In, _transposed);
        Bias = bias;
    }

    public int In { get; }

    public int Out { get; }

    public float[]? Bias { get; }

    /// <summary>Applies the layer to <paramref name="rows"/> rows of <see cref="In"/> values.</summary>
    public float[] Forward(ReadOnlyMemory<float> x, int rows)
    {
        float[] y = new float[rows * Out];
        IndraOcr.Core.Gemm.Multiply(rows, Out, In, x.Span, In, _transposed, Out, y, Out);
        if (Bias is not null)
        {
            for (int r = 0; r < rows; r++)
            {
                LayoutMath.Add(y.AsSpan(r * Out, Out), Bias);
            }
        }

        return y;
    }

    /// <summary>A slice of the output rows of the weight, as its own layer (a packed in-projection's q, k or v).</summary>
    public static Dense Rows(StateDict state, string prefix, int start, int count)
    {
        WeightTensor w = state.Get(prefix + "_weight");
        int cols = w.Shape[1];
        WeightMatrix m = WeightMatrix.Create(w.Bytes.Slice(start * cols * 4, count * cols * 4), DType.Float32, count, cols);
        float[] b = state.Floats(prefix + "_bias").AsSpan(start, count).ToArray();
        return new Dense(m, b);
    }
}

/// <summary><c>nn.LayerNorm</c> over the last dimension (also the projector's channel <c>LayerNorm</c> on channels-last data).</summary>
internal sealed class LayerNormLayer(float[] weight, float[] bias, float epsilon)
{
    public void Apply(float[] values) =>
        Norms.LayerNormParallel(values, weight.Length, weight, bias, epsilon);
}

/// <summary><c>MLP</c> of <c>lwdetr.py</c> / <c>transformer.py</c>: Linear layers with ReLU between them.</summary>
internal sealed class Mlp(Dense[] layers)
{
    public static Mlp Load(StateDict state, string prefix, int count) =>
        new(Enumerable.Range(0, count).Select(i => state.Dense($"{prefix}.layers.{i}")).ToArray());

    public float[] Forward(float[] x, int rows)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            x = layers[i].Forward(x, rows);
            if (i < layers.Length - 1)
            {
                LayoutMath.Relu(x);
            }
        }

        return x;
    }
}

/// <summary>
/// <c>nn.MultiheadAttention</c> (batch-first, no dropout): a packed <c>in_proj</c> split into q, k, v,
/// scaled dot-product attention per head, and <c>out_proj</c>.
/// </summary>
internal sealed class MultiHeadAttention
{
    private readonly Dense _q, _k, _v, _out;
    private readonly int _heads;

    public MultiHeadAttention(StateDict state, string prefix, int dim, int heads)
    {
        _q = Dense.Rows(state, prefix + ".in_proj", 0, dim);
        _k = Dense.Rows(state, prefix + ".in_proj", dim, dim);
        _v = Dense.Rows(state, prefix + ".in_proj", 2 * dim, dim);
        _out = state.Dense(prefix + ".out_proj");
        _heads = heads;
    }

    public int Dim => _q.Out;

    public float[] ProjectQuery(float[] x, int rows) => _q.Forward(x, rows);

    public float[] ProjectKey(float[] x, int rows) => _k.Forward(x, rows);

    public float[] ProjectValue(float[] x, int rows) => _v.Forward(x, rows);

    public float[] Output(float[] context, int rows) => _out.Forward(context, rows);

    /// <summary><c>attn(query, key, value)[0]</c> with no mask.</summary>
    public float[] Forward(float[] query, int queries, float[] key, float[] value, int keys)
    {
        float[] q = _q.Forward(query, queries);
        float[] k = _k.Forward(key, keys);
        float[] v = _v.Forward(value, keys);
        float[] context = LayoutMath.Attend(q, queries, k, v, keys, _heads, Dim / _heads, causal: false);
        return _out.Forward(context, queries);
    }

    /// <summary>Attention of already-projected rows; <see cref="LayoutMath.Attend"/>.</summary>
    public float[] Attend(float[] q, int queries, float[] k, float[] v, int keys) =>
        LayoutMath.Attend(q, queries, k, v, keys, _heads, Dim / _heads, causal: false);
}

/// <summary>Small element-wise and attention helpers.</summary>
internal static class LayoutMath
{
    public static void Relu(Span<float> x) => System.Numerics.Tensors.TensorPrimitives.Max(x, 0f, x);

    public static void Add(Span<float> x, ReadOnlySpan<float> y) => System.Numerics.Tensors.TensorPrimitives.Add(x, y, x);

    public static float[] Sum(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        float[] r = new float[x.Length];
        System.Numerics.Tensors.TensorPrimitives.Add(x, y, r);
        return r;
    }

    /// <summary>
    /// <c>softmax(q·kᵀ / √d)·v</c> per head over row-major <c>[tokens, heads·d]</c> operands — what
    /// <c>scaled_dot_product_attention</c> computes for <c>nn.MultiheadAttention</c>.
    /// </summary>
    public static float[] Attend(
        float[] q, int queries, float[] k, float[] v, int keys, int heads, int headDim, bool causal)
    {
        int width = heads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);
        float[] output = new float[queries * width];
        Parallel.For(0, heads * queries, Parallelism.Options, () => new float[keys], (index, _, scores) =>
        {
            int h = index / queries, i = index % queries;
            ReadOnlySpan<float> qi = q.AsSpan((i * width) + (h * headDim), headDim);
            int visible = causal ? Math.Min(keys, i + 1 + (keys - queries)) : keys;
            Span<float> row = scores.AsSpan(0, visible);
            for (int j = 0; j < visible; j++)
            {
                row[j] = Gemm.Dot(qi, k.AsSpan((j * width) + (h * headDim), headDim));
            }

            Kernels.ScaledSoftmax(row, scale);
            Span<float> o = output.AsSpan((i * width) + (h * headDim), headDim);
            for (int j = 0; j < visible; j++)
            {
                Kernels.AddScaled(o, v.AsSpan((j * width) + (h * headDim), headDim), row[j]);
            }

            return scores;
        }, _ => { });
        return output;
    }

    /// <summary><c>[rows, cols]</c> → <c>[cols, rows]</c>.</summary>
    public static float[] Transpose(ReadOnlySpan<float> x, int rows, int cols)
    {
        float[] t = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                t[(c * rows) + r] = x[(r * cols) + c];
            }
        }

        return t;
    }
}
