using System.Numerics.Tensors;

namespace IndraOcr.Onnx;

internal static partial class Operators
{
    internal enum ReduceOp { Sum, Mean, Max }

    internal static Tensor Reduce(OnnxNode node, List<Tensor?> x, ReduceOp op)
    {
        Tensor t = x[0]!;
        long[]? axesSpec = x.Count > 1 && x[1] is not null ? x[1]!.ToLongArray() : node.Ints("axes");
        bool keep = node.Int("keepdims", 1) != 0;
        int rank = t.Rank;
        bool[] reduce = new bool[rank];
        if (axesSpec is null || axesSpec.Length == 0)
        {
            if (node.Int("noop_with_empty_axes", 0) != 0)
            {
                return t.Reshape((int[])t.Shape.Clone());
            }

            Array.Fill(reduce, true);
        }
        else
        {
            foreach (long a in axesSpec)
            {
                reduce[Axis(a, rank)] = true;
            }
        }

        var outShape = new List<int>();
        var keptShape = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            keptShape[d] = reduce[d] ? 1 : t.Shape[d];
            if (!reduce[d])
            {
                outShape.Add(t.Shape[d]);
            }
            else if (keep)
            {
                outShape.Add(1);
            }
        }

        int outLen = Tensor.SizeOf(keptShape);
        int count = outLen == 0 ? 0 : t.Length / outLen;
        if (t.Type == ElementType.Int64)
        {
            Tensor ri = Tensor.Int64([.. outShape], clear: op != ReduceOp.Max);
            if (op == ReduceOp.Max)
            {
                ri.Longs.Fill(long.MinValue);
            }

            int[] oi = Broadcast.Offsets(keptShape, t.Shape);
            Span<long> src = t.Longs, dst = ri.Longs;
            for (int i = 0; i < src.Length; i++)
            {
                dst[oi[i]] = op == ReduceOp.Max ? Math.Max(dst[oi[i]], src[i]) : dst[oi[i]] + src[i];
            }

            if (op == ReduceOp.Mean)
            {
                for (int i = 0; i < dst.Length; i++)
                {
                    dst[i] /= Math.Max(count, 1);
                }
            }

            return ri;
        }

        Tensor result = Tensor.Float([.. outShape]);
        Span<float> output = result.Floats;

        // Common case: the reduced axes are a contiguous suffix (e.g. global pooling, last axis).
        int firstReduced = Array.IndexOf(reduce, true);
        bool suffix = firstReduced >= 0 && reduce.Skip(firstReduced).All(r => r);
        if (suffix)
        {
            ReadOnlySpan<float> src = t.Floats;
            for (int o = 0; o < outLen; o++)
            {
                ReadOnlySpan<float> row = src.Slice(o * count, count);
                output[o] = op switch
                {
                    ReduceOp.Max => count == 0 ? float.NegativeInfinity : TensorPrimitives.Max(row),
                    ReduceOp.Sum => TensorPrimitives.Sum(row),
                    _ => TensorPrimitives.Sum(row) / count,
                };
            }

            return result;
        }

        // General case: accumulate through broadcast offsets.
        int[] offsets = Broadcast.Offsets(keptShape, t.Shape);
        if (op == ReduceOp.Max)
        {
            output.Fill(float.NegativeInfinity);
        }
        else
        {
            output.Clear();
        }

        ReadOnlySpan<float> data = t.Floats;
        for (int i = 0; i < data.Length; i++)
        {
            int o = offsets[i];
            output[o] = op == ReduceOp.Max ? MathF.Max(output[o], data[i]) : output[o] + data[i];
        }

        if (op == ReduceOp.Mean && count > 0)
        {
            TensorPrimitives.Multiply(output, 1f / count, output);
        }

        return result;
    }

    internal static Tensor ArgMax(Tensor t, int axis, bool keepDims, bool selectLast)
    {
        axis = Axis(axis, t.Rank);
        int outer = 1, inner = 1, dim = t.Shape[axis];
        for (int i = 0; i < axis; i++)
        {
            outer *= t.Shape[i];
        }

        for (int i = axis + 1; i < t.Rank; i++)
        {
            inner *= t.Shape[i];
        }

        var shape = new List<int>();
        for (int i = 0; i < t.Rank; i++)
        {
            if (i != axis)
            {
                shape.Add(t.Shape[i]);
            }
            else if (keepDims)
            {
                shape.Add(1);
            }
        }

        Tensor result = Tensor.Int64([.. shape]);
        float[] data = t.FloatArray;
        long[] dst = result.LongArray;
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int best = 0;
                float bestValue = data[(o * dim * inner) + i];
                for (int k = 1; k < dim; k++)
                {
                    float v = data[(((o * dim) + k) * inner) + i];
                    if (v > bestValue || (selectLast && v == bestValue))
                    {
                        bestValue = v;
                        best = k;
                    }
                }

                dst[(o * inner) + i] = best;
            }
        }

        return result;
    }

    internal static Tensor Softmax(Tensor t, int axis)
    {
        axis = Axis(axis, t.Rank);
        Tensor result = Tensor.Float((int[])t.Shape.Clone());
        int dim = t.Shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < t.Rank; i++)
        {
            inner *= t.Shape[i];
        }

        int outer = dim == 0 ? 0 : t.Length / (dim * inner);
        float[] src = t.FloatArray, dst = result.FloatArray;
        if (inner == 1)
        {
            Core.Parallelism.For(outer, (long)t.Length * 4, (s, e) =>
            {
                for (int o = s; o < e; o++)
                {
                    SoftmaxRow(src.AsSpan(o * dim, dim), dst.AsSpan(o * dim, dim));
                }
            });
            return result;
        }

        var row = new float[dim];
        var outRow = new float[dim];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                for (int k = 0; k < dim; k++)
                {
                    row[k] = src[(((o * dim) + k) * inner) + i];
                }

                SoftmaxRow(row, outRow);
                for (int k = 0; k < dim; k++)
                {
                    dst[(((o * dim) + k) * inner) + i] = outRow[k];
                }
            }
        }

        return result;
    }

    /// <summary>Softmax of one row: subtract the maximum, exponentiate, divide by the sum.</summary>
    internal static void SoftmaxRow(ReadOnlySpan<float> x, Span<float> z)
    {
        if (x.Length == 0)
        {
            return;
        }

        float max = TensorPrimitives.Max(x);
        if (float.IsNegativeInfinity(max))
        {
            z.Fill(float.NaN);
            return;
        }

        TensorPrimitives.Subtract(x, max, z);
        TensorPrimitives.Exp(z, z);
        float sum = TensorPrimitives.Sum(z);
        TensorPrimitives.Divide(z, sum, z);
    }
}
