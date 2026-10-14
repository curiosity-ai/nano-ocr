using System.Runtime.Intrinsics;

namespace NanoOcr.Onnx;

internal static partial class Operators
{
    internal static void Copy(Tensor src, int srcOffset, Tensor dst, int dstOffset, int count)
    {
        switch (src.Type)
        {
            case ElementType.Float:
                Array.Copy(src.FloatArray, srcOffset, dst.FloatArray, dstOffset, count);
                break;
            case ElementType.Int64:
                Array.Copy(src.LongArray, srcOffset, dst.LongArray, dstOffset, count);
                break;
            default:
                Array.Copy(src.BoolArray, srcOffset, dst.BoolArray, dstOffset, count);
                break;
        }
    }

    internal static int[] ToInts(Tensor t)
    {
        var r = new int[t.Length];
        Span<long> v = t.Longs;
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = checked((int)Math.Clamp(v[i], int.MinValue, int.MaxValue));
        }

        return r;
    }

    internal static Tensor ShapeOf(OnnxNode node, Tensor x)
    {
        int rank = x.Rank;
        int start = (int)node.Int("start", 0);
        int end = (int)node.Int("end", rank);
        start = Math.Clamp(start < 0 ? start + rank : start, 0, rank);
        end = Math.Clamp(end < 0 ? end + rank : end, 0, rank);
        int n = Math.Max(0, end - start);
        var data = new long[Math.Max(1, n)];
        for (int i = 0; i < n; i++)
        {
            data[i] = x.Shape[start + i];
        }

        return Tensor.FromLongs(data, n);
    }

    internal static Tensor Reshape(Tensor x, Tensor shapeTensor, bool allowZero)
    {
        long[] spec = shapeTensor.ToLongArray();
        var shape = new int[spec.Length];
        int infer = -1;
        long known = 1;
        for (int i = 0; i < spec.Length; i++)
        {
            long v = spec[i];
            if (v == 0 && !allowZero)
            {
                v = x.Shape[i];
            }

            if (v == -1)
            {
                infer = i;
                shape[i] = 1;
                continue;
            }

            shape[i] = (int)v;
            known *= v;
        }

        if (infer >= 0)
        {
            shape[infer] = known == 0 ? 0 : (int)(x.Length / known);
        }

        return x.Reshape(shape);
    }

    internal static Tensor Flatten(Tensor x, int axis)
    {
        axis = Axis(axis, x.Rank);
        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= x.Shape[i];
        }

        return x.Reshape([outer, outer == 0 ? 0 : x.Length / outer]);
    }

    internal static Tensor Squeeze(OnnxNode node, List<Tensor?> x)
    {
        Tensor t = x[0]!;
        long[]? axes = x.Count > 1 && x[1] is not null ? x[1]!.ToLongArray() : node.Ints("axes");
        var keep = new List<int>();
        var drop = new HashSet<int>((axes ?? []).Select(a => Axis(a, t.Rank)));
        for (int i = 0; i < t.Rank; i++)
        {
            bool squeeze = axes is null ? t.Shape[i] == 1 : drop.Contains(i);
            if (!squeeze)
            {
                keep.Add(t.Shape[i]);
            }
        }

        return t.Reshape([.. keep]);
    }

    internal static Tensor Unsqueeze(OnnxNode node, List<Tensor?> x)
    {
        Tensor t = x[0]!;
        long[] axes = x.Count > 1 && x[1] is not null ? x[1]!.ToLongArray() : node.Ints("axes")!;
        int rank = t.Rank + axes.Length;
        var inserted = new HashSet<int>(axes.Select(a => Axis(a, rank)));
        var shape = new int[rank];
        int src = 0;
        for (int i = 0; i < rank; i++)
        {
            shape[i] = inserted.Contains(i) ? 1 : t.Shape[src++];
        }

        return t.Reshape(shape);
    }

    internal static Tensor Concat(List<Tensor?> inputs, int axis)
    {
        var parts = inputs.Where(t => t is not null).Select(t => t!).ToList();
        Tensor first = parts.First(p => p.Length > 0 || parts.All(q => q.Length == 0));
        int rank = first.Rank;
        axis = Axis(axis, rank);
        int[] shape = (int[])first.Shape.Clone();
        shape[axis] = parts.Where(p => p.Rank == rank).Sum(p => p.Shape[axis]);
        Tensor result = Tensor.Create(first.Type, shape);
        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= shape[i];
        }

        int inner = 1;
        for (int i = axis + 1; i < rank; i++)
        {
            inner *= shape[i];
        }

        int outRow = shape[axis] * inner;
        int offset = 0;
        foreach (Tensor p in parts)
        {
            if (p.Rank != rank)
            {
                continue;
            }

            int row = p.Shape[axis] * inner;
            for (int o = 0; o < outer; o++)
            {
                Copy(p, o * row, result, (o * outRow) + offset, row);
            }

            offset += row;
        }

        return result;
    }

    internal static Tensor[] Split(OnnxNode node, List<Tensor?> x)
    {
        Tensor t = x[0]!;
        int axis = Axis(node.Int("axis", 0), t.Rank);
        int count = node.Outputs.Count;
        long[]? split = x.Count > 1 && x[1] is not null ? x[1]!.ToLongArray() : node.Ints("split");
        if (split is null)
        {
            int dim = t.Shape[axis];
            int size = (dim + count - 1) / count;
            split = new long[count];
            for (int i = 0; i < count; i++)
            {
                split[i] = Math.Min(size, dim - (i * size));
            }
        }

        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= t.Shape[i];
        }

        int inner = 1;
        for (int i = axis + 1; i < t.Rank; i++)
        {
            inner *= t.Shape[i];
        }

        int inRow = t.Shape[axis] * inner;
        var results = new Tensor[split.Length];
        int start = 0;
        for (int s = 0; s < split.Length; s++)
        {
            int[] shape = (int[])t.Shape.Clone();
            shape[axis] = (int)split[s];
            Tensor r = Tensor.Create(t.Type, shape);
            int row = shape[axis] * inner;
            for (int o = 0; o < outer; o++)
            {
                Copy(t, (o * inRow) + (start * inner), r, o * row, row);
            }

            start += shape[axis];
            results[s] = r;
        }

        return results;
    }

    internal static Tensor Slice(List<Tensor?> x)
    {
        Tensor t = x[0]!;
        long[] starts = x[1]!.ToLongArray();
        long[] ends = x[2]!.ToLongArray();
        long[] axes = x.Count > 3 && x[3] is not null ? x[3]!.ToLongArray() : Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
        long[] steps = x.Count > 4 && x[4] is not null ? x[4]!.ToLongArray() : Enumerable.Repeat(1L, starts.Length).ToArray();
        int rank = t.Rank;
        var begin = new int[rank];
        var step = new int[rank];
        var shape = (int[])t.Shape.Clone();
        for (int d = 0; d < rank; d++)
        {
            step[d] = 1;
        }

        for (int i = 0; i < starts.Length; i++)
        {
            int d = Axis(axes[i], rank);
            long dim = t.Shape[d];
            long st = steps[i];
            long s = starts[i], e = ends[i];
            if (s < 0)
            {
                s += dim;
            }

            if (e < 0)
            {
                e += dim;
            }

            if (st > 0)
            {
                s = Math.Clamp(s, 0, dim);
                e = Math.Clamp(e, 0, dim);
                shape[d] = (int)Math.Max(0, (e - s + st - 1) / st);
            }
            else
            {
                s = Math.Clamp(s, -1, dim - 1);
                e = Math.Clamp(e, -1, dim - 1);
                shape[d] = (int)Math.Max(0, (s - e + (-st) - 1) / (-st));
            }

            begin[d] = (int)s;
            step[d] = (int)st;
        }

        Tensor result = Tensor.Create(t.Type, shape);
        if (result.Length == 0)
        {
            return result;
        }

        int[] srcStrides = Tensor.Strides(t.Shape);
        // Copy the innermost dimension as a block when its step is 1.
        int last = rank - 1;
        bool contiguous = rank > 0 && step[last] == 1;
        int run = contiguous ? shape[last] : 1;
        int outerRank = contiguous ? rank - 1 : rank;
        int outerCount = result.Length / run;
        var counter = new int[Math.Max(outerRank, 1)];
        for (int o = 0; o < outerCount; o++)
        {
            int src = 0;
            for (int d = 0; d < outerRank; d++)
            {
                src += (begin[d] + (counter[d] * step[d])) * srcStrides[d];
            }

            if (contiguous)
            {
                src += begin[last];
            }

            Copy(t, src, result, o * run, run);
            for (int d = outerRank - 1; d >= 0; d--)
            {
                if (++counter[d] < shape[d])
                {
                    break;
                }

                counter[d] = 0;
            }
        }

        return result;
    }

    internal static Tensor Gather(Tensor data, Tensor indices, int axis)
    {
        axis = Axis(axis, data.Rank);
        long[] idx = indices.ToLongArray();
        int dim = data.Shape[axis];
        var shape = new List<int>();
        shape.AddRange(data.Shape.Take(axis));
        shape.AddRange(indices.Shape);
        shape.AddRange(data.Shape.Skip(axis + 1));
        Tensor result = Tensor.Create(data.Type, [.. shape]);
        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= data.Shape[i];
        }

        int inner = 1;
        for (int i = axis + 1; i < data.Rank; i++)
        {
            inner *= data.Shape[i];
        }

        int dst = 0;
        for (int o = 0; o < outer; o++)
        {
            foreach (long raw in idx)
            {
                long j = raw < 0 ? raw + dim : raw;
                Copy(data, (int)((((long)o * dim) + j) * inner), result, dst, inner);
                dst += inner;
            }
        }

        return result;
    }

    internal static Tensor Transpose(Tensor x, long[]? permSpec)
    {
        int rank = x.Rank;
        int[] perm = permSpec?.Select(p => (int)p).ToArray() ?? Enumerable.Range(0, rank).Reverse().ToArray();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = x.Shape[perm[i]];
        }

        Tensor result = Tensor.Create(x.Type, shape);
        int[] srcStrides = Tensor.Strides(x.Shape);
        var strides = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            strides[i] = srcStrides[perm[i]];
        }

        int n = result.Length;
        if (n == 0)
        {
            return result;
        }

        // Fast path: a 2-D transpose of the last two axes over a batch.
        if (x.Type == ElementType.Float && rank >= 2 && perm[rank - 1] == rank - 2 && perm[rank - 2] == rank - 1
            && Enumerable.Range(0, rank - 2).All(i => perm[i] == i))
        {
            int rows = x.Shape[rank - 2], cols = x.Shape[rank - 1];
            Core.Gemm.TransposeBatch(x.FloatArray, result.FloatArray, n / (rows * cols), rows, cols);
            return result;
        }

        int innerLen = shape[rank - 1];
        int innerStride = strides[rank - 1];
        var counter = new int[rank];
        int outer = n / innerLen;
        int srcBase = 0;
        for (int o = 0; o < outer; o++)
        {
            int dst = o * innerLen;
            switch (x.Type)
            {
                case ElementType.Float:
                    {
                        float[] s = x.FloatArray, d = result.FloatArray;
                        if (innerStride == 1)
                        {
                            s.AsSpan(srcBase, innerLen).CopyTo(d.AsSpan(dst, innerLen));
                        }
                        else if (innerStride == 2)
                        {
                            // A space-to-depth's inner run (every other element): vector deinterleave.
                            Deinterleave(s.AsSpan(srcBase, (2 * innerLen) - 1), 0, d.AsSpan(dst, innerLen));
                        }
                        else
                        {
                            for (int i = 0, p = srcBase; i < innerLen; i++, p += innerStride)
                            {
                                d[dst + i] = s[p];
                            }
                        }

                        break;
                    }

                case ElementType.Int64:
                    {
                        long[] s = x.LongArray, d = result.LongArray;
                        for (int i = 0, p = srcBase; i < innerLen; i++, p += innerStride)
                        {
                            d[dst + i] = s[p];
                        }

                        break;
                    }

                default:
                    {
                        bool[] s = x.BoolArray, d = result.BoolArray;
                        for (int i = 0, p = srcBase; i < innerLen; i++, p += innerStride)
                        {
                            d[dst + i] = s[p];
                        }

                        break;
                    }
            }

            for (int d = rank - 2; d >= 0; d--)
            {
                counter[d]++;
                srcBase += strides[d];
                if (counter[d] < shape[d])
                {
                    break;
                }

                srcBase -= strides[d] * counter[d];
                counter[d] = 0;
            }
        }

        return result;
    }

    internal static Tensor Expand(Tensor x, Tensor shapeTensor)
    {
        int[] target = ToInts(shapeTensor);
        int[] shape = Broadcast.Shape(x.Shape, target);
        Tensor result = Tensor.Create(x.Type, shape);
        int[] offsets = Broadcast.Offsets(x.Shape, shape);
        for (int i = 0; i < result.Length; i++)
        {
            Copy(x, offsets[i], result, i, 1);
        }

        return result;
    }

    internal static Tensor ConstantOfShape(OnnxNode node, Tensor shapeTensor)
    {
        int[] shape = ToInts(shapeTensor);
        Tensor? value = node.Attributes.TryGetValue("value", out var a) ? a.T : null;
        if (value is null)
        {
            return Tensor.Float(shape, clear: true);
        }

        Tensor result = Tensor.Create(value.Type, shape);
        switch (value.Type)
        {
            case ElementType.Float: result.Floats.Fill(value.Floats[0]); break;
            case ElementType.Int64: result.Longs.Fill(value.Longs[0]); break;
            default: result.Bools.Fill(value.Bools[0]); break;
        }

        return result;
    }

    internal static Tensor Range(Tensor start, Tensor limit, Tensor delta)
    {
        if (start.Type == ElementType.Float)
        {
            float s = start.Floats[0], l = limit.Floats[0], d = delta.Floats[0];
            int n = Math.Max(0, (int)MathF.Ceiling((l - s) / d));
            Tensor r = Tensor.Float([n]);
            for (int i = 0; i < n; i++)
            {
                r.FloatArray[i] = s + (i * d);
            }

            return r;
        }
        else
        {
            long s = start.Longs[0], l = limit.Longs[0], d = delta.Longs[0];
            int n = (int)Math.Max(0, (long)Math.Ceiling((double)(l - s) / d));
            Tensor r = Tensor.Int64([n]);
            for (int i = 0; i < n; i++)
            {
                r.LongArray[i] = s + (i * d);
            }

            return r;
        }
    }

    internal static Tensor ScatterNd(Tensor data, Tensor indices, Tensor updates)
    {
        Tensor result = Tensor.Create(data.Type, (int[])data.Shape.Clone());
        Copy(data, 0, result, 0, data.Length);
        int k = indices.Shape[^1];
        int count = indices.Length / Math.Max(k, 1);
        int[] strides = Tensor.Strides(data.Shape);
        int slice = 1;
        for (int i = k; i < data.Rank; i++)
        {
            slice *= data.Shape[i];
        }

        long[] idx = indices.ToLongArray();
        for (int u = 0; u < count; u++)
        {
            int offset = 0;
            for (int j = 0; j < k; j++)
            {
                long v = idx[(u * k) + j];
                if (v < 0)
                {
                    v += data.Shape[j];
                }

                offset += (int)v * strides[j];
            }

            Copy(updates, u * slice, result, offset, slice);
        }

        return result;
    }

    /// <summary><c>dst[2i] = a[i]</c>, <c>dst[2i + 1] = b[i]</c>.</summary>
    private static void Interleave(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
    {
        int i = 0;
        for (; i + 8 <= a.Length; i += 8)
        {
            var va = System.Runtime.Intrinsics.Vector256.Create(a.Slice(i, 8));
            var vb = System.Runtime.Intrinsics.Vector256.Create(b.Slice(i, 8));
            var lo = System.Runtime.Intrinsics.X86.Avx.UnpackLow(va, vb);
            var hi = System.Runtime.Intrinsics.X86.Avx.UnpackHigh(va, vb);
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(lo, hi, 0x20).CopyTo(dst.Slice(2 * i, 8));
            System.Runtime.Intrinsics.X86.Avx.Permute2x128(lo, hi, 0x31).CopyTo(dst.Slice((2 * i) + 8, 8));
        }

        for (; i < a.Length; i++)
        {
            dst[2 * i] = a[i];
            dst[(2 * i) + 1] = b[i];
        }
    }

    internal static Tensor DepthToSpace(Tensor x, int block, string mode)
    {
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        int oc = c / (block * block);
        Tensor result = Tensor.Float([n, oc, h * block, w * block]);
        float[] s = x.FloatArray, d = result.FloatArray;
        bool crd = mode == "CRD";
        if (block == 2 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
        {
            // Each output row interleaves two source rows (bx = 0, 1).
            for (int b = 0; b < n; b++)
            {
                for (int ch = 0; ch < oc; ch++)
                {
                    for (int by = 0; by < 2; by++)
                    {
                        int c0 = crd ? (ch * 4) + (by * 2) : (by * 2 * oc) + ch;
                        int c1 = crd ? c0 + 1 : c0 + oc;
                        for (int y = 0; y < h; y++)
                        {
                            Interleave(
                                s.AsSpan((((b * c) + c0) * h * w) + (y * w), w),
                                s.AsSpan((((b * c) + c1) * h * w) + (y * w), w),
                                d.AsSpan(((((b * oc) + ch) * h * 2) + (y * 2) + by) * w * 2, w * 2));
                        }
                    }
                }
            }

            return result;
        }

        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < oc; ch++)
            {
                for (int by = 0; by < block; by++)
                {
                    for (int bx = 0; bx < block; bx++)
                    {
                        int srcC = crd ? (ch * block * block) + (by * block) + bx : (((by * block) + bx) * oc) + ch;
                        int srcBase = ((b * c) + srcC) * h * w;
                        for (int y = 0; y < h; y++)
                        {
                            int dstRow = ((((b * oc) + ch) * h * block) + (y * block) + by) * w * block;
                            for (int xx = 0; xx < w; xx++)
                            {
                                d[dstRow + (xx * block) + bx] = s[srcBase + (y * w) + xx];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }
}
