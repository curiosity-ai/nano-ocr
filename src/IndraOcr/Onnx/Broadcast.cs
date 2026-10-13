namespace IndraOcr.Onnx;

/// <summary>Numpy-style broadcasting: shapes, strides and a walk over coalesced runs.</summary>
internal static class Broadcast
{
    /// <summary>The broadcast shape of <paramref name="shapes"/>.</summary>
    public static int[] Shape(params int[][] shapes)
    {
        int rank = shapes.Max(s => s.Length);
        var result = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            int size = 1;
            foreach (int[] s in shapes)
            {
                int i = d - (rank - s.Length);
                if (i < 0)
                {
                    continue;
                }

                int v = s[i];
                if (v == 1)
                {
                    continue;
                }

                if (size != 1 && size != v)
                {
                    throw new InvalidOperationException(
                        $"Shapes {string.Join(" and ", shapes.Select(x => "[" + string.Join(",", x) + "]"))} do not broadcast.");
                }

                size = v;
            }

            // A zero-sized dimension broadcasts as zero.
            if (shapes.Any(s => d - (rank - s.Length) >= 0 && s[d - (rank - s.Length)] == 0))
            {
                size = 0;
            }

            result[d] = size;
        }

        return result;
    }

    /// <summary>Strides of <paramref name="shape"/> laid over <paramref name="outShape"/>; 0 where broadcast.</summary>
    public static int[] Strides(int[] shape, int[] outShape)
    {
        int rank = outShape.Length;
        var strides = new int[rank];
        int s = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            int i = d - (rank - shape.Length);
            int size = i >= 0 ? shape[i] : 1;
            strides[d] = size == 1 && outShape[d] != 1 ? 0 : s;
            if (size == 1)
            {
                strides[d] = 0;
            }

            s *= size;
        }

        return strides;
    }

    /// <summary>Per-element offsets of an input laid over the output (for the generic slow paths).</summary>
    public static int[] Offsets(int[] shape, int[] outShape)
    {
        int[] strides = Strides(shape, outShape);
        int n = Tensor.SizeOf(outShape);
        var offsets = new int[n];
        int rank = outShape.Length;
        if (rank == 0)
        {
            return offsets;
        }

        var counter = new int[rank];
        int offset = 0;
        for (int i = 0; i < n; i++)
        {
            offsets[i] = offset;
            for (int d = rank - 1; d >= 0; d--)
            {
                counter[d]++;
                offset += strides[d];
                if (counter[d] < outShape[d])
                {
                    break;
                }

                offset -= strides[d] * counter[d];
                counter[d] = 0;
            }
        }

        return offsets;
    }

    /// <summary>
    /// Walks two inputs over the output as coalesced runs. For each run the callback receives the
    /// output offset, each input's offset and step (0 or 1), and the run length.
    /// </summary>
    public static void Runs(int[] outShape, int[] stridesA, int[] stridesB, RunAction action)
    {
        int rank = outShape.Length;
        if (rank == 0)
        {
            action(0, 0, 0, 0, 0, 1);
            return;
        }

        // Coalesce dimensions from the right while both inputs keep the same kind of stride.
        var dims = new List<int>();
        var sa = new List<int>();
        var sb = new List<int>();
        for (int d = rank - 1; d >= 0; d--)
        {
            if (outShape[d] == 1)
            {
                continue;
            }

            if (dims.Count > 0)
            {
                int last = dims.Count - 1;
                bool aOk = (stridesA[d] == 0 && sa[last] == 0) || (sa[last] != 0 && stridesA[d] == sa[last] * dims[last]);
                bool bOk = (stridesB[d] == 0 && sb[last] == 0) || (sb[last] != 0 && stridesB[d] == sb[last] * dims[last]);
                if (aOk && bOk)
                {
                    dims[last] *= outShape[d];
                    continue;
                }
            }

            dims.Add(outShape[d]);
            sa.Add(stridesA[d]);
            sb.Add(stridesB[d]);
        }

        if (dims.Count == 0)
        {
            action(0, 0, 0, 0, 0, 1);
            return;
        }

        // dims[0] is innermost.
        int inner = dims[0];
        int stepA = sa[0] == 0 ? 0 : 1;
        int stepB = sb[0] == 0 ? 0 : 1;
        int outer = 1;
        for (int i = 1; i < dims.Count; i++)
        {
            outer *= dims[i];
        }

        var counter = new int[dims.Count];
        int offA = 0, offB = 0;
        for (int o = 0; o < outer; o++)
        {
            action(o * inner, offA, stepA, offB, stepB, inner);
            for (int i = 1; i < dims.Count; i++)
            {
                counter[i]++;
                offA += sa[i];
                offB += sb[i];
                if (counter[i] < dims[i])
                {
                    break;
                }

                offA -= sa[i] * counter[i];
                offB -= sb[i] * counter[i];
                counter[i] = 0;
            }
        }
    }

    /// <summary>One coalesced run of a two-input broadcast.</summary>
    public delegate void RunAction(int outOffset, int aOffset, int aStep, int bOffset, int bStep, int length);
}
