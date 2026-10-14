using System.Numerics.Tensors;
using NanoOcr.Core;

namespace NanoOcr.Onnx;

internal static partial class Operators
{
    internal enum BinaryOp { Add, Sub, Mul, Div, Pow, Mod, FMod }

    internal enum CompareOp { Less, Greater, Equal }

    internal enum LogicOp { And, Or, Xor }

    internal enum UnaryOp { Abs, Neg, Sqrt, Exp, Sin, Cos, Erf, Reciprocal, Relu, Sigmoid, HardSwish, Softplus, Tanh }

    internal static Tensor Binary(Tensor a, Tensor b, BinaryOp op)
    {
        if (a.Type != b.Type)
        {
            throw new InvalidOperationException($"{op} of {a.Type} and {b.Type}.");
        }

        int[] shape = Broadcast.Shape(a.Shape, b.Shape);
        if (a.Type == ElementType.Float)
        {
            Tensor result = Tensor.Float(shape);
            float[] x = a.FloatArray, y = b.FloatArray, z = result.FloatArray;
            var plan = new Broadcast.RunPlan(shape, Broadcast.Strides(a.Shape, shape), Broadcast.Strides(b.Shape, shape));
            if (plan.Count == 1 && result.Length >= 1 << 16)
            {
                // One long run: split it so every core takes a share.
                plan.Offsets(0, out int ao, out int bo);
                int sa = plan.StepA, sb = plan.StepB;
                Core.Parallelism.For(plan.Inner, plan.Inner, (s0, s1) =>
                    FloatRun(op, x, ao + (s0 * sa), sa, y, bo + (s0 * sb), sb, z, s0, s1 - s0));
                return result;
            }

            if (Core.Parallelism.Workers <= 1 || plan.Count == 1)
            {
                FloatRuns(op, plan, x, y, z, 0, plan.Count);
            }
            else
            {
                Core.Parallelism.For(plan.Count, result.Length, (s0, s1) => FloatRuns(op, plan, x, y, z, s0, s1));
            }

            return result;
        }

        if (a.Type == ElementType.Int64)
        {
            Tensor result = Tensor.Int64(shape);
            long[] x = a.LongArray, y = b.LongArray, z = result.LongArray;
            int[] oa = Broadcast.Offsets(a.Shape, shape), ob = Broadcast.Offsets(b.Shape, shape);
            for (int i = 0; i < z.Length && i < oa.Length; i++)
            {
                long p = x[oa[i]], q = y[ob[i]];
                z[i] = op switch
                {
                    BinaryOp.Add => p + q,
                    BinaryOp.Sub => p - q,
                    BinaryOp.Mul => p * q,
                    BinaryOp.Div => p / q,
                    BinaryOp.Pow => (long)Math.Pow(p, q),
                    BinaryOp.FMod => p % q,
                    _ => IntFloorMod(p, q),
                };
            }

            return result;
        }

        throw new NotSupportedException($"{op} on {a.Type}.");
    }

    private static void FloatRuns(BinaryOp op, Broadcast.RunPlan plan, float[] x, float[] y, float[] z, int r0, int r1)
    {
        for (int r = r0; r < r1; r++)
        {
            plan.Offsets(r, out int ao, out int bo);
            FloatRun(op, x, ao, plan.StepA, y, bo, plan.StepB, z, r * plan.Inner, plan.Inner);
        }
    }

    private static void FloatRun(BinaryOp op, float[] x, int xo, int xs, float[] y, int yo, int ys, float[] z, int zo, int n)
    {
        Span<float> dst = z.AsSpan(zo, n);
        if (xs == 1 && ys == 1)
        {
            ReadOnlySpan<float> p = x.AsSpan(xo, n), q = y.AsSpan(yo, n);
            switch (op)
            {
                case BinaryOp.Add: TensorPrimitives.Add(p, q, dst); return;
                case BinaryOp.Sub: TensorPrimitives.Subtract(p, q, dst); return;
                case BinaryOp.Mul: TensorPrimitives.Multiply(p, q, dst); return;
                case BinaryOp.Div: TensorPrimitives.Divide(p, q, dst); return;
            }
        }
        else if (xs == 1 && ys == 0)
        {
            ReadOnlySpan<float> p = x.AsSpan(xo, n);
            float q = y[yo];
            switch (op)
            {
                case BinaryOp.Add: TensorPrimitives.Add(p, q, dst); return;
                case BinaryOp.Sub: TensorPrimitives.Subtract(p, q, dst); return;
                case BinaryOp.Mul: TensorPrimitives.Multiply(p, q, dst); return;
                case BinaryOp.Div: TensorPrimitives.Divide(p, q, dst); return;
                case BinaryOp.Pow when q == 2f: TensorPrimitives.Multiply(p, p, dst); return;
                case BinaryOp.Pow when q == 0.5f: TensorPrimitives.Sqrt(p, dst); return;
            }
        }
        else if (xs == 0 && ys == 1)
        {
            float p = x[xo];
            ReadOnlySpan<float> q = y.AsSpan(yo, n);
            switch (op)
            {
                case BinaryOp.Add: TensorPrimitives.Add(q, p, dst); return;
                case BinaryOp.Sub: TensorPrimitives.Subtract(p, q, dst); return;
                case BinaryOp.Mul: TensorPrimitives.Multiply(q, p, dst); return;
                case BinaryOp.Div: TensorPrimitives.Divide(p, q, dst); return;
            }
        }

        for (int i = 0; i < n; i++)
        {
            float p = x[xo + (i * xs)], q = y[yo + (i * ys)];
            dst[i] = op switch
            {
                BinaryOp.Add => p + q,
                BinaryOp.Sub => p - q,
                BinaryOp.Mul => p * q,
                BinaryOp.Div => p / q,
                BinaryOp.Pow => Pow(p, q),
                BinaryOp.FMod => p % q,
                _ => FloorMod(p, q),
            };
        }
    }

    private static float Pow(float p, float q) => q switch
    {
        2f => p * p,
        3f => p * p * p,
        0.5f => MathF.Sqrt(p),
        _ => MathF.Pow(p, q),
    };

    private static long IntFloorMod(long p, long q)
    {
        long r = p % q;
        return r != 0 && ((r < 0) != (q < 0)) ? r + q : r;
    }

    private static float FloorMod(float p, float q)
    {
        float r = p % q;
        return r != 0 && ((r < 0) != (q < 0)) ? r + q : r;
    }

    internal static Tensor Compare(Tensor a, Tensor b, CompareOp op)
    {
        int[] shape = Broadcast.Shape(a.Shape, b.Shape);
        Tensor result = Tensor.Bool(shape);
        bool[] z = result.BoolArray;
        int[] oa = Broadcast.Offsets(a.Shape, shape), ob = Broadcast.Offsets(b.Shape, shape);
        int n = result.Length;
        if (a.Type == ElementType.Float)
        {
            float[] x = a.FloatArray, y = b.FloatArray;
            for (int i = 0; i < n; i++)
            {
                float p = x[oa[i]], q = y[ob[i]];
                z[i] = op switch { CompareOp.Less => p < q, CompareOp.Greater => p > q, _ => p == q };
            }
        }
        else if (a.Type == ElementType.Int64)
        {
            long[] x = a.LongArray, y = b.LongArray;
            for (int i = 0; i < n; i++)
            {
                long p = x[oa[i]], q = y[ob[i]];
                z[i] = op switch { CompareOp.Less => p < q, CompareOp.Greater => p > q, _ => p == q };
            }
        }
        else
        {
            bool[] x = a.BoolArray, y = b.BoolArray;
            for (int i = 0; i < n; i++)
            {
                bool p = x[oa[i]], q = y[ob[i]];
                z[i] = op switch { CompareOp.Equal => p == q, CompareOp.Less => !p && q, _ => p && !q };
            }
        }

        return result;
    }

    internal static Tensor Logical(Tensor a, Tensor b, LogicOp op)
    {
        int[] shape = Broadcast.Shape(a.Shape, b.Shape);
        Tensor result = Tensor.Bool(shape);
        bool[] x = a.BoolArray, y = b.BoolArray, z = result.BoolArray;
        int[] oa = Broadcast.Offsets(a.Shape, shape), ob = Broadcast.Offsets(b.Shape, shape);
        for (int i = 0; i < result.Length; i++)
        {
            bool p = x[oa[i]], q = y[ob[i]];
            z[i] = op switch { LogicOp.And => p && q, LogicOp.Or => p || q, _ => p ^ q };
        }

        return result;
    }

    internal static Tensor Not(Tensor a)
    {
        Tensor result = Tensor.Bool((int[])a.Shape.Clone());
        Span<bool> x = a.Bools, z = result.Bools;
        for (int i = 0; i < x.Length; i++)
        {
            z[i] = !x[i];
        }

        return result;
    }

    internal static Tensor Where(Tensor cond, Tensor a, Tensor b)
    {
        int[] shape = Broadcast.Shape(cond.Shape, a.Shape, b.Shape);
        Tensor result = Tensor.Create(a.Type, shape);
        int[] oc = Broadcast.Offsets(cond.Shape, shape);
        int[] oa = Broadcast.Offsets(a.Shape, shape);
        int[] ob = Broadcast.Offsets(b.Shape, shape);
        bool[] c = cond.BoolArray;
        int n = result.Length;
        switch (a.Type)
        {
            case ElementType.Float:
                {
                    float[] x = a.FloatArray, y = b.FloatArray, z = result.FloatArray;
                    for (int i = 0; i < n; i++)
                    {
                        z[i] = c[oc[i]] ? x[oa[i]] : y[ob[i]];
                    }

                    break;
                }

            case ElementType.Int64:
                {
                    long[] x = a.LongArray, y = b.LongArray, z = result.LongArray;
                    for (int i = 0; i < n; i++)
                    {
                        z[i] = c[oc[i]] ? x[oa[i]] : y[ob[i]];
                    }

                    break;
                }

            default:
                {
                    bool[] x = a.BoolArray, y = b.BoolArray, z = result.BoolArray;
                    for (int i = 0; i < n; i++)
                    {
                        z[i] = c[oc[i]] ? x[oa[i]] : y[ob[i]];
                    }

                    break;
                }
        }

        return result;
    }

    internal static Tensor Unary(Tensor a, UnaryOp op)
    {
        if (a.Type == ElementType.Int64)
        {
            Tensor r = Tensor.Int64((int[])a.Shape.Clone());
            Span<long> xs = a.Longs, zs = r.Longs;
            for (int i = 0; i < xs.Length; i++)
            {
                zs[i] = op switch
                {
                    UnaryOp.Abs => Math.Abs(xs[i]),
                    UnaryOp.Neg => -xs[i],
                    UnaryOp.Relu => Math.Max(0, xs[i]),
                    _ => throw new NotSupportedException($"{op} on integers."),
                };
            }

            return r;
        }

        Tensor result = Tensor.Float((int[])a.Shape.Clone());
        float[] x = a.FloatArray, z = result.FloatArray;
        int n = a.Length;
        Parallelism.For(n, (long)n * 8, (s, e) => UnaryRange(op, x.AsSpan(s, e - s), z.AsSpan(s, e - s)));
        return result;
    }

    private static void UnaryRange(UnaryOp op, ReadOnlySpan<float> x, Span<float> z)
    {
        switch (op)
        {
            case UnaryOp.Abs: TensorPrimitives.Abs(x, z); break;
            case UnaryOp.Neg: TensorPrimitives.Negate(x, z); break;
            case UnaryOp.Sqrt: TensorPrimitives.Sqrt(x, z); break;
            case UnaryOp.Exp: TensorPrimitives.Exp(x, z); break;
            case UnaryOp.Sin: TensorPrimitives.Sin(x, z); break;
            case UnaryOp.Cos: TensorPrimitives.Cos(x, z); break;
            case UnaryOp.Tanh: TensorPrimitives.Tanh(x, z); break;
            case UnaryOp.Reciprocal: TensorPrimitives.Reciprocal(x, z); break;
            case UnaryOp.Relu: TensorPrimitives.Max(x, 0f, z); break;
            case UnaryOp.Sigmoid: TensorPrimitives.Sigmoid(x, z); break;
            case UnaryOp.HardSwish: Activations.HardSwish(x, z); break;
            case UnaryOp.Softplus:
                Activations.Softplus(x, z);
                break;
            case UnaryOp.Erf:
                Activations.ErfFast(x, z);
                break;
        }
    }

    internal static Tensor Clip(OnnxNode node, List<Tensor?> x)
    {
        Tensor a = x[0]!;
        Tensor? lo = x.Count > 1 ? x[1] : null;
        Tensor? hi = x.Count > 2 ? x[2] : null;
        if (a.Type == ElementType.Int64)
        {
            long min = lo?.Longs[0] ?? long.MinValue, max = hi?.Longs[0] ?? long.MaxValue;
            Tensor r = Tensor.Int64((int[])a.Shape.Clone());
            Span<long> xs = a.Longs, zs = r.Longs;
            for (int i = 0; i < xs.Length; i++)
            {
                zs[i] = Math.Min(max, Math.Max(min, xs[i]));
            }

            return r;
        }

        float fmin = lo?.Floats[0] ?? node.Float("min", float.MinValue);
        float fmax = hi?.Floats[0] ?? node.Float("max", float.MaxValue);
        Tensor result = Tensor.Float((int[])a.Shape.Clone());
        TensorPrimitives.Clamp(a.Floats, fmin, fmax, result.Floats);
        return result;
    }

    internal static Tensor Cast(Tensor a, int to)
    {
        ElementType target = OnnxModel.ElementTypeOf(to);
        int[] shape = (int[])a.Shape.Clone();
        if (target == a.Type && target != ElementType.Int64)
        {
            return a.Reshape(shape);
        }

        Tensor r = Tensor.Create(target, shape);
        int n = a.Length;
        switch (target, a.Type)
        {
            case (ElementType.Float, ElementType.Int64):
                for (int i = 0; i < n; i++) { r.FloatArray[i] = a.LongArray[i]; }
                break;
            case (ElementType.Float, ElementType.Bool):
                for (int i = 0; i < n; i++) { r.FloatArray[i] = a.BoolArray[i] ? 1f : 0f; }
                break;
            case (ElementType.Int64, ElementType.Float):
                for (int i = 0; i < n; i++) { r.LongArray[i] = FloatToInt(a.FloatArray[i], to); }
                break;
            case (ElementType.Int64, ElementType.Bool):
                for (int i = 0; i < n; i++) { r.LongArray[i] = a.BoolArray[i] ? 1 : 0; }
                break;
            case (ElementType.Int64, ElementType.Int64):
                for (int i = 0; i < n; i++) { r.LongArray[i] = WrapInt(a.LongArray[i], to); }
                break;
            case (ElementType.Bool, ElementType.Float):
                for (int i = 0; i < n; i++) { r.BoolArray[i] = a.FloatArray[i] != 0; }
                break;
            case (ElementType.Bool, ElementType.Int64):
                for (int i = 0; i < n; i++) { r.BoolArray[i] = a.LongArray[i] != 0; }
                break;
        }

        return r;
    }

    private static long FloatToInt(float v, int to)
    {
        if (float.IsNaN(v))
        {
            return to == OnnxModel.DataTypes.Int64 ? long.MinValue : 0;
        }

        return WrapInt((long)v, to);
    }

    private static long WrapInt(long v, int to) => to switch
    {
        OnnxModel.DataTypes.Int32 => (int)v,
        OnnxModel.DataTypes.Uint8 => (byte)v,
        OnnxModel.DataTypes.Int8 => (sbyte)v,
        OnnxModel.DataTypes.Int16 => (short)v,
        OnnxModel.DataTypes.Uint16 => (ushort)v,
        OnnxModel.DataTypes.Uint32 => (uint)v,
        _ => v,
    };
}
