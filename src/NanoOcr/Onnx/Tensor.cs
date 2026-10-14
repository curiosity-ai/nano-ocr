using System.Buffers;
using System.Runtime.CompilerServices;

namespace NanoOcr.Onnx;

/// <summary>Element types the interpreter stores. Every integer type is held as <see cref="long"/>.</summary>
public enum ElementType
{
    /// <summary>32-bit IEEE float (ONNX FLOAT, and FLOAT16/DOUBLE/BFLOAT16 widened or narrowed to it).</summary>
    Float,

    /// <summary>Any ONNX integer type, held as 64-bit integers.</summary>
    Int64,

    /// <summary>ONNX BOOL.</summary>
    Bool,
}

/// <summary>
/// The backing array of one or more tensors. Reshapes and squeezes share a storage instead of
/// copying, so the array is returned to the pool only when the last tensor over it dies.
/// </summary>
internal sealed class Storage
{
    public float[]? Floats;
    public long[]? Longs;
    public bool[]? Bools;
    public int References;

    /// <summary>Weights and constants: never pooled.</summary>
    public bool Pinned;

    public void Release()
    {
        if (Pinned || --References > 0)
        {
            return;
        }

        if (Floats is not null)
        {
            ArrayPool<float>.Shared.Return(Floats);
        }
        else if (Longs is not null)
        {
            ArrayPool<long>.Shared.Return(Longs);
        }
        else if (Bools is not null)
        {
            ArrayPool<bool>.Shared.Return(Bools);
        }

        Floats = null;
        Longs = null;
        Bools = null;
    }
}

/// <summary>
/// A dense, row-major tensor. Arrays may be longer than <see cref="Length"/> because they come
/// from <see cref="ArrayPool{T}.Shared"/>; only the first <see cref="Length"/> elements are data.
/// </summary>
public sealed class Tensor
{
    internal readonly Storage Store;

    /// <summary>The dimensions, outermost first.</summary>
    public int[] Shape { get; }

    /// <summary>The element type.</summary>
    public ElementType Type { get; }

    /// <summary>Number of elements (the product of <see cref="Shape"/>).</summary>
    public int Length { get; }

    internal Tensor(int[] shape, ElementType type, Storage store)
    {
        Shape = shape;
        Type = type;
        Store = store;
        Length = SizeOf(shape);
    }

    /// <summary>Rank (number of dimensions).</summary>
    public int Rank => Shape.Length;

    /// <summary>The float data (only for <see cref="ElementType.Float"/>).</summary>
    public Span<float> Floats => Store.Floats.AsSpan(0, Length);

    /// <summary>The integer data (only for <see cref="ElementType.Int64"/>).</summary>
    public Span<long> Longs => Store.Longs.AsSpan(0, Length);

    /// <summary>The boolean data (only for <see cref="ElementType.Bool"/>).</summary>
    public Span<bool> Bools => Store.Bools.AsSpan(0, Length);

    internal float[] FloatArray => Store.Floats!;

    internal long[] LongArray => Store.Longs!;

    internal bool[] BoolArray => Store.Bools!;

    /// <summary>Product of the dimensions.</summary>
    public static int SizeOf(ReadOnlySpan<int> shape)
    {
        long n = 1;
        foreach (int d in shape)
        {
            n *= d;
        }

        return checked((int)n);
    }

    /// <summary>A new pooled float tensor; its contents are undefined unless <paramref name="clear"/>.</summary>
    public static Tensor Float(int[] shape, bool clear = false)
    {
        int n = SizeOf(shape);
        float[] array = ArrayPool<float>.Shared.Rent(Math.Max(n, 1));
        if (clear)
        {
            Array.Clear(array, 0, n);
        }

        return new Tensor(shape, ElementType.Float, new Storage { Floats = array, References = 1 });
    }

    /// <summary>A new pooled integer tensor; its contents are undefined unless <paramref name="clear"/>.</summary>
    public static Tensor Int64(int[] shape, bool clear = false)
    {
        int n = SizeOf(shape);
        long[] array = ArrayPool<long>.Shared.Rent(Math.Max(n, 1));
        if (clear)
        {
            Array.Clear(array, 0, n);
        }

        return new Tensor(shape, ElementType.Int64, new Storage { Longs = array, References = 1 });
    }

    /// <summary>A new pooled boolean tensor; its contents are undefined unless <paramref name="clear"/>.</summary>
    public static Tensor Bool(int[] shape, bool clear = false)
    {
        int n = SizeOf(shape);
        bool[] array = ArrayPool<bool>.Shared.Rent(Math.Max(n, 1));
        if (clear)
        {
            Array.Clear(array, 0, n);
        }

        return new Tensor(shape, ElementType.Bool, new Storage { Bools = array, References = 1 });
    }

    /// <summary>A new tensor of <paramref name="type"/>.</summary>
    public static Tensor Create(ElementType type, int[] shape, bool clear = false) => type switch
    {
        ElementType.Float => Float(shape, clear),
        ElementType.Int64 => Int64(shape, clear),
        _ => Bool(shape, clear),
    };

    /// <summary>Wraps an existing float array (not pooled).</summary>
    public static Tensor FromFloats(float[] data, params int[] shape)
    {
        if (data.Length < SizeOf(shape))
        {
            throw new ArgumentException("The array is smaller than the shape.", nameof(data));
        }

        return new Tensor(shape, ElementType.Float, new Storage { Floats = data, References = 1, Pinned = true });
    }

    /// <summary>Wraps an existing integer array (not pooled).</summary>
    public static Tensor FromLongs(long[] data, params int[] shape)
    {
        if (data.Length < SizeOf(shape))
        {
            throw new ArgumentException("The array is smaller than the shape.", nameof(data));
        }

        return new Tensor(shape, ElementType.Int64, new Storage { Longs = data, References = 1, Pinned = true });
    }

    /// <summary>Wraps an existing boolean array (not pooled).</summary>
    public static Tensor FromBools(bool[] data, params int[] shape)
    {
        if (data.Length < SizeOf(shape))
        {
            throw new ArgumentException("The array is smaller than the shape.", nameof(data));
        }

        return new Tensor(shape, ElementType.Bool, new Storage { Bools = data, References = 1, Pinned = true });
    }

    /// <summary>A scalar integer tensor.</summary>
    public static Tensor Scalar(long value) => FromLongs([value]);

    /// <summary>A view of the same data with another shape (no copy).</summary>
    public Tensor Reshape(int[] shape)
    {
        if (SizeOf(shape) != Length)
        {
            throw new InvalidOperationException(
                $"Cannot reshape [{string.Join(",", Shape)}] to [{string.Join(",", shape)}].");
        }

        Store.References++;
        return new Tensor(shape, Type, Store);
    }

    /// <summary>Marks this tensor as never returned to the pool (weights, constants).</summary>
    internal Tensor Pin()
    {
        Store.Pinned = true;
        return this;
    }

    /// <summary>Gives the backing array back to the pool once nothing else references it.</summary>
    public void Release() => Store.Release();

    /// <summary>A copy of the data as a new float array (exactly <see cref="Length"/> long).</summary>
    public float[] ToFloatArray() => Type switch
    {
        ElementType.Float => Floats.ToArray(),
        ElementType.Int64 => Longs.ToArray().Select(v => (float)v).ToArray(),
        _ => Bools.ToArray().Select(v => v ? 1f : 0f).ToArray(),
    };

    /// <summary>A copy of the data as a new integer array.</summary>
    public long[] ToLongArray() => Type switch
    {
        ElementType.Int64 => Longs.ToArray(),
        ElementType.Float => Floats.ToArray().Select(v => (long)v).ToArray(),
        _ => Bools.ToArray().Select(v => v ? 1L : 0L).ToArray(),
    };

    /// <summary>Strides (in elements) of a row-major layout of <paramref name="shape"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int[] Strides(ReadOnlySpan<int> shape)
    {
        var strides = new int[shape.Length];
        int s = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = s;
            s *= shape[i];
        }

        return strides;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Type}[{string.Join(",", Shape)}]";
}
