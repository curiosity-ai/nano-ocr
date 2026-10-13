using System.Buffers;

namespace IndraOcr.Vlm.Models;

/// <summary>Receives a named intermediate tensor, <c>[rows, width]</c> row-major.</summary>
public delegate void TraceSink(string name, ReadOnlySpan<float> values, int width);

/// <summary>
/// What one sequence carries between decoder steps: a key/value cache for each full-attention
/// layer, and for each gated-delta-net layer its convolution history and recurrent state
/// (<c>Qwen3_5DynamicCache</c>'s <c>key_cache</c>/<c>value_cache</c>, <c>conv_states</c> and
/// <c>recurrent_states</c>).
/// </summary>
public sealed class DecoderState : IDisposable
{
    private readonly Qwen35Config _config;
    private readonly int _keyValueWidth;
    private int _capacity;

    internal DecoderState(Qwen35Config config, int capacity)
    {
        _config = config;
        _keyValueWidth = config.NumKeyValueHeads * config.HeadDim;
        _capacity = Math.Max(16, capacity);
        int layers = config.NumLayers;
        Keys = new float[layers][];
        Values = new float[layers][];
        ConvHistory = new float[layers][];
        Recurrent = new float[layers][];
        int convDim = (2 * config.LinearKeyDim) + config.LinearValueDim;
        for (int i = 0; i < layers; i++)
        {
            if (config.LayerTypes[i] == "full_attention")
            {
                Keys[i] = ArrayPool<float>.Shared.Rent(_capacity * _keyValueWidth);
                Values[i] = ArrayPool<float>.Shared.Rent(_capacity * _keyValueWidth);
            }
            else
            {
                Keys[i] = Values[i] = [];
                ConvHistory[i] = new float[(config.LinearConvKernel - 1) * convDim];
                Recurrent[i] = new float[config.LinearValueHeads * config.LinearKeyHeadDim * config.LinearValueHeadDim];
            }
        }
    }

    /// <summary>Tokens the state has consumed.</summary>
    public int Length { get; internal set; }

    /// <summary>The M-RoPE offset decode steps add to their cache position (<c>rope_deltas</c>).</summary>
    public int RopeDelta { get; internal set; }

    internal float[][] Keys { get; private set; }

    internal float[][] Values { get; private set; }

    internal float[][] ConvHistory { get; }

    internal float[][] Recurrent { get; }

    internal int KeyValueWidth => _keyValueWidth;

    /// <summary>Grows the caches to hold <paramref name="extra"/> more positions.</summary>
    internal void Reserve(int extra)
    {
        int required = Length + extra;
        if (required <= _capacity)
        {
            return;
        }

        int capacity = _capacity;
        while (capacity < required)
        {
            capacity *= 2;
        }

        for (int i = 0; i < Keys.Length; i++)
        {
            if (Keys[i].Length == 0)
            {
                continue;
            }

            float[] keys = ArrayPool<float>.Shared.Rent(capacity * _keyValueWidth);
            float[] values = ArrayPool<float>.Shared.Rent(capacity * _keyValueWidth);
            Keys[i].AsSpan(0, Length * _keyValueWidth).CopyTo(keys);
            Values[i].AsSpan(0, Length * _keyValueWidth).CopyTo(values);
            ArrayPool<float>.Shared.Return(Keys[i]);
            ArrayPool<float>.Shared.Return(Values[i]);
            Keys[i] = keys;
            Values[i] = values;
        }

        _capacity = capacity;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < Keys.Length; i++)
        {
            if (Keys[i].Length > 0)
            {
                ArrayPool<float>.Shared.Return(Keys[i]);
                ArrayPool<float>.Shared.Return(Values[i]);
            }

            Keys[i] = Values[i] = [];
        }
    }
}
