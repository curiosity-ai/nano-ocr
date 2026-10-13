using IndraOcr.Vlm.Core;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>
/// A float32 activation map in channels-last order (<c>height × width × channels</c>), backed by
/// a pooled buffer.
/// </summary>
/// <remarks>
/// PyTorch holds the detector's activations as NCHW. Channels-last makes every 1×1 convolution a
/// plain <c>[pixels, in] · [in, out]</c> product, lets the depthwise convolutions vectorise across
/// channels, and lays LiteMLA's heads out contiguously; the parity tests transpose the reference.
/// </remarks>
public sealed class FeatureMap : IDisposable
{
    private float[]? _data;

    private FeatureMap(float[] data, int height, int width, int channels)
    {
        _data = data;
        Height = height;
        Width = width;
        Channels = channels;
    }

    /// <summary>Rows.</summary>
    public int Height { get; }

    /// <summary>Columns.</summary>
    public int Width { get; }

    /// <summary>Channels per pixel.</summary>
    public int Channels { get; }

    /// <summary>Pixels (<c>height × width</c>).</summary>
    public int Pixels => Height * Width;

    /// <summary>Elements.</summary>
    public int Length => Height * Width * Channels;

    /// <summary>The backing array (longer than <see cref="Length"/>).</summary>
    public float[] Array => _data ?? throw new ObjectDisposedException(nameof(FeatureMap));

    /// <summary>The values, <c>[height, width, channels]</c>.</summary>
    public Span<float> Span => Array.AsSpan(0, Length);

    /// <summary>Rents an uninitialised map.</summary>
    public static FeatureMap Rent(int height, int width, int channels) =>
        new(TensorPool.RentArray(height * width * channels), height, width, channels);

    /// <summary>The map as NCHW (<c>[channels, height, width]</c>), the layout PyTorch reports.</summary>
    public float[] ToChannelsFirst()
    {
        var result = new float[Length];
        int pixels = Pixels;
        int channels = Channels;
        float[] data = Array;
        Parallel.For(0, channels, Parallelism.Options, c =>
        {
            int dst = c * pixels;
            for (int p = 0, src = c; p < pixels; p++, src += channels)
            {
                result[dst + p] = data[src];
            }
        });
        return result;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_data is not null)
        {
            TensorPool.Return(_data);
            _data = null;
        }
    }
}
