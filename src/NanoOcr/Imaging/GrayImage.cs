namespace NanoOcr.Imaging;

/// <summary>An 8-bit grayscale image, row-major with no row padding (the pipeline's page type).</summary>
public sealed class GrayImage
{
    /// <summary>Creates a zero-filled image.</summary>
    public GrayImage(int width, int height)
        : this(width, height, new byte[checked(width * height)])
    {
    }

    /// <summary>Wraps existing pixels (not copied).</summary>
    public GrayImage(int width, int height, byte[] pixels)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(height);
        if (pixels.Length < width * height)
        {
            throw new ArgumentException("The pixel buffer is smaller than the image.", nameof(pixels));
        }

        Width = width;
        Height = height;
        Pixels = pixels;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>The pixels, <c>Height × Width</c>, row-major.</summary>
    public byte[] Pixels { get; }

    /// <summary>One row of pixels.</summary>
    public Span<byte> Row(int y) => Pixels.AsSpan(y * Width, Width);

    /// <summary>The pixel at (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public byte this[int x, int y]
    {
        get => Pixels[(y * Width) + x];
        set => Pixels[(y * Width) + x] = value;
    }

    /// <summary>A deep copy.</summary>
    public GrayImage Clone() => new(Width, Height, (byte[])Pixels.Clone());

    /// <summary>
    /// The image turned by <paramref name="quarterTurns"/> quarter turns clockwise — numpy's
    /// <c>np.rot90(img, -k)</c>, which the pipeline uses to undo a counter-clockwise turn.
    /// </summary>
    public GrayImage RotateClockwise(int quarterTurns)
    {
        int k = ((quarterTurns % 4) + 4) % 4;
        if (k == 0)
        {
            return Clone();
        }

        int w = Width, h = Height;
        GrayImage r = k == 2 ? new GrayImage(w, h) : new GrayImage(h, w);
        byte[] s = Pixels, d = r.Pixels;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                byte v = s[(y * w) + x];
                switch (k)
                {
                    case 1: d[(x * h) + (h - 1 - y)] = v; break; // clockwise: (x, y) -> (h-1-y, x)
                    case 2: d[((h - 1 - y) * w) + (w - 1 - x)] = v; break;
                    default: d[((w - 1 - x) * h) + y] = v; break; // counter-clockwise
                }
            }
        }

        return r;
    }
}
