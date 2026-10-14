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

    /// <summary>An image over a buffer rented from <see cref="System.Buffers.ArrayPool{T}.Shared"/>; contents undefined.</summary>
    internal static GrayImage Rent(int width, int height) =>
        new(width, height, System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Max(1, checked(width * height))));

    /// <summary>Gives a <see cref="Rent"/>ed image's buffer back; the image must not be used afterwards.</summary>
    internal void Return() => System.Buffers.ArrayPool<byte>.Shared.Return(Pixels);

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
        if (k == 2)
        {
            // Half a turn: the pixels in reverse order.
            s.AsSpan(0, w * h).CopyTo(d);
            d.AsSpan(0, w * h).Reverse();
            return r;
        }

        // A quarter turn is a transpose with one axis flipped; 64×64 blocks keep both the rows
        // read and the columns written in cache, and each output row of a block is written
        // contiguously from a column of it.
        QuarterTurn(s, d, w, h, k == 1);
        return r;
    }

    private static unsafe void QuarterTurn(byte[] src, byte[] dst, int w, int h, bool clockwise)
    {
        const int B = 64;
        fixed (byte* s = src)
        fixed (byte* d = dst)
        {
            for (int y0 = 0; y0 < h; y0 += B)
            {
                int y1 = Math.Min(h, y0 + B);
                for (int x0 = 0; x0 < w; x0 += B)
                {
                    int x1 = Math.Min(w, x0 + B);
                    for (int x = x0; x < x1; x++)
                    {
                        byte* from = s + (y0 * (long)w) + x;
                        if (clockwise)
                        {
                            // (x, y) -> (h-1-y, x): row x, written right to left.
                            byte* to = d + (x * (long)h) + h - 1 - y0;
                            for (int y = y0; y < y1; y++, from += w)
                            {
                                *to-- = *from;
                            }
                        }
                        else
                        {
                            // (x, y) -> (y, w-1-x): row w-1-x, left to right.
                            byte* to = d + ((w - 1 - x) * (long)h) + y0;
                            for (int y = y0; y < y1; y++, from += w)
                            {
                                *to++ = *from;
                            }
                        }
                    }
                }
            }
        }
    }
}
