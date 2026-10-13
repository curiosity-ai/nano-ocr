using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>
/// Pillow's <c>Image.thumbnail(size, LANCZOS)</c> (<c>PIL/Image.py</c>) with its default
/// <c>reducing_gap=2.0</c>: the aspect-preserving target size, then — when the image is at least
/// twice that size along an axis — <c>Image.reduce</c> by the integer factor
/// (<c>libImaging/Reduce.c</c>), then a Lanczos resize of the reduced image's fractional box.
/// </summary>
public static class PilThumbnail
{
    /// <summary>
    /// <c>preserve_aspect_ratio</c>: the largest size within <c>maxWidth × maxHeight</c> whose
    /// aspect ratio is nearest the image's, or <see langword="null"/> when the image already fits.
    /// </summary>
    public static (int Width, int Height)? TargetSize(int width, int height, int maxWidth, int maxHeight)
    {
        int x = maxWidth;
        int y = maxHeight;
        if (x >= width && y >= height)
        {
            return null;
        }

        double aspect = width / (double)height;
        if (x / (double)y >= aspect)
        {
            double number = y * aspect;
            x = RoundAspect(number, n => Math.Abs(aspect - (n / (double)y)));
        }
        else
        {
            double number = x / aspect;
            y = RoundAspect(number, n => n == 0 ? 0 : Math.Abs(aspect - (x / (double)n)));
        }

        return (x, y);
    }

    // max(min(floor(number), ceil(number), key=key), 1): min keeps the first of equal keys.
    private static int RoundAspect(double number, Func<int, double> key)
    {
        int floor = (int)Math.Floor(number);
        int ceil = (int)Math.Ceiling(number);
        int best = key(ceil) < key(floor) ? ceil : floor;
        return Math.Max(best, 1);
    }

    /// <summary>
    /// Returns the thumbnail of <paramref name="image"/> — a copy when it already fits.
    /// </summary>
    public static RgbImage Thumbnail(RgbImage image, int maxWidth, int maxHeight)
    {
        (int Width, int Height)? target = TargetSize(image.Width, image.Height, maxWidth, maxHeight);
        if (target is null || (target.Value.Width == image.Width && target.Value.Height == image.Height))
        {
            return image.Clone();
        }

        (int w, int h) = target.Value;
        const double ReducingGap = 2.0;
        int factorX = Math.Max(1, (int)(image.Width / (double)w / ReducingGap));
        int factorY = Math.Max(1, (int)(image.Height / (double)h / ReducingGap));
        if (factorX == 1 && factorY == 1)
        {
            return PilResize.ResizeLanczos(image, w, h);
        }

        // _get_safe_box widens the full-image box by the filter support and clamps it back to
        // the image, so the whole image is reduced; the box left for the resize is fractional.
        using RgbImage reduced = Reduce(image, factorX, factorY);
        var box = ((float)0, (float)0, (float)(image.Width / (double)factorX), (float)(image.Height / (double)factorY));
        return PilResize.Resize(reduced, w, h, PilResize.Filter.Lanczos, box);
    }

    /// <summary>
    /// <c>Image.reduce((factorX, factorY))</c> of the whole image: each output pixel is the
    /// rounded mean of its <c>factorX × factorY</c> block (the partial blocks of the last row and
    /// column included), computed as Pillow does —
    /// <c>((Σ + n/2) · ⌊2³² / (256·n)⌋) ≫ 24</c> with the multiplier formed in float.
    /// </summary>
    public static RgbImage Reduce(RgbImage image, int factorX, int factorY)
    {
        int outW = (image.Width + factorX - 1) / factorX;
        int outH = (image.Height + factorY - 1) / factorY;
        RgbImage result = RgbImage.Rent(outW, outH);
        int width = image.Width;
        int height = image.Height;
        Parallel.For(0, outH, IndraOcr.Vlm.Core.Parallelism.Options, y =>
        {
            int y0 = y * factorY;
            int y1 = Math.Min(height, y0 + factorY);
            Span<byte> dst = result.Row(y);
            for (int x = 0; x < outW; x++)
            {
                int x0 = x * factorX;
                int x1 = Math.Min(width, x0 + factorX);
                uint count = (uint)((x1 - x0) * (y1 - y0));
                uint multiplier = Multiplier(count);
                uint amend = count / 2;
                uint r = amend, g = amend, b = amend;
                for (int yy = y0; yy < y1; yy++)
                {
                    ReadOnlySpan<byte> row = image.Row(yy);
                    for (int xx = x0; xx < x1; xx++)
                    {
                        r += row[3 * xx];
                        g += row[(3 * xx) + 1];
                        b += row[(3 * xx) + 2];
                    }
                }

                dst[3 * x] = (byte)((r * multiplier) >> 24);
                dst[(3 * x) + 1] = (byte)((g * multiplier) >> 24);
                dst[(3 * x) + 2] = (byte)((b * multiplier) >> 24);
            }
        });

        return result;
    }

    /// <summary><c>division_UINT32(divider, 8)</c>: <c>(UINT32)((float)2³² / (float)(256 · divider))</c>.</summary>
    private static uint Multiplier(uint divider)
    {
        float maxInt = 4294967296f;
        uint maxDividend = 256u * divider;
        return (uint)(maxInt / maxDividend);
    }
}
