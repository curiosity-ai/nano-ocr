using SkiaSharp;

namespace IndraOcr.Imaging;

/// <summary>
/// Image decoding and encoding with SkiaSharp. Decoding reproduces what <c>cv2.imread(path,
/// IMREAD_GRAYSCALE)</c> hands the Python pipeline: gray images as stored, and for PNG colour
/// images libpng's own fixed-point <c>rgb_to_gray</c> (0.299 R + 0.587 G + 0.114 B in 15-bit
/// arithmetic, truncated), which is what OpenCV asks libpng for.
/// </summary>
public static class ImageIO
{
    /// <summary>Decodes an image file to one grayscale page per frame (animated images give every frame).</summary>
    public static List<GrayImage> ReadPages(string path)
    {
        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(path);
        }
        catch (IOException e)
        {
            throw new Pipeline.InputException($"{path}: {e.Message}", e);
        }

        List<GrayImage> pages = Decode(bytes);
        if (pages.Count == 0)
        {
            throw new Pipeline.InputException($"{path}: not a PDF or a readable image");
        }

        return pages;
    }

    /// <summary>Decodes encoded image bytes (PNG, JPEG, WebP, GIF, BMP, ICO, multi-page TIFF, …) to grayscale frames.</summary>
    public static List<GrayImage> Decode(byte[] bytes)
    {
        if (TiffDecoder.IsTiff(bytes))
        {
            return TiffDecoder.Decode(bytes);
        }

        var pages = new List<GrayImage>();
        using var data = SKData.CreateCopy(bytes);
        using SKCodec? codec = SKCodec.Create(data);
        if (codec is null)
        {
            return pages;
        }

        int frames = Math.Max(1, codec.FrameCount);
        for (int f = 0; f < frames; f++)
        {
            GrayImage? page = DecodeFrame(codec, f);
            if (page is not null)
            {
                pages.Add(page);
            }
        }

        return pages;
    }

    private static GrayImage? DecodeFrame(SKCodec codec, int frame)
    {
        SKImageInfo info = codec.Info;
        int w = info.Width, h = info.Height;
        var options = new SKCodecOptions(frame);
        if (info.ColorType == SKColorType.Gray8)
        {
            var gray = new GrayImage(w, h);
            var grayInfo = new SKImageInfo(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
            unsafe
            {
                fixed (byte* p = gray.Pixels)
                {
                    SKCodecResult r = codec.GetPixels(grayInfo, (IntPtr)p, w, options);
                    if (r is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
                    {
                        return null;
                    }
                }
            }

            return gray;
        }

        var rgbaInfo = new SKImageInfo(w, h, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var rgba = new byte[w * h * 4];
        unsafe
        {
            fixed (byte* p = rgba)
            {
                SKCodecResult r = codec.GetPixels(rgbaInfo, (IntPtr)p, w * 4, options);
                if (r is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
                {
                    return null;
                }
            }
        }

        return FromRgba(rgba, w, h, codec.EncodedFormat == SKEncodedImageFormat.Png ? GrayFormula.LibPng : GrayFormula.Bt601);
    }

    /// <summary>How colour pixels become gray.</summary>
    public enum GrayFormula
    {
        /// <summary>libpng's <c>png_set_rgb_to_gray(…, 0.299, 0.587)</c>: 15-bit coefficients 9797/19234/3737, truncated.</summary>
        LibPng,

        /// <summary>OpenCV's <c>cvtColor(RGB2GRAY)</c>: 14-bit coefficients 4899/9617/1868, rounded.</summary>
        Bt601,
    }

    /// <summary>Converts RGBA bytes to gray (alpha ignored, as <c>IMREAD_GRAYSCALE</c> does).</summary>
    public static GrayImage FromRgba(ReadOnlySpan<byte> rgba, int w, int h, GrayFormula formula = GrayFormula.Bt601)
    {
        var g = new GrayImage(w, h);
        byte[] d = g.Pixels;
        for (int i = 0; i < w * h; i++)
        {
            int r = rgba[i * 4], gr = rgba[(i * 4) + 1], b = rgba[(i * 4) + 2];
            d[i] = formula == GrayFormula.LibPng
                ? (r == gr && gr == b ? (byte)r : (byte)(((r * 9797) + (gr * 19234) + (b * 3737)) >> 15))
                : (byte)(((r * 4899) + (gr * 9617) + (b * 1868) + 8192) >> 14);
        }

        return g;
    }

    /// <summary>Wraps an <see cref="SKBitmap"/> (any colour type) as a grayscale page.</summary>
    public static GrayImage FromBitmap(SKBitmap bitmap)
    {
        if (bitmap.ColorType == SKColorType.Gray8)
        {
            var g = new GrayImage(bitmap.Width, bitmap.Height);
            for (int y = 0; y < bitmap.Height; y++)
            {
                bitmap.GetPixelSpan().Slice(y * bitmap.RowBytes, bitmap.Width).CopyTo(g.Pixels.AsSpan(y * bitmap.Width));
            }

            return g;
        }

        using SKBitmap rgba = new(new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        bitmap.CopyTo(rgba, SKColorType.Rgba8888);
        return FromRgba(rgba.GetPixelSpan(), bitmap.Width, bitmap.Height);
    }

    /// <summary>Encodes a gray image as PNG.</summary>
    public static byte[] EncodePng(GrayImage image)
    {
        var info = new SKImageInfo(image.Width, image.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        using var bmp = new SKBitmap(info);
        image.Pixels.AsSpan(0, image.Width * image.Height).CopyTo(bmp.GetPixelSpan());
        using SKData png = bmp.Encode(SKEncodedImageFormat.Png, 100);
        return png.ToArray();
    }
}
