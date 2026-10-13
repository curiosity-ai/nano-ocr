using IndraOcr.Imaging;
using SkiaSharp;

namespace IndraOcr.Vlm.Imaging;

/// <summary>Colour image decoding and encoding (SkiaSharp; TIFF through IndraOcr's decoder).</summary>
public static class RgbImageIO
{
    /// <summary>Whether the bytes are a TIFF.</summary>
    public static bool IsTiff(ReadOnlySpan<byte> bytes) => TiffDecoder.IsTiff(bytes);

    /// <summary>
    /// Decodes the first frame as <c>PIL.Image.open(path).convert("RGB")</c> does: alpha is dropped,
    /// grey is replicated, palettes are expanded.
    /// </summary>
    public static RgbImage Decode(byte[] bytes)
    {
        if (TiffDecoder.IsTiff(bytes))
        {
            // IndraOcr's TIFF decoder yields grey frames; the scans TIFF is used for are grey.
            GrayImage gray = TiffDecoder.Decode(bytes)[0];
            return FromGray(gray.Pixels, gray.Width, gray.Height);
        }

        using SKBitmap? bitmap = SKBitmap.Decode(bytes, new SKImageInfo { ColorType = SKColorType.Rgba8888, AlphaType = SKAlphaType.Unpremul })
            ?? SKBitmap.Decode(bytes);
        if (bitmap is null)
        {
            throw new InvalidDataException("cannot decode the image");
        }

        using SKBitmap rgba = bitmap.ColorType == SKColorType.Rgba8888 && bitmap.AlphaType == SKAlphaType.Unpremul
            ? bitmap.Copy()
            : Convert(bitmap);
        ReadOnlySpan<byte> pixels = rgba.GetPixelSpan();
        int width = rgba.Width;
        int height = rgba.Height;
        var rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = pixels.Slice(y * rgba.RowBytes, width * 4);
            Span<byte> target = rgb.AsSpan(y * width * 3, width * 3);
            for (int x = 0; x < width; x++)
            {
                target[(x * 3) + 0] = row[(x * 4) + 0];
                target[(x * 3) + 1] = row[(x * 4) + 1];
                target[(x * 3) + 2] = row[(x * 4) + 2];
            }
        }

        return RgbImage.Wrap(rgb, width, height);
    }

    /// <summary>Reads and decodes a file.</summary>
    public static RgbImage Read(string path) => Decode(File.ReadAllBytes(path));

    /// <summary>An RGB image from 8-bit grey.</summary>
    public static RgbImage FromGray(ReadOnlySpan<byte> gray, int width, int height)
    {
        var rgb = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++)
        {
            rgb[i * 3] = rgb[(i * 3) + 1] = rgb[(i * 3) + 2] = gray[i];
        }

        return RgbImage.Wrap(rgb, width, height);
    }

    /// <summary>Encodes as PNG.</summary>
    public static byte[] EncodePng(RgbImage image) => Encode(image, SKEncodedImageFormat.Png, 100);

    /// <summary>Encodes as WebP at Pillow's default quality (80).</summary>
    public static byte[] EncodeWebp(RgbImage image) => Encode(image, SKEncodedImageFormat.Webp, 80);

    /// <summary>Encodes as JPEG.</summary>
    public static byte[] EncodeJpeg(RgbImage image, int quality = 95) => Encode(image, SKEncodedImageFormat.Jpeg, quality);

    private static byte[] Encode(RgbImage image, SKEncodedImageFormat format, int quality)
    {
        using var bitmap = new SKBitmap(new SKImageInfo(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Opaque));
        Span<byte> target = bitmap.GetPixelSpan();
        ReadOnlySpan<byte> source = image.Pixels;
        for (int y = 0; y < image.Height; y++)
        {
            Span<byte> row = target.Slice(y * bitmap.RowBytes, image.Width * 4);
            ReadOnlySpan<byte> src = source.Slice(y * image.Width * 3, image.Width * 3);
            for (int x = 0; x < image.Width; x++)
            {
                row[(x * 4) + 0] = src[(x * 3) + 0];
                row[(x * 4) + 1] = src[(x * 3) + 1];
                row[(x * 4) + 2] = src[(x * 3) + 2];
                row[(x * 4) + 3] = 255;
            }
        }

        using SKData data = bitmap.Encode(format, quality);
        return data.ToArray();
    }

    private static SKBitmap Convert(SKBitmap source)
    {
        var target = new SKBitmap(new SKImageInfo(source.Width, source.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        if (!source.CopyTo(target, SKColorType.Rgba8888))
        {
            using var canvas = new SKCanvas(target);
            canvas.DrawBitmap(source, 0, 0);
        }

        return target;
    }
}
