using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>One horizontal slice of a page, as <c>split_image</c> cuts it.</summary>
/// <param name="Image">The slice, padded to the chunk height when it is the short last one.</param>
/// <param name="Height">Rows of the page the slice holds (before padding).</param>
public readonly record struct ImageChunk(RgbImage Image, int Height);

/// <summary>
/// The detector's input side: <c>split_image</c> (<c>surya/detection/util.py</c>),
/// <c>DetectionPredictor.prepare_image</c> (<c>surya/detection/__init__.py</c>) and the
/// <c>SegformerImageProcessor</c> it ends with (<c>surya/detection/processor.py</c>), reproduced
/// to the byte.
/// </summary>
public sealed class DetectionPreprocessor
{
    private readonly DetectionConfig _config;
    private readonly float[][] _lut;

    /// <summary>Creates the preprocessor for <paramref name="config"/>.</summary>
    /// <param name="config">The checkpoint's processor settings.</param>
    /// <param name="maxChunkHeight"><c>settings.DETECTOR_IMAGE_CHUNK_HEIGHT</c>.</param>
    public DetectionPreprocessor(DetectionConfig config, int maxChunkHeight = 1400)
    {
        _config = config;
        MaxChunkHeight = maxChunkHeight;

        // rescale: (uint8 → float64) · factor → float32; normalize: (x − mean) / std in float32.
        _lut = new float[3][];
        for (int c = 0; c < 3; c++)
        {
            _lut[c] = new float[256];
            float mean = (float)config.ImageMean[c];
            float std = (float)config.ImageStd[c];
            for (int v = 0; v < 256; v++)
            {
                float x = config.DoRescale ? (float)(v * config.RescaleFactor) : v;
                _lut[c][v] = config.DoNormalize ? (x - mean) / std : x;
            }
        }
    }

    /// <summary>Pages taller than this are cut into chunks of the processor's height.</summary>
    public int MaxChunkHeight { get; }

    /// <summary>
    /// <c>get_total_splits</c>: <c>ceil(height / processor_height)</c> when the page is taller than
    /// <see cref="MaxChunkHeight"/>, else 1.
    /// </summary>
    public int TotalSplits(int imageHeight) =>
        imageHeight > MaxChunkHeight ? (imageHeight + _config.ImageHeight - 1) / _config.ImageHeight : 1;

    /// <summary>
    /// <c>split_image(img, processor_height)</c>: a page taller than <see cref="MaxChunkHeight"/> is
    /// cut every <c>processor_height</c> rows, and a short last slice is padded by
    /// <c>ImageOps.pad(..., color=255, centering=(0, 0))</c> — which, because the colour is the
    /// integer 255 on an RGB image, pads with <b>red</b> <c>(255, 0, 0)</c>, not white.
    /// </summary>
    public List<ImageChunk> Split(RgbImage image)
    {
        int height = _config.ImageHeight;
        var chunks = new List<ImageChunk>();
        if (image.Height <= MaxChunkHeight)
        {
            chunks.Add(new ImageChunk(image.Clone(), image.Height));
            return chunks;
        }

        int splits = TotalSplits(image.Height);
        for (int i = 0; i < splits; i++)
        {
            int top = i * height;
            int bottom = Math.Min((i + 1) * height, image.Height);
            RgbImage cropped = image.Crop(0, top, image.Width, bottom);
            int chunkHeight = bottom - top;
            if (chunkHeight < height)
            {
                RgbImage padded = Pad(cropped, image.Width, height);
                cropped.Dispose();
                cropped = padded;
            }

            chunks.Add(new ImageChunk(cropped, chunkHeight));
        }

        return chunks;
    }

    /// <summary>
    /// <c>ImageOps.pad(image, size, color=255, centering=(0, 0))</c> for an image as wide as the
    /// target and shorter: <c>contain</c> (a bicubic resize only if the aspect arithmetic does not
    /// return the height unchanged), pasted at the top-left of a <c>(255, 0, 0)</c> canvas.
    /// </summary>
    private static RgbImage Pad(RgbImage image, int width, int height)
    {
        double imageRatio = image.Width / (double)image.Height;
        double targetRatio = width / (double)height;
        int newWidth = width;
        int newHeight = height;
        if (imageRatio != targetRatio)
        {
            if (imageRatio > targetRatio)
            {
                int h = (int)Math.Round(image.Height / (double)image.Width * width, MidpointRounding.ToEven);
                if (h != height)
                {
                    newHeight = h;
                }
            }
            else
            {
                int w = (int)Math.Round(image.Width / (double)image.Height * height, MidpointRounding.ToEven);
                if (w != width)
                {
                    newWidth = w;
                }
            }
        }

        using RgbImage resized = newWidth == image.Width && newHeight == image.Height
            ? image.Clone()
            : PilResize.ResizeBicubic(image, newWidth, newHeight);
        if (resized.Width == width && resized.Height == height)
        {
            return resized.Clone();
        }

        RgbImage canvas = RgbImage.Rent(width, height);
        Span<byte> pixels = canvas.Pixels;
        for (int i = 0; i < pixels.Length; i += 3)
        {
            pixels[i] = 255;
            pixels[i + 1] = 0;
            pixels[i + 2] = 0;
        }

        // centering (0, 0): x = round((width − w) · 0) = 0, y likewise.
        for (int y = 0; y < resized.Height; y++)
        {
            resized.Row(y).CopyTo(canvas.Row(y));
        }

        return canvas;
    }

    /// <summary>
    /// <c>prepare_image</c>'s two resizes: <c>img.thumbnail(size, LANCZOS)</c>, then
    /// <c>img.resize(size, LANCZOS)</c> to stretch the shorter side.
    /// </summary>
    public RgbImage Resize(RgbImage chunk)
    {
        int width = _config.ImageWidth;
        int height = _config.ImageHeight;
        using RgbImage thumb = PilThumbnail.Thumbnail(chunk, width, height);
        return PilResize.ResizeLanczos(thumb, width, height);
    }

    /// <summary>
    /// The processor: rescale and normalise, channels first — <c>pixel_values[0]</c>,
    /// <c>[3, height, width]</c> float32.
    /// </summary>
    public float[] Normalize(RgbImage image)
    {
        int plane = image.Width * image.Height;
        var result = new float[3 * plane];
        ReadOnlySpan<byte> pixels = image.Pixels;
        float[] r = _lut[0], g = _lut[1], b = _lut[2];
        for (int p = 0; p < plane; p++)
        {
            result[p] = r[pixels[3 * p]];
            result[plane + p] = g[pixels[(3 * p) + 1]];
            result[(2 * plane) + p] = b[pixels[(3 * p) + 2]];
        }

        return result;
    }
}
