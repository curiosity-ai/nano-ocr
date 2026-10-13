namespace IndraOcr.Vlm.Imaging;

/// <summary>
/// <c>scale_to_fit</c> (<c>surya/inference/util.py</c>, identical in <c>chandra/model/util.py</c>):
/// the pre-resize both pipelines apply before the model's own processor.
/// </summary>
public static class ScaleToFit
{
    /// <summary>
    /// The size an image is scaled to: a whole number of <paramref name="grid"/>-pixel cells, the
    /// area between <c>minWidth × minHeight</c> and <c>maxWidth × maxHeight</c>, the aspect ratio
    /// kept as closely as the cells allow.
    /// </summary>
    public static (int Width, int Height) TargetSize(
        int width,
        int height,
        int maxWidth = 3072,
        int maxHeight = 2048,
        int minWidth = 1792,
        int minHeight = 28,
        int grid = 28)
    {
        if (width <= 0 || height <= 0)
        {
            return (width, height);
        }

        double originalAspect = width / (double)height;
        long currentPixels = (long)width * height;
        long maxPixels = (long)maxWidth * maxHeight;
        long minPixels = (long)minWidth * minHeight;

        double scale = 1.0;
        if (currentPixels > maxPixels)
        {
            scale = Math.Pow(maxPixels / (double)currentPixels, 0.5);
        }
        else if (currentPixels < minPixels)
        {
            scale = Math.Pow(minPixels / (double)currentPixels, 0.5);
        }

        // Python's round() rounds half to even.
        long wBlocks = Math.Max(1, (long)Math.Round(width * scale / grid));
        long hBlocks = Math.Max(1, (long)Math.Round(height * scale / grid));

        while (wBlocks * hBlocks * grid * grid > maxPixels)
        {
            if (wBlocks == 1 && hBlocks == 1)
            {
                break;
            }

            if (wBlocks == 1)
            {
                hBlocks--;
                continue;
            }

            if (hBlocks == 1)
            {
                wBlocks--;
                continue;
            }

            double wLoss = Math.Abs(((wBlocks - 1) / (double)hBlocks) - originalAspect);
            double hLoss = Math.Abs((wBlocks / (double)(hBlocks - 1)) - originalAspect);
            if (wLoss < hLoss)
            {
                wBlocks--;
            }
            else
            {
                hBlocks--;
            }
        }

        return ((int)(wBlocks * grid), (int)(hBlocks * grid));
    }

    /// <summary>Scales an image with Pillow's Lanczos filter; returns a copy when no scaling is needed.</summary>
    public static RgbImage Apply(RgbImage image)
    {
        (int width, int height) = TargetSize(image.Width, image.Height);
        return width == image.Width && height == image.Height
            ? image.Clone()
            : PilResize.ResizeLanczos(image, width, height);
    }
}
