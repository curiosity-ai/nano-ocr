using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Models;

/// <summary>An image as the vision tower takes it: flattened patches and their grid.</summary>
/// <param name="Pixels">
/// <c>[GridH · GridW, 3 · temporal · patch · patch]</c>, row-major, patches in spatial-merge-block
/// order (each 2×2 block of patches consecutive).
/// </param>
/// <param name="GridH">Patch rows.</param>
/// <param name="GridW">Patch columns.</param>
public sealed record PreprocessedImage(float[] Pixels, int GridH, int GridW)
{
    /// <summary>Frames (always one for an image).</summary>
    public int GridT => 1;

    /// <summary>Patches the tower encodes.</summary>
    public int PatchCount => GridH * GridW;
}

/// <summary>
/// <c>Qwen2VLImageProcessor</c> (transformers' torchvision backend): <c>smart_resize</c>, a uint8
/// antialiased bicubic resize, the fused rescale-and-normalise, and <c>patchify</c>.
/// </summary>
public sealed class ImageProcessor
{
    private readonly int _patch;
    private readonly int _merge;
    private readonly int _temporal;
    private readonly float[] _mean;
    private readonly float[] _std;

    /// <summary>Creates a processor for a checkpoint's configuration.</summary>
    /// <param name="config">The checkpoint.</param>
    /// <param name="minPixels">Overrides <c>size.shortest_edge</c> (Surya's server passes 3136).</param>
    /// <param name="maxPixels">Overrides <c>size.longest_edge</c> (Surya's server passes 6291456).</param>
    public ImageProcessor(Qwen35Config config, int? minPixels = null, int? maxPixels = null)
    {
        _patch = config.PatchSize;
        _merge = config.SpatialMergeSize;
        _temporal = config.TemporalPatchSize;
        MinPixels = minPixels ?? config.MinPixels;
        MaxPixels = maxPixels ?? config.MaxPixels;

        // _fuse_mean_std_and_rescale_factor: mean and std are multiplied by 1 / rescale_factor
        // (255.00000000000003, which is 255 once it meets a float32 tensor) and the image is
        // normalised in float32 as (x - mean) / std.
        _mean = config.ImageMean.Select(m => m * 255f).ToArray();
        _std = config.ImageStd.Select(s => s * 255f).ToArray();
    }

    /// <summary>The minimum pixel count after resizing.</summary>
    public int MinPixels { get; }

    /// <summary>The maximum pixel count after resizing.</summary>
    public int MaxPixels { get; }

    /// <summary>
    /// <c>smart_resize</c> (<c>image_processing_qwen2_vl.py</c>): both sides a multiple of
    /// <paramref name="factor"/>, the area inside the pixel bounds, the aspect ratio kept.
    /// </summary>
    public static (int Height, int Width) SmartResize(int height, int width, int factor, int minPixels, int maxPixels)
    {
        if (Math.Max(height, width) / (double)Math.Min(height, width) > 200)
        {
            throw new ArgumentException(
                $"absolute aspect ratio must be smaller than 200, got {Math.Max(height, width) / (double)Math.Min(height, width)}");
        }

        // Python's round() is round-half-to-even, which is Math.Round's default.
        int hBar = (int)Math.Round(height / (double)factor) * factor;
        int wBar = (int)Math.Round(width / (double)factor) * factor;
        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt(height * (double)width / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt(minPixels / (height * (double)width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }

        return (hBar, wBar);
    }

    /// <summary>The size an image is resized to before patchifying.</summary>
    public (int Height, int Width) TargetSize(int height, int width) =>
        SmartResize(height, width, _patch * _merge, MinPixels, MaxPixels);

    /// <summary>Resizes an image the way the processor does (uint8 in, uint8 out).</summary>
    public RgbImage Resize(RgbImage image)
    {
        (int height, int width) = TargetSize(image.Height, image.Width);
        return width == image.Width && height == image.Height
            ? image.Clone()
            : PilResize.ResizeTorchBicubic(image, width, height);
    }

    /// <summary>Resizes, normalises and patchifies an image.</summary>
    public PreprocessedImage Process(RgbImage image)
    {
        using RgbImage resized = Resize(image);
        return Patchify(resized);
    }

    /// <summary>Normalises and patchifies an image already at its target size.</summary>
    public PreprocessedImage Patchify(RgbImage image)
    {
        int p = _patch;
        int m = _merge;
        int gridH = image.Height / p;
        int gridW = image.Width / p;
        int patchLength = 3 * _temporal * p * p;
        var pixels = new float[gridH * gridW * patchLength];

        // patches.permute(0, 2, 5, 3, 6, 1, 4, 7): [blockRow, blockCol, inRow, inCol, channel,
        // (temporal copies), py, px].
        Parallel.For(0, gridH / m, blockRow =>
        {
            ReadOnlySpan<byte> pixelsIn = image.Pixels;
            for (int blockCol = 0; blockCol < gridW / m; blockCol++)
            {
                for (int inRow = 0; inRow < m; inRow++)
                {
                    for (int inCol = 0; inCol < m; inCol++)
                    {
                        int patchIndex = (((((blockRow * (gridW / m)) + blockCol) * m) + inRow) * m) + inCol;
                        int y0 = ((blockRow * m) + inRow) * p;
                        int x0 = ((blockCol * m) + inCol) * p;
                        Span<float> target = pixels.AsSpan(patchIndex * patchLength, patchLength);
                        for (int c = 0; c < 3; c++)
                        {
                            float mean = _mean[c];
                            float std = _std[c];
                            int frame = p * p;
                            Span<float> first = target.Slice(c * _temporal * frame, frame);
                            for (int py = 0; py < p; py++)
                            {
                                int row = ((y0 + py) * image.Width + x0) * 3 + c;
                                for (int px = 0; px < p; px++)
                                {
                                    first[(py * p) + px] = (pixelsIn[row + (px * 3)] - mean) / std;
                                }
                            }

                            for (int t = 1; t < _temporal; t++)
                            {
                                first.CopyTo(target.Slice(((c * _temporal) + t) * frame, frame));
                            }
                        }
                    }
                }
            }
        });

        return new PreprocessedImage(pixels, gridH, gridW);
    }
}
