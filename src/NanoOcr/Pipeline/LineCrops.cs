using NanoOcr.Imaging;
using NanoOcr.Onnx;

namespace NanoOcr.Pipeline;

/// <summary>A recognizer input: an ink map <c>[Height, Width]</c> (0 = paper, 1 = ink).</summary>
public sealed class LineCrop(float[] ink, int height, int width)
{
    /// <summary>The ink values, row-major.</summary>
    public float[] Ink { get; } = ink;

    /// <summary>Rows (the recognizer's line height).</summary>
    public int Height { get; } = height;

    /// <summary>Columns.</summary>
    public int Width { get; } = width;
}

/// <summary>
/// The line crop rule shared by recognition training and inference (port of
/// <c>indra_ocr/rec_preproc.py</c>): pad the box, crop it with paper outside the page, and scale
/// to a fixed height keeping the aspect ratio.
/// </summary>
public static class LineCrops
{
    /// <summary>Default recognizer input height.</summary>
    public const int LineHeight = 32;

    /// <summary>Box heights added above and below.</summary>
    public const double PadY = 0.2;

    /// <summary>Box heights added left and right.</summary>
    public const double PadX = 0.3;

    /// <summary>Widths are padded to a multiple of this (the recognizer's timestep stride).</summary>
    public const int WidthMultiple = 8;

    /// <summary><c>crop_line</c>: one line of a page as a recognizer ink map.</summary>
    public static LineCrop Crop(GrayImage gray, in LineBox box, int height = LineHeight, double padY = PadY, double padX = PadX, int maxWidth = 4096)
    {
        double x0 = box.X0, y0 = box.Y0, x1 = box.X1, y1 = box.Y1;
        double h = Math.Max(1.0, y1 - y0);
        double bx0 = x0 - (padX * h), bx1 = x1 + (padX * h);
        double by0 = y0 - (padY * h), by1 = y1 + (padY * h);
        int ix0 = (int)Math.Floor(bx0), iy0 = (int)Math.Floor(by0);
        int ix1 = (int)Math.Ceiling(bx1), iy1 = (int)Math.Ceiling(by1);
        int cw = Math.Max(1, ix1 - ix0), ch = Math.Max(1, iy1 - iy0);
        GrayImage crop = GrayImage.Rent(cw, ch);
        crop.Pixels.AsSpan(0, cw * ch).Fill(255);
        int sx0 = Math.Max(0, ix0), sy0 = Math.Max(0, iy0), sx1 = Math.Min(gray.Width, ix1), sy1 = Math.Min(gray.Height, iy1);
        if (sx1 > sx0 && sy1 > sy0)
        {
            for (int y = sy0; y < sy1; y++)
            {
                gray.Pixels.AsSpan((y * gray.Width) + sx0, sx1 - sx0).CopyTo(crop.Pixels.AsSpan(((y - iy0) * cw) + (sx0 - ix0)));
            }
        }

        int outW = (int)Math.Max(WidthMultiple, Math.Min(maxWidth, Py.Round(cw * (double)height / ch)));
        GrayImage resized = GrayImage.Rent(outW, height);
        CvResize.ResizeInto(crop, resized, ch > height ? Interpolation.Area : Interpolation.Linear);
        crop.Return();
        var ink = new float[height * outW];
        Detection.InkMap(resized, ink, outW);
        resized.Return();
        return new LineCrop(ink, height, outW);
    }

    /// <summary>
    /// <c>pad_batch</c>: right-pads crops to one width (a multiple of 8) as <c>[B, 1, H, W]</c>,
    /// with each crop's true width.
    /// </summary>
    public static (Tensor Batch, Tensor Widths) PadBatch(IReadOnlyList<LineCrop> lines, int multiple = WidthMultiple)
    {
        int h = lines[0].Height;
        int wm = lines.Max(l => l.Width);
        wm = (wm + multiple - 1) / multiple * multiple;
        Tensor batch = Tensor.Float([lines.Count, 1, h, wm], clear: true);
        Tensor widths = Tensor.Int64([lines.Count]);
        float[] dst = batch.FloatArray;
        for (int i = 0; i < lines.Count; i++)
        {
            LineCrop l = lines[i];
            for (int y = 0; y < h; y++)
            {
                l.Ink.AsSpan(y * l.Width, l.Width).CopyTo(dst.AsSpan((((i * h) + y) * wm), l.Width));
            }

            widths.LongArray[i] = l.Width;
        }

        return (batch, widths);
    }

    /// <summary>
    /// <c>width_batches</c>: line indices grouped into batches of similar width, narrowest first.
    /// A batch closes at <paramref name="maxBatch"/> lines, when the next line is wider than
    /// <paramref name="maxRatio"/> times the batch's first, or past <paramref name="maxArea"/> columns.
    /// </summary>
    public static List<List<int>> WidthBatches(IReadOnlyList<int> widths, int maxBatch = 64, double maxRatio = 1.3, int minWidth = 64, int maxArea = 64 * 640)
    {
        int[] order = Enumerable.Range(0, widths.Count).OrderBy(i => widths[i]).ToArray();
        var batches = new List<List<int>>();
        foreach (int i in order)
        {
            if (batches.Count > 0)
            {
                List<int> last = batches[^1];
                int wi = Math.Max(widths[i], minWidth);
                if (last.Count < maxBatch
                    && wi <= maxRatio * Math.Max(widths[last[0]], minWidth)
                    && (long)(last.Count + 1) * wi <= maxArea)
                {
                    last.Add(i);
                    continue;
                }
            }

            batches.Add([i]);
        }

        return batches;
    }
}
