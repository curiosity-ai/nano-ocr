namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>
/// <c>surya.common.polygon.PolygonBox</c>: four corners, clockwise from the top-left, and a
/// confidence. After rescaling to the page every coordinate is a whole number, as in Surya.
/// </summary>
public sealed class PolygonBox
{
    /// <summary>Creates a box from its corners, <c>[[x, y] × 4]</c>.</summary>
    public PolygonBox(double[][] polygon, double? confidence)
    {
        Polygon = polygon;
        Confidence = confidence;
    }

    /// <summary>The corners, <c>[[x, y] × 4]</c>.</summary>
    public double[][] Polygon { get; private set; }

    /// <summary>The line's peak heatmap value relative to the page's strongest line.</summary>
    public double? Confidence { get; }

    /// <summary><c>[x_min, y_min, x_max, y_max]</c> of the corners.</summary>
    public double[] Bbox => [
        Polygon.Min(p => p[0]), Polygon.Min(p => p[1]), Polygon.Max(p => p[0]), Polygon.Max(p => p[1]),
    ];

    /// <summary><c>bbox[3] − bbox[1]</c>.</summary>
    public double Height
    {
        get
        {
            double[] b = Bbox;
            return b[3] - b[1];
        }
    }

    /// <summary><c>bbox[2] − bbox[0]</c>.</summary>
    public double Width
    {
        get
        {
            double[] b = Bbox;
            return b[2] - b[0];
        }
    }

    /// <summary><c>rescale(processor_size, image_size)</c>: scales and truncates every corner.</summary>
    public void Rescale((int Width, int Height) processorSize, (int Width, int Height) imageSize)
    {
        double widthScaler = imageSize.Width / (double)processorSize.Width;
        double heightScaler = imageSize.Height / (double)processorSize.Height;
        foreach (double[] corner in Polygon)
        {
            corner[0] = Math.Truncate(corner[0] * widthScaler);
            corner[1] = Math.Truncate(corner[1] * heightScaler);
        }
    }

    /// <summary><c>fit_to_bounds([x0, y0, x1, y1])</c>: clamps every corner.</summary>
    public void FitToBounds(double x0, double y0, double x1, double y1)
    {
        Polygon = Polygon.Select(c => new[] { Math.Max(Math.Min(c[0], x1), x0), Math.Max(Math.Min(c[1], y1), y0) }).ToArray();
    }

    /// <summary>
    /// <c>expand(x_margin, y_margin)</c>: moves the corners outward by the margins times the
    /// box's width and height, truncating.
    /// </summary>
    public void Expand(double xMargin, double yMargin)
    {
        xMargin *= Width;
        yMargin *= Height;
        var result = new double[4][];
        for (int i = 0; i < 4; i++)
        {
            double[] p = Polygon[i];
            result[i] = i switch
            {
                0 => [Math.Truncate(p[0] - xMargin), Math.Truncate(p[1] - yMargin)],
                1 => [Math.Truncate(p[0] + xMargin), Math.Truncate(p[1] - yMargin)],
                2 => [Math.Truncate(p[0] + xMargin), Math.Truncate(p[1] + yMargin)],
                _ => [Math.Truncate(p[0] - xMargin), Math.Truncate(p[1] + yMargin)],
            };
        }

        Polygon = result;
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"[{string.Join(", ", Polygon.Select(p => $"({p[0]}, {p[1]})"))}] {Confidence:F4}";
}

/// <summary><c>surya.detection.schema.TextDetectionResult</c>.</summary>
public sealed class TextDetectionResult
{
    /// <summary>The text lines, in the order the heatmap's connected components were labelled.</summary>
    public required IReadOnlyList<PolygonBox> Bboxes { get; init; }

    /// <summary>
    /// The stitched text heatmap (<c>[rows, processor_width]</c>, 0-1), when requested — the
    /// float map <c>Image.fromarray((heatmap * 255).astype(np.uint8))</c> is made from.
    /// </summary>
    public float[]? Heatmap { get; init; }

    /// <summary>The stitched affinity map, when requested.</summary>
    public float[]? AffinityMap { get; init; }

    /// <summary>Rows of <see cref="Heatmap"/>.</summary>
    public int HeatmapHeight { get; init; }

    /// <summary>Columns of <see cref="Heatmap"/>.</summary>
    public int HeatmapWidth { get; init; }

    /// <summary><c>[0, 0, width, height]</c> of the page.</summary>
    public required double[] ImageBbox { get; init; }

    /// <summary>
    /// A heatmap as the 8-bit grey image Surya returns: <c>(map · 255).astype(np.uint8)</c>
    /// (float32 product, truncated).
    /// </summary>
    public static byte[] ToGray(float[] map)
    {
        var bytes = new byte[map.Length];
        for (int i = 0; i < map.Length; i++)
        {
            bytes[i] = (byte)(int)(map[i] * 255f);
        }

        return bytes;
    }
}
