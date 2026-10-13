using System.Diagnostics;
using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>
/// One layout region — <c>surya.layout.schema.LayoutBox</c>: a polygon (here always the four
/// corners of an axis-aligned box), the canonical and raw labels, the reading position and the
/// detector's confidence.
/// </summary>
public sealed class LayoutBox
{
    /// <summary>Corners <c>[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]</c> (<c>_poly</c>).</summary>
    public double[][] Polygon { get; internal set; } = [];

    /// <summary>The label after <c>LAYOUT_PRED_RELABEL</c> (e.g. <c>SectionHeader</c>).</summary>
    public string Label { get; init; } = string.Empty;

    /// <summary>The detector's own label (e.g. <c>Section-Header</c>).</summary>
    public string RawLabel { get; init; } = string.Empty;

    /// <summary>Reading-order index, 0 first.</summary>
    public int Position { get; internal set; }

    /// <summary>Sigmoid confidence of the detection.</summary>
    public double Confidence { get; init; }

    /// <summary>The model's token estimate (<c>count</c>); the fast layout path leaves it 0.</summary>
    public int Count { get; init; }

    /// <summary><c>PolygonBox.bbox</c>: <c>[min x, min y, max x, max y]</c>.</summary>
    public double[] Bbox => [
        Polygon.Min(p => p[0]), Polygon.Min(p => p[1]), Polygon.Max(p => p[0]), Polygon.Max(p => p[1])];

    /// <summary><c>PolygonBox.area</c>.</summary>
    public double Area
    {
        get
        {
            double[] b = Bbox;
            return (b[2] - b[0]) * (b[3] - b[1]);
        }
    }

    /// <summary><c>PolygonBox.intersection_area</c> (no margins).</summary>
    public double IntersectionArea(LayoutBox other)
    {
        double[] a = Bbox, b = other.Bbox;
        double x = Math.Max(0, Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]));
        double y = Math.Max(0, Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]));
        return x * y;
    }

    internal static double[][] Poly(double x0, double y0, double x1, double y1) =>
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

/// <summary><c>surya.layout.schema.LayoutResult</c>: a page's boxes in reading order.</summary>
/// <param name="Boxes">The regions, sorted by <see cref="LayoutBox.Position"/>.</param>
/// <param name="ImageBbox"><c>[0, 0, width, height]</c>.</param>
public sealed record LayoutResult(IReadOnlyList<LayoutBox> Boxes, double[] ImageBbox);

/// <summary>Options for <see cref="FastLayoutPredictor.Predict"/>.</summary>
public sealed record FastLayoutOptions
{
    /// <summary><c>FAST_LAYOUT_CONFIDENCE_THRESHOLD</c>.</summary>
    public float Threshold { get; init; } = 0.4f;

    /// <summary><c>FAST_LAYOUT_CONTAINMENT_THRESHOLD</c>; above 1 disables the merge.</summary>
    public double ContainmentThreshold { get; init; } = 0.9;

    /// <summary><c>FAST_LAYOUT_USE_ORDER</c>: run the learned reading-order head, else raster order.</summary>
    public bool UseOrder { get; init; } = true;
}

/// <summary>
/// Surya's fast layout: the RF-DETR detector plus the learned reading-order head —
/// <c>surya.fast_layout</c> (<c>FastLayoutPredictor</c>, served by <c>LayoutEngine.run_batch</c>, and
/// <c>build_layout_result</c>) in process, without the shared server in front of it.
/// </summary>
/// <remarks>
/// Load it from the <c>datalab-to/surya_layout2</c> folder: <c>config.json</c> and
/// <c>rfdetr_layout.pth</c> for the detector, <c>order/order_ar.pt</c> for the reading order
/// (optional; without it pages are raster-sorted, as upstream does when the order model is
/// unavailable).
/// </remarks>
public sealed class FastLayoutPredictor
{
    /// <summary><c>surya.layout.label.LAYOUT_PRED_RELABEL</c>.</summary>
    public static readonly IReadOnlyDictionary<string, string> Relabel = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["Caption"] = "Caption",
        ["Footnote"] = "Footnote",
        ["Equation-Block"] = "Equation",
        ["List-Group"] = "ListGroup",
        ["Page-Header"] = "PageHeader",
        ["Page-Footer"] = "PageFooter",
        ["Image"] = "Picture",
        ["Section-Header"] = "SectionHeader",
        ["Table"] = "Table",
        ["Text"] = "Text",
        ["Complex-Block"] = "Figure",
        ["Code-Block"] = "Code",
        ["Form"] = "Form",
        ["Table-Of-Contents"] = "TableOfContents",
        ["Figure"] = "Figure",
        ["Chemical-Block"] = "ChemicalBlock",
        ["Diagram"] = "Diagram",
        ["Bibliography"] = "Bibliography",
        ["Blank-Page"] = "BlankPage",
    };

    private FastLayoutPredictor(RfDetr detector, ReadingOrderModel? order)
    {
        Detector = detector;
        Order = order;
    }

    /// <summary>The detector.</summary>
    public RfDetr Detector { get; }

    /// <summary>The reading-order head, when its checkpoint was present.</summary>
    public ReadingOrderModel? Order { get; }

    /// <summary>Stage timings of the last <see cref="Predict"/> call, in milliseconds.</summary>
    public IReadOnlyDictionary<string, double> LastTimings { get; private set; } = new Dictionary<string, double>();

    /// <summary>Loads the detector from <paramref name="directory"/> and the order head from its <c>order</c> subfolder.</summary>
    public static FastLayoutPredictor Load(string directory)
    {
        RfDetr detector = RfDetr.Load(directory);
        string orderDir = Path.Combine(directory, "order");
        ReadingOrderModel? order = File.Exists(Path.Combine(orderDir, "order_ar.pt")) ? ReadingOrderModel.Load(orderDir) : null;
        return new FastLayoutPredictor(detector, order);
    }

    /// <summary>Detects and orders the layout regions of one page.</summary>
    public LayoutResult Predict(RgbImage image, FastLayoutOptions? options = null, LayoutTrace? trace = null)
    {
        options ??= new FastLayoutOptions();
        var timings = new Dictionary<string, double>();
        long mark = Stopwatch.GetTimestamp();
        float[] pixels = Detector.Preprocess(image);
        trace?.Invoke("pixel_values", pixels);
        timings["preprocess"] = Stopwatch.GetElapsedTime(mark).TotalMilliseconds;

        mark = Stopwatch.GetTimestamp();
        RfDetr.Output output = Detector.Forward(pixels, trace);
        timings["model"] = Stopwatch.GetElapsedTime(mark).TotalMilliseconds;
        foreach ((string stage, double ms) in Detector.LastTimings)
        {
            timings["model." + stage] = ms;
        }

        List<RfDetrDetection> dets = Detector.PostProcess(output, image.Width, image.Height, options.Threshold, trace);

        mark = Stopwatch.GetTimestamp();
        int[] positions;
        var boxes = dets.Select(d => (d.X0, d.Y0, d.X1, d.Y1)).ToList();
        if (options.UseOrder && Order is not null && dets.Count > 0)
        {
            positions = Order.OrderPage(output.Features, boxes, dets.Select(d => d.Label).ToList(), image.Width, image.Height, trace);
        }
        else
        {
            // build_layout_result's raster fallback: sorted by (y0, x0).
            positions = ReadingOrderModel.RasterPositions(boxes);
        }

        timings["order"] = Stopwatch.GetElapsedTime(mark).TotalMilliseconds;
        LastTimings = timings;
        return BuildLayoutResult(image.Width, image.Height, dets, positions, options.ContainmentThreshold);
    }

    /// <summary><c>build_layout_result</c> after the positions are known: relabel, merge contained boxes, sort.</summary>
    public static LayoutResult BuildLayoutResult(
        int width, int height, IReadOnlyList<RfDetrDetection> dets, IReadOnlyList<int> positions, double containment)
    {
        var boxes = new List<LayoutBox>(dets.Count);
        for (int i = 0; i < dets.Count; i++)
        {
            RfDetrDetection d = dets[i];
            boxes.Add(new LayoutBox
            {
                Polygon = LayoutBox.Poly(d.X0, d.Y0, d.X1, d.Y1),
                Label = Relabel.TryGetValue(d.Label, out string? l) ? l : d.Label,
                RawLabel = d.Label,
                Position = positions[i],
                Confidence = d.Score,
            });
        }

        List<LayoutBox> merged = MergeContainedBoxes(boxes, containment);
        merged = merged.OrderBy(b => b.Position).ToList();
        return new LayoutResult(merged, [0.0, 0.0, width, height]);
    }

    /// <summary>
    /// <c>_merge_contained_boxes</c>: largest first, a box with at least <paramref name="threshold"/>
    /// of its area inside a surviving box of the same label is absorbed into it (the survivor grows
    /// to their union); positions are then renumbered contiguously.
    /// </summary>
    public static List<LayoutBox> MergeContainedBoxes(List<LayoutBox> boxes, double threshold)
    {
        if (threshold > 1 || boxes.Count < 2)
        {
            return boxes;
        }

        var survivors = new List<LayoutBox>();
        foreach (LayoutBox b in boxes.OrderByDescending(x => x.Area))
        {
            bool absorbed = false;
            foreach (LayoutBox s in survivors)
            {
                if (s.Label != b.Label)
                {
                    continue;
                }

                double area = b.Area;
                if (area > 0 && s.IntersectionArea(b) / area >= threshold)
                {
                    double[] sb = s.Bbox, bb = b.Bbox;
                    s.Polygon = LayoutBox.Poly(
                        Math.Min(sb[0], bb[0]), Math.Min(sb[1], bb[1]), Math.Max(sb[2], bb[2]), Math.Max(sb[3], bb[3]));
                    absorbed = true;
                    break;
                }
            }

            if (!absorbed)
            {
                survivors.Add(b);
            }
        }

        survivors = survivors.OrderBy(x => x.Position).ToList();
        for (int rank = 0; rank < survivors.Count; rank++)
        {
            survivors[rank].Position = rank;
        }

        return survivors;
    }
}
