using System.Diagnostics;
using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya.Layout;

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

    /// <summary>The Hub repository of the checkpoint (<c>FAST_LAYOUT_MODEL_CHECKPOINT</c>).</summary>
    public const string Repository = "datalab-to/surya_layout2";

    /// <summary>
    /// Loads the detector from <paramref name="directory"/> and the order head from its
    /// <c>order</c> subfolder; without a directory, <c>SURYA_LAYOUT_MODEL_DIR</c> or the download cache.
    /// </summary>
    public static FastLayoutPredictor Load(string? directory = null)
    {
        directory ??= Environment.GetEnvironmentVariable("SURYA_LAYOUT_MODEL_DIR");
        directory ??= Common.ModelDownloader.Ensure(Repository, ["config.json", "rfdetr_layout.pth", "order/config.json", "order/order_ar.pt"]);
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
                Polygon = PolygonBox.FromBox(d.X0, d.Y0, d.X1, d.Y1),
                Label = Relabel.TryGetValue(d.Label, out string? l) ? l : d.Label,
                RawLabel = d.Label,
                Position = positions[i],
                Confidence = d.Score,
            });
        }

        List<LayoutBox> merged = MergeContainedBoxes(boxes, containment);
        merged = merged.OrderBy(b => b.Position).ToList();
        return new LayoutResult { Bboxes = merged, ImageBbox = [0.0, 0.0, width, height] };
    }

    /// <summary><c>PolygonBox.area</c>.</summary>
    private static double Area(LayoutBox box)
    {
        double[] b = box.Bbox;
        return (b[2] - b[0]) * (b[3] - b[1]);
    }

    /// <summary><c>PolygonBox.intersection_area</c> (no margins).</summary>
    private static double IntersectionArea(LayoutBox a, LayoutBox b)
    {
        double[] p = a.Bbox, q = b.Bbox;
        double x = Math.Max(0, Math.Min(p[2], q[2]) - Math.Max(p[0], q[0]));
        double y = Math.Max(0, Math.Min(p[3], q[3]) - Math.Max(p[1], q[1]));
        return x * y;
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
        foreach (LayoutBox b in boxes.OrderByDescending(Area))
        {
            bool absorbed = false;
            foreach (LayoutBox s in survivors)
            {
                if (s.Label != b.Label)
                {
                    continue;
                }

                double area = Area(b);
                if (area > 0 && IntersectionArea(s, b) / area >= threshold)
                {
                    double[] sb = s.Bbox, bb = b.Bbox;
                    s.Polygon = PolygonBox.FromBox(
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
