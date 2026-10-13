using System.Diagnostics;
using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Models;
using IndraOcr.Onnx;

namespace IndraOcr.Pipeline;

/// <summary>Options of <see cref="IndraEngine"/>.</summary>
public sealed class EngineOptions
{
    /// <summary>Where the models come from (default: the embedded copies).</summary>
    public ModelStore? Models { get; set; }

    /// <summary>Run a second detection pass at a higher resolution on pages of small text.</summary>
    public bool Adaptive { get; set; } = true;

    /// <summary>Build tables from the order model's row / column / cell heads instead of from geometry.</summary>
    public bool TableHeads { get; set; }

    /// <summary>Keep page headers and footers in the markdown.</summary>
    public bool Furniture { get; set; }

    /// <summary>Read lines with math as LaTeX (slower on math pages).</summary>
    public bool Math { get; set; }
}

/// <summary>
/// The OCR pipeline (port of <c>indra_ocr/pipeline.py</c>'s <c>OrtOCR</c>): orientation and
/// deskew, line detection, CTC line recognition routed by script, line splitting at gaps, reading
/// order, layout classes and markdown. Thread-safe: pages may be processed concurrently.
/// </summary>
public sealed class IndraEngine
{
    /// <summary>Median line height (input px) that triggers a second detection pass.</summary>
    public const double AdaptiveMinH = 9.0;

    /// <summary>Target median line height of the second pass.</summary>
    public const double AdaptiveTargetH = 12.0;

    /// <summary>Cut a detected line at a wider gap (line heights between character centres).</summary>
    public const double SplitGap = 3.0;

    /// <summary>Cut a table line into cells at a wider gap.</summary>
    public const double CellGap = 1.8;

    /// <summary>Cut at a smaller gap when other lines' edges line up with it.</summary>
    public const double GutterGap = 1.6;

    /// <summary>How many other line edges must line up with a gutter.</summary>
    public const int GutterEdges = 3;

    /// <summary>Longest side of the adaptive second pass.</summary>
    public const int AdaptiveMaxSide = 2048;

    /// <summary>Side of the orientation model's square input.</summary>
    public const int OrientSide = 768;

    /// <summary>Probability above which a turn is applied.</summary>
    public const float OrientMin = 0.95f;

    /// <summary>A recognized line under this confidence gets no text.</summary>
    public const double MinConf = 0.5;

    /// <summary>Degrees of tilt searched.</summary>
    public const double DeskewMax = 5.0;

    /// <summary>Smaller tilts are left alone.</summary>
    public const double DeskewMin = 0.3;

    private readonly InferenceSession _det;
    private readonly InferenceSession _rec;
    private readonly InferenceSession? _order;
    private readonly InferenceSession? _orient;
    private readonly InferenceSession? _script;
    private readonly InferenceSession? _recCjk;
    private readonly int _spaceId;

    /// <summary>Loads the models.</summary>
    public IndraEngine(EngineOptions? options = null)
    {
        Options = options ?? new EngineOptions();
        ModelStore store = Options.Models ?? ModelStore.Embedded;
        _det = store.Session("det");
        _rec = store.Session("rec");
        _order = store.HasModel("order") ? store.Session("order") : null;
        _orient = store.HasModel("orient") ? store.Session("orient") : null;
        using JsonDocument meta = JsonDocument.Parse(store.ReadText("meta.json"));
        Vocab = Vocab.FromJson(meta.RootElement.GetProperty("vocab").GetString()!);
        if (store.HasModel("script") && store.HasModel("rec_cjk"))
        {
            _script = store.Session("script");
            _recCjk = store.Session("rec_cjk");
            VocabCjk = Vocab.FromJson(meta.RootElement.GetProperty("vocab_cjk").GetString()!);
        }

        if (Options.Math)
        {
            MathReader = MathReader.Load(store) ?? throw new ModelException("Math=true needs the math models.");
        }

        LongSide = meta.RootElement.GetProperty("det_long_side").GetInt32();
        Height = meta.RootElement.GetProperty("rec_height").GetInt32();
        _spaceId = Vocab.IndexOf(" ");
    }

    /// <summary>The options this engine was built with.</summary>
    public EngineOptions Options { get; }

    /// <summary>Vocabulary of the main recognizer.</summary>
    public Vocab Vocab { get; }

    /// <summary>Vocabulary of the CJK recognizer, when present.</summary>
    public Vocab? VocabCjk { get; }

    /// <summary>The math reader, when <see cref="EngineOptions.Math"/> is on.</summary>
    public MathReader? MathReader { get; }

    /// <summary>The detector's long side.</summary>
    public int LongSide { get; }

    /// <summary>The recognizer's input height.</summary>
    public int Height { get; }

    // ------------------------------------------------------------------ orientation and deskew

    /// <summary>
    /// <c>estimate_skew</c>: degrees (counter-clockwise) that straighten the text rows — the
    /// angle whose horizontal ink profile changes most sharply row to row, on an 800-px thumbnail,
    /// searched in 0.5° then 0.1° steps.
    /// </summary>
    public static double EstimateSkew(GrayImage gray, int side = 800)
    {
        int h = gray.Height, w = gray.Width;
        double s = (double)side / Math.Max(h, w);
        int sw = (int)Math.Max(1, Py.Round(w * s)), sh = (int)Math.Max(1, Py.Round(h * s));
        GrayImage small = CvResize.Resize(gray, sw, sh, Interpolation.Area);
        var ink = new float[sw * sh];
        for (int i = 0; i < ink.Length; i++)
        {
            ink[i] = (255f - small.Pixels[i]) / 255f;
        }

        if (Py.Mean(ink) < 0.003)
        {
            return 0.0;
        }

        double cx = sw / 2.0, cy = sh / 2.0;

        // sharpness(a): the warped page's row sums (numpy's pairwise float32 sums), then the
        // pairwise sum of their squared differences. Each row is reduced as soon as it is warped,
        // and the angles of one search run in parallel; the first best angle wins, as max() does.
        double Sharpness(double a)
        {
            float[] m = CvWarp.InverseFloat(CvWarp.RotationMatrix(cx, cy, a));
            var row = new float[sw];
            var rows = new float[sh];
            for (int y = 0; y < sh; y++)
            {
                CvWarp.WarpConstantZeroRow(ink, sw, sh, m, y, row);
                rows[y] = Py.PairwiseSum(row);
            }

            var diffs = new float[Math.Max(0, sh - 1)];
            for (int y = 0; y + 1 < sh; y++)
            {
                float d = rows[y + 1] - rows[y];
                diffs[y] = d * d;
            }

            return Py.PairwiseSum(diffs);
        }

        double Search(IEnumerable<double> angles)
        {
            double[] a = angles.ToArray();
            var values = new double[a.Length];
            Parallel.For(0, a.Length, Core.Parallelism.Options, i => values[i] = Sharpness(a[i]));
            int best = 0;
            for (int i = 1; i < a.Length; i++)
            {
                if (values[i] > values[best])
                {
                    best = i;
                }
            }

            return a[best];
        }

        double coarse = Search(Arange(-DeskewMax, DeskewMax + 1e-6, 0.5));
        return Search(Arange(coarse - 0.5, coarse + 0.5 + 1e-6, 0.1));
    }

    /// <summary><c>np.arange(start, stop, step)</c>.</summary>
    private static IEnumerable<double> Arange(double start, double stop, double step)
    {
        int n = (int)Math.Ceiling((stop - start) / step);
        for (int i = 0; i < n; i++)
        {
            yield return start + (i * step);
        }
    }

    /// <summary><c>rotate_page</c>: the page rotated counter-clockwise about its centre, same size, edges replicated.</summary>
    public static GrayImage RotatePage(GrayImage gray, double degrees) =>
        CvWarp.WarpReplicate(gray, CvWarp.RotationMatrix(gray.Width / 2.0, gray.Height / 2.0, degrees));

    /// <summary>
    /// <c>unrotate_box</c>: a box of the page <see cref="RotatePage"/> made, mapped back to the page
    /// before it — the bounding box of the four mapped corners, clipped, rounded to 0.1 px.
    /// </summary>
    public static double[] UnrotateBox(double[] box, double degrees, int w, int h)
    {
        double[] m = CvWarp.RotationMatrix(w / 2.0, h / 2.0, -degrees);
        double[] xs = [box[0], box[2], box[0], box[2]];
        double[] ys = [box[1], box[1], box[3], box[3]];
        double minX = double.PositiveInfinity, minY = double.PositiveInfinity, maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        for (int i = 0; i < 4; i++)
        {
            double px = (m[0] * xs[i]) + (m[1] * ys[i]) + m[2];
            double py = (m[3] * xs[i]) + (m[4] * ys[i]) + m[5];
            minX = Math.Min(minX, px);
            maxX = Math.Max(maxX, px);
            minY = Math.Min(minY, py);
            maxY = Math.Max(maxY, py);
        }

        return
        [
            Py.Round(Math.Clamp(minX, 0, w), 1),
            Py.Round(Math.Clamp(minY, 0, h), 1),
            Py.Round(Math.Clamp(maxX, 0, w), 1),
            Py.Round(Math.Clamp(maxY, 0, h), 1),
        ];
    }

    /// <summary><c>orient_input</c>: a <c>[side, side]</c> ink map, long side scaled to <paramref name="side"/>, padded.</summary>
    public static float[] OrientInput(GrayImage gray, int side = OrientSide)
    {
        int h = gray.Height, w = gray.Width;
        double s = (double)side / Math.Max(h, w);
        int sw = (int)Math.Max(1, Py.Round(w * s)), sh = (int)Math.Max(1, Py.Round(h * s));
        GrayImage small = CvResize.Resize(gray, sw, sh, Interpolation.Area);
        var result = new float[side * side];
        for (int y = 0; y < Math.Min(sh, side); y++)
        {
            for (int x = 0; x < Math.Min(sw, side); x++)
            {
                result[(y * side) + x] = 1f - (small.Pixels[(y * sw) + x] / 255f);
            }
        }

        return result;
    }

    /// <summary>
    /// <c>orientation</c>: quarter turns (counter-clockwise) the page content is turned by; 0
    /// without the model or when no turn is more likely than <see cref="OrientMin"/>.
    /// </summary>
    public int Orientation(GrayImage gray)
    {
        if (_orient is null)
        {
            return 0;
        }

        Tensor x = Tensor.FromFloats(OrientInput(gray), 1, 1, OrientSide, OrientSide);
        Tensor[] outs = _orient.Run(new Dictionary<string, Tensor> { ["x"] = x });
        float[] p = outs[0].ToFloatArray();
        outs[0].Release();
        int k = Py.ArgMax(p);
        return p[k] > OrientMin ? k : 0;
    }

    /// <summary><c>upright</c>: the page turned upright and straightened, with the quarter turns and skew undone.</summary>
    public (GrayImage Page, int QuarterTurns, double Skew) Upright(GrayImage gray, bool turn = true)
    {
        int k = turn ? Orientation(gray) : 0;
        if (k != 0)
        {
            gray = gray.RotateClockwise(k);
        }

        double a = EstimateSkew(gray);
        if (Math.Abs(a) >= DeskewMin)
        {
            gray = RotatePage(gray, a);
        }
        else
        {
            a = 0.0;
        }

        return (gray, k, a);
    }

    // ------------------------------------------------------------------ detection

    private List<LineBox> DetectAt(GrayImage gray, int longSide, out double scale)
    {
        (float[] ink, int h, int w, double s) = Detection.PageToInput(gray, longSide);
        scale = s;
        Tensor x = Tensor.FromFloats(ink, 1, 1, h, w);
        Tensor[] outs = _det.Run(new Dictionary<string, Tensor> { ["x"] = x });
        Tensor o = outs[0];
        List<LineBox> boxes = Detection.Decode(o.Floats, o.Shape[2], o.Shape[3]);
        o.Release();
        return boxes;
    }

    /// <summary><c>detect</c>: line boxes in page pixels, with an adaptive second pass for small text.</summary>
    public List<LineBox> Detect(GrayImage gray)
    {
        List<LineBox> boxes = DetectAt(gray, LongSide, out double s);
        if (Options.Adaptive && boxes.Count >= 8)
        {
            double medH = Py.Median(boxes.Select(b => b.Y1 - b.Y0).ToArray());
            if (medH < AdaptiveMinH)
            {
                int ls = (int)Math.Min(AdaptiveMaxSide, LongSide * AdaptiveTargetH / Math.Max(medH, 1.0)) / 32 * 32;
                if (ls > LongSide)
                {
                    boxes = DetectAt(gray, ls, out s);
                }
            }
        }

        float fs = (float)s;
        for (int i = 0; i < boxes.Count; i++)
        {
            LineBox b = boxes[i];
            b.X0 = Math.Clamp(b.X0 / fs, 0, gray.Width);
            b.Y0 = Math.Clamp(b.Y0 / fs, 0, gray.Height);
            b.X1 = Math.Clamp(b.X1 / fs, 0, gray.Width);
            b.Y1 = Math.Clamp(b.Y1 / fs, 0, gray.Height);
            boxes[i] = b;
        }

        return boxes;
    }

    // ------------------------------------------------------------------ recognition

    /// <summary>
    /// <c>recognize</c>: (text, confidence, tokens) per line, tokens carrying the page x of their
    /// CTC peak. Lines with Chinese characters go to the CJK recognizer; with the math reader on,
    /// confident math readings replace their lines. A line under <see cref="MinConf"/> comes back empty.
    /// </summary>
    public LineReading[] Recognize(GrayImage gray, IReadOnlyList<LineBox> boxes, int batch = 64)
    {
        var crops = boxes.Select(b => LineCrops.Crop(gray, b, Height)).ToList();
        var result = new LineReading[crops.Count];
        Array.Fill(result, LineReading.Empty);
        var groups = new List<(InferenceSession Session, Vocab Vocab, List<int> Lines, bool AsChars)>
        {
            (_rec, Vocab, Enumerable.Range(0, crops.Count).ToList(), false),
        };
        if (_script is not null && crops.Count > 0)
        {
            bool[] hanzi = HanziLines(crops, batch);
            groups =
            [
                (_rec, Vocab, Enumerable.Range(0, crops.Count).Where(j => !hanzi[j]).ToList(), false),
                (_recCjk!, VocabCjk!, Enumerable.Range(0, crops.Count).Where(j => hanzi[j]).ToList(), true),
            ];
        }

        foreach (var (session, vocab, sel, asChars) in groups)
        {
            foreach (List<int> part in LineCrops.WidthBatches(sel.Select(j => crops[j].Width).ToList(), batch))
            {
                List<int> idx = part.Select(i => sel[i]).ToList();
                (Tensor x, Tensor widths) = LineCrops.PadBatch(idx.Select(j => crops[j]).ToList());
                int padded = x.Shape[3];
                Tensor[] outs = session.Run(new Dictionary<string, Tensor> { ["x"] = x, ["widths"] = widths });
                x.Release();
                widths.Release();
                Tensor best = outs[0], pmax = outs[1], lengths = outs[2];
                int steps = best.Shape[1];
                double stride = (double)padded / Math.Max(steps, 1);
                for (int k = 0; k < idx.Count; k++)
                {
                    int j = idx[k];
                    int n = (int)lengths.Longs[k];
                    ReadOnlySpan<long> seq = best.Longs.Slice(k * steps, n);
                    ReadOnlySpan<float> pm = pmax.Floats.Slice(k * steps, n);
                    var keptIds = new List<int>();
                    var keptP = new List<float>();
                    var toks = new List<Token>();
                    LineBox box = boxes[j];
                    double x0 = box.X0, y0 = box.Y0, x1 = box.X1, y1 = box.Y1;
                    double h = Math.Max(1.0, y1 - y0);
                    double ix0 = Math.Floor(x0 - (LineCrops.PadX * h));
                    double cw = Math.Ceiling(x1 + (LineCrops.PadX * h)) - ix0;
                    double f = cw / crops[j].Width;
                    for (int t = 0; t < n; t++)
                    {
                        if (seq[t] == 0 || (t > 0 && seq[t] == seq[t - 1]))
                        {
                            continue;
                        }

                        int id = (int)seq[t];
                        double xs = ix0 + ((t + 0.5) * stride * f);
                        keptIds.Add(id);
                        keptP.Add(pm[t]);
                        toks.Add(asChars ? new Token(-1, vocab.Tokens[id - 1], xs) : new Token(id, null, xs));
                    }

                    double conf = keptP.Count > 0 ? Py.Mean(keptP.ToArray()) : 0.0;
                    result[j] = conf >= MinConf
                        ? new LineReading(vocab.Decode(keptIds), conf, toks)
                        : new LineReading("", conf, []);
                }

                foreach (Tensor t in outs)
                {
                    t.Release();
                }
            }
        }

        if (MathReader is not null && crops.Count > 0)
        {
            bool[] flags = MathReader.Flags(crops);
            var math = Enumerable.Range(0, crops.Count).Where(j => flags[j]).ToList();
            if (math.Count > 0)
            {
                var readings = MathReader.Read(gray, math.Select(j => boxes[j]).ToList());
                for (int m = 0; m < math.Count; m++)
                {
                    (string text, double conf) = readings[m];
                    if (text.Contains('$', StringComparison.Ordinal) && conf >= MathReader.MathMinConf)
                    {
                        int j = math[m];
                        double xc = (double)(boxes[j].X0 + boxes[j].X1) / 2;
                        result[j] = new LineReading(text, 1.0, [new Token(-1, text, xc)]);
                    }
                }
            }
        }

        return result;
    }

    /// <summary><c>hanzi_lines</c>: per line crop, does it hold Chinese characters?</summary>
    public bool[] HanziLines(IReadOnlyList<LineCrop> crops, int batch = 64)
    {
        var result = new bool[crops.Count];
        if (_script is null)
        {
            return result;
        }

        foreach (List<int> idx in LineCrops.WidthBatches(crops.Select(c => c.Width).ToList(), batch))
        {
            (Tensor x, Tensor widths) = LineCrops.PadBatch(idx.Select(j => crops[j]).ToList());
            Tensor[] outs = _script.Run(new Dictionary<string, Tensor> { ["x"] = x, ["widths"] = widths });
            for (int k = 0; k < idx.Count; k++)
            {
                result[idx[k]] = outs[0].Floats[k] > 0.5f;
            }

            outs[0].Release();
            x.Release();
            widths.Release();
        }

        return result;
    }

    // ------------------------------------------------------------------ line splitting

    private bool IsSpace(Token t) => t.IsText ? t.Text == " " : t.Id == _spaceId;

    /// <summary><c>_decode</c>: text of tokens (vocabulary ids, or characters from a text layer).</summary>
    public string DecodeTokens(IReadOnlyList<Token> toks)
    {
        if (toks.Count > 0 && toks[0].IsText)
        {
            return Py.CollapseSpaces(string.Concat(toks.Select(t => t.Text)));
        }

        return Vocab.Decode(toks.Select(t => t.Id).ToList());
    }

    /// <summary>
    /// <c>gutter_cuts</c>: per line, the x of gaps that sit on a column gutter — a gap over
    /// <see cref="GutterGap"/> line heights where at least <see cref="GutterEdges"/> other lines end
    /// at its left side or start at its right side.
    /// </summary>
    public List<List<double>> GutterCuts(IReadOnlyList<LineBox> boxes, IReadOnlyList<LineReading> rec)
    {
        var cuts = new List<List<double>>(boxes.Count);
        for (int i = 0; i < boxes.Count; i++)
        {
            LineBox b = boxes[i];
            double h = Math.Max(1.0, (double)(b.Y1 - b.Y0));
            var xs = rec[i].Tokens.Where(t => !IsSpace(t)).Select(t => t.X).ToList();
            var mine = new List<double>();
            float half = (float)(0.5 * h);
            for (int k = 0; k + 1 < xs.Count; k++)
            {
                double a = xs[k], c = xs[k + 1];
                if (c - a <= GutterGap * h)
                {
                    continue;
                }

                float gl = (float)(a + (0.35 * h)), gr = (float)(c - (0.35 * h));
                int near = 0;
                for (int j = 0; j < boxes.Count; j++)
                {
                    if (j != i && (MathF.Abs(boxes[j].X1 - gl) < half || MathF.Abs(boxes[j].X0 - gr) < half))
                    {
                        near++;
                    }
                }

                if (near >= GutterEdges)
                {
                    mine.Add((a + c) / 2);
                }
            }

            cuts.Add(mine);
        }

        return cuts;
    }

    /// <summary>
    /// <c>split_line</c>: cuts a line where neighbouring characters are more than
    /// <paramref name="gap"/> line heights apart (or at the given gutter positions).
    /// </summary>
    public List<(LineBox Box, string Text, List<Token> Tokens)> SplitLine(LineBox box, IReadOnlyList<Token> toks, double gap, IReadOnlyList<double>? at = null)
    {
        double h = Math.Max(1.0, (double)(box.Y1 - box.Y0));
        var segs = new List<List<Token>> { new() };
        double? prev = null;
        double[] cuts = (at ?? []).Order().ToArray();
        foreach (Token tk in toks)
        {
            double x = tk.X;
            if (!IsSpace(tk))
            {
                bool cut = prev is double p && (x - p > gap * h || cuts.Any(c => p < c && c < x));
                if (cut)
                {
                    segs.Add([]);
                }

                prev = x;
            }

            segs[^1].Add(tk);
        }

        if (segs.Count == 1)
        {
            return [(new LineBox(box.X0, box.Y0, box.X1, box.Y1, box.Score), DecodeTokens(toks), [.. toks])];
        }

        var result = new List<(LineBox, string, List<Token>)>();
        foreach (List<Token> sg in segs)
        {
            string text = Py.Strip(DecodeTokens(sg));
            var xs = sg.Where(t => !IsSpace(t)).Select(t => t.X).ToList();
            if (text.Length > 0 && xs.Count > 0)
            {
                var bx = new LineBox(
                    (float)Math.Max(box.X0, xs.Min() - (0.35 * h)),
                    box.Y0,
                    (float)Math.Min(box.X1, xs.Max() + (0.35 * h)),
                    box.Y1,
                    box.Score);
                result.Add((bx, text, sg));
            }
        }

        return result;
    }

    // ------------------------------------------------------------------ order and assembly

    /// <summary>The layout heads' outputs for one page.</summary>
    public sealed record LayoutHeads(float[] Cls, float[] Same, float[]? Tab);

    /// <summary>
    /// <c>reading_order</c>: the order of the lines, the layout heads' logits when the order
    /// model has them, and the pairwise order logits.
    /// </summary>
    public (List<int> Order, LayoutHeads? Layout, float[]? Logits) ReadingOrderOf(IReadOnlyList<LineBox> boxes, IReadOnlyList<string> texts, int pageW, int pageH)
    {
        int n = boxes.Count;
        if (n < 2 || _order is null)
        {
            return (n > 0 ? ReadingOrder.XyCut(boxes) : [], null, null);
        }

        float[] feats = ReadingOrder.LineFeatures(boxes, pageW, pageH, texts);
        var mask = new bool[n];
        Array.Fill(mask, true);
        Tensor[] outs = _order.Run(new Dictionary<string, Tensor>
        {
            ["feats"] = Tensor.FromFloats(feats, 1, n, ReadingOrder.Features),
            ["mask"] = Tensor.FromBools(mask, 1, n),
        });
        float[] a = outs[0].ToFloatArray();
        try
        {
            if (outs.Length >= 3)
            {
                float[] cls = outs[1].ToFloatArray();
                float[] same = outs[2].ToFloatArray();
                float[]? tab = outs.Length > 3 && Options.TableHeads ? outs[3].ToFloatArray() : null;
                List<int> order = ReadingOrder.DecodeBlocksOrder(a, same, boxes, cls, ReadingOrder.BlockClasses.Length);
                return (order, new LayoutHeads(cls, same, tab), a);
            }

            return (ReadingOrder.DecodeOrder(a, n), null, a);
        }
        finally
        {
            foreach (Tensor t in outs)
            {
                t.Release();
            }
        }
    }

    /// <summary>OCR of a page image (<c>OrtOCR.__call__</c>).</summary>
    /// <param name="gray">The page.</param>
    /// <param name="turn">Whether to ask the orientation model (false: the page is known to be upright).</param>
    public Page Process(GrayImage gray, bool turn = true)
    {
        var t = new Dictionary<string, double>();
        long t0 = Stopwatch.GetTimestamp();
        (gray, int k, double skew) = Upright(gray, turn);
        t["orient_ms"] = Ms(t0);
        t0 = Stopwatch.GetTimestamp();
        List<LineBox> boxes = Detect(gray);
        t["det_ms"] = Ms(t0);
        t0 = Stopwatch.GetTimestamp();
        LineReading[] rec = Recognize(gray, boxes);
        t["rec_ms"] = Ms(t0);
        Page page = Assemble(gray.Width, gray.Height, boxes, rec, t);
        page.Rotation = 90 * k;
        page.Skew = Py.Round(skew, 2);
        if (skew != 0)
        {
            foreach (Line ln in page.Lines)
            {
                ln.Box = UnrotateBox(ln.Box, skew, gray.Width, gray.Height);
            }

            foreach (Block b in page.Blocks)
            {
                b.BBox = UnrotateBox(b.BBox, skew, gray.Width, gray.Height);
            }
        }

        return page;
    }

    internal static double Ms(long start) => Stopwatch.GetElapsedTime(start).TotalMilliseconds;

    /// <summary>
    /// <c>assemble</c>: lines split at wide gaps, reading order, layout blocks and markdown. With
    /// <paramref name="contentOrder"/> and text-layer tokens on every line, the PDF content order
    /// is kept when it is coherent.
    /// </summary>
    public Page Assemble(int pageW, int pageH, IReadOnlyList<LineBox> boxesIn, IReadOnlyList<LineReading> recIn, Dictionary<string, double> t, bool contentOrder = false)
    {
        long t0 = Stopwatch.GetTimestamp();
        var boxes = new List<LineBox>();
        var rec = new List<LineReading>();
        List<List<double>> cuts = boxesIn.Count > 0 ? GutterCuts(boxesIn, recIn) : [];
        for (int i = 0; i < boxesIn.Count; i++)
        {
            foreach (var (bx, tx, tk) in SplitLine(boxesIn[i], recIn[i].Tokens, SplitGap, cuts[i]))
            {
                boxes.Add(bx);
                rec.Add(new LineReading(tx, recIn[i].Conf, tk));
            }
        }

        var texts = rec.Select(r => r.Text).ToList();
        (List<int> order, LayoutHeads? layout, float[]? logits) = ReadingOrderOf(boxes, texts, pageW, pageH);
        if (layout?.Tab is not null)
        {
            // The table heads learned one line per cell, but the detector can join cells: split
            // table lines into cells, then run the order model again.
            int tableCls = Array.IndexOf(ReadingOrder.BlockClasses, "Table");
            int classes = ReadingOrder.BlockClasses.Length;
            var isTable = Enumerable.Range(0, boxes.Count).Select(i => Py.ArgMax(layout.Cls.AsSpan(i * classes, classes)) == tableCls).ToArray();
            if (isTable.Any(x => x))
            {
                var nb = new List<LineBox>();
                var nrec = new List<LineReading>();
                for (int i = 0; i < boxes.Count; i++)
                {
                    if (isTable[i])
                    {
                        foreach (var (bx, tx, tk) in SplitLine(boxes[i], rec[i].Tokens, CellGap))
                        {
                            nb.Add(bx);
                            nrec.Add(new LineReading(tx, rec[i].Conf, tk));
                        }
                    }
                    else
                    {
                        nb.Add(boxes[i]);
                        nrec.Add(rec[i]);
                    }
                }

                if (nb.Count != boxes.Count)
                {
                    boxes = nb;
                    rec = nrec;
                    texts = rec.Select(r => r.Text).ToList();
                    (order, layout, logits) = ReadingOrderOf(boxes, texts, pageW, pageH);
                }
            }
        }

        string note = "order=model";
        if (contentOrder && boxes.Count > 1)
        {
            var keys = rec.Select(r => r.Tokens.Where(tk => tk.HasContentIndex).Select(tk => (int?)tk.ContentIndex).Min()).ToList();
            if (keys.All(k => k is not null))
            {
                List<int> pdf = Enumerable.Range(0, keys.Count).OrderBy(i => keys[i]!.Value).ToList();
                float[]? sub = null;
                if (logits is not null)
                {
                    int n = boxes.Count;
                    sub = new float[n * n];
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            sub[(i * n) + j] = logits[(pdf[i] * n) + pdf[j]];
                        }
                    }
                }

                if (TextLayer.PdfOrderCoherent(pdf.Select(i => boxes[i]).ToList(), sub))
                {
                    order = pdf;
                    note = "order=pdf";
                }
            }
        }

        t["order_ms"] = Ms(t0);
        t["total_ms"] = t.Where(kv => kv.Key != "total_ms").Sum(kv => kv.Value);
        var lines = order.Select(i => new Line(
            [Py.Round(boxes[i].X0, 1), Py.Round(boxes[i].Y0, 1), Py.Round(boxes[i].X1, 1), Py.Round(boxes[i].Y1, 1)],
            rec[i].Text,
            Py.Round(rec[i].Conf, 4))).ToList();
        string? md = null;
        var blocks = new List<Block>();
        if (layout is not null)
        {
            blocks = Layout.PageBlocks(boxes, texts, order, layout.Cls, layout.Same)
                .Select(b => new Block { Label = b.Label, BBox = b.BBox, Text = b.Text }).ToList();
            var cells = new Dictionary<int, List<(LineBox, string)>>();
            for (int i = 0; i < boxes.Count; i++)
            {
                cells[i] = SplitLine(boxes[i], rec[i].Tokens, CellGap).Select(p => (p.Box, p.Text)).ToList();
            }

            md = Layout.PageMarkdown(boxes, texts, order, layout.Cls, layout.Same, Options.Furniture, cells, layout.Tab);
        }
        else if (boxes.Count < 2)
        {
            md = string.Join("\n", texts);
        }

        return new Page { Lines = lines, Timings = t, Markdown = md, Note = note, Blocks = blocks };
    }
}
