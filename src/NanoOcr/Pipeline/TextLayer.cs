using System.Globalization;
using System.Text;
using NanoOcr.Imaging;
using NanoOcr.Pdf;

namespace NanoOcr.Pipeline;

/// <summary>One text-layer run: its box (points, top-left origin), text, and characters with boxes in PDF order.</summary>
public sealed record TextRun(double[] BBox, string Text, List<(string Char, double[] Box)> Chars);

/// <summary>The runs of a page and the verdict on them.</summary>
public sealed record TextLayerResult(List<TextRun> Lines, bool Trusted, string Reason, Dictionary<string, double> Stats);

/// <summary>
/// Text-layer runs of a PDF page and the checks that decide whether to trust them (port of
/// <c>indra_ocr/textlayer.py</c>): invisible text marks an OCR layer, replacement / private-use /
/// control characters mark a broken font map, and a run with no ink under it marks hidden text.
/// </summary>
public static class TextLayer
{
    /// <summary>More invisible text than this is an OCR layer.</summary>
    public const double MaxInvisibleFrac = 0.05;

    /// <summary>More bad characters than this is a broken font map.</summary>
    public const double MaxBadCharFrac = 0.005;

    /// <summary>More runs without ink than this and the layer is not trusted.</summary>
    public const double MaxNoInkLineFrac = 0.03;

    /// <summary>Grey level below which a pixel counts as ink.</summary>
    public const int Dark = 160;

    /// <summary><see cref="TextLayerResult.Reason"/> of a layer whose codes are not text.</summary>
    public const string BrokenFontMap = "broken font map";

    /// <summary>The model must agree with this share of consecutive pairs for the PDF order to be kept.</summary>
    public const double MinModelAgreement = 0.85;

    /// <summary><c>bad_char_count</c>: replacement, private-use, surrogate, unassigned and control characters.</summary>
    public static int BadCharCount(string text)
    {
        int n = 0;
        foreach (Rune r in text.EnumerateRunes())
        {
            UnicodeCategory cat = Rune.GetUnicodeCategory(r);
            if (r.Value == 0xFFFD || cat is UnicodeCategory.PrivateUse or UnicodeCategory.Surrogate or UnicodeCategory.OtherNotAssigned
                || (cat == UnicodeCategory.Control && r.Value != '\t'))
            {
                n++;
            }
        }

        return n;
    }

    /// <summary><c>_device_affine</c>: the 2×3 map from PDF page space to the rendered frame (points, top-left origin).</summary>
    private static double[] DeviceAffine(PdfPage page)
    {
        const int S = 1000;
        int sw = (int)Py.Round(page.Width * S), sh = (int)Py.Round(page.Height * S);
        (double X, double Y) Dev(double x, double y)
        {
            Pdfium.FPDF_PageToDevice(page.Handle, 0, 0, sw, sh, 0, x, y, out int dx, out int dy);
            return (dx / (double)S, dy / (double)S);
        }

        var o = Dev(0, 0);
        var ex = Dev(1, 0);
        var ey = Dev(0, 1);
        return [ex.X - o.X, ey.X - o.X, o.X, ex.Y - o.Y, ey.Y - o.Y, o.Y];
    }

    /// <summary><c>extract_runs</c>: upright text runs of a page in content order, and character statistics.</summary>
    public static (List<TextRun> Runs, Dictionary<string, double> Stats) ExtractRuns(PdfPage page)
    {
        var chars = new List<string>();
        var raw = new List<(double L, double B, double R, double T)>();
        var breaks = new List<int>();
        int nChars = 0, nRot = 0, nBad = 0;
        double[] a;
        lock (Pdfium.Lock)
        {
            a = DeviceAffine(page);
            IntPtr tp = Pdfium.FPDFText_LoadPage(page.Handle);
            try
            {
                int n = Pdfium.FPDFText_CountChars(tp);
                var upright = new Dictionary<float, bool>();
                var bad = new Dictionary<string, int>(StringComparer.Ordinal);
                for (int i = 0; i < n; i++)
                {
                    uint u = Pdfium.FPDFText_GetUnicode(tp, i);
                    string c = u <= 0x10FFFF && !(u >= 0xD800 && u <= 0xDFFF) ? char.ConvertFromUtf32((int)u) : ((char)(u & 0xFFFF)).ToString();
                    if (c is "\r" or "\n")
                    {
                        breaks.Add(chars.Count);
                        continue;
                    }

                    if (c == "\x02")
                    {
                        c = "-";
                    }

                    if (!IsSpace(c))
                    {
                        nChars++;
                        if (!bad.TryGetValue(c, out int k))
                        {
                            k = bad[c] = BadCharCount(c);
                        }

                        nBad += k;
                        float angle = Pdfium.FPDFText_GetCharAngle(tp, i);
                        if (!upright.TryGetValue(angle, out bool up))
                        {
                            double cx = Math.Cos(angle), cy = Math.Sin(angle);
                            double d0 = (a[0] * cx) + (a[1] * cy);
                            double d1 = (a[3] * cx) + (a[4] * cy);
                            up = upright[angle] = !(d0 <= 0 || Math.Abs(d1) > 1e-3 * Math.Max(Math.Abs(d0), 1e-9));
                        }

                        if (!up)
                        {
                            nRot++;
                            breaks.Add(chars.Count);
                            continue;
                        }
                    }

                    chars.Add(c);
                    if (!Pdfium.FPDFText_GetCharBox(tp, i, out double l, out double r, out double b, out double t))
                    {
                        throw new PdfException("Failed to get charbox.");
                    }

                    raw.Add((l, b, r, t));
                }
            }
            finally
            {
                Pdfium.FPDFText_ClosePage(tp);
            }
        }

        var stats = new Dictionary<string, double> { ["chars"] = nChars, ["rotated"] = nRot, ["bad"] = nBad };
        if (chars.Count == 0)
        {
            return ([], stats);
        }

        double w = page.Width, h = page.Height;
        var boxes = new double[chars.Count][];
        var visible = new bool[chars.Count];
        int offPage = 0;
        for (int i = 0; i < chars.Count; i++)
        {
            var (l, b, r, t) = raw[i];
            double[] xs = [l, r, l, r];
            double[] ys = [b, b, t, t];
            double minX = double.PositiveInfinity, minY = double.PositiveInfinity, maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
            for (int k = 0; k < 4; k++)
            {
                double px = (a[0] * xs[k]) + (a[1] * ys[k]) + a[2];
                double py = (a[3] * xs[k]) + (a[4] * ys[k]) + a[5];
                minX = Math.Min(minX, px);
                maxX = Math.Max(maxX, px);
                minY = Math.Min(minY, py);
                maxY = Math.Max(maxY, py);
            }

            boxes[i] = [minX, minY, maxX, maxY];
            double ccx = (minX + maxX) / 2, ccy = (minY + maxY) / 2;
            visible[i] = ccx >= 0 && ccx <= w && ccy >= 0 && ccy <= h;
            offPage += visible[i] ? 0 : 1;
        }

        stats["off_page"] = offPage;
        var runs = new List<TextRun>();
        var cuts = breaks.Append(chars.Count).Distinct().Order().ToList();
        int start = 0;
        foreach (int end in cuts)
        {
            var seg = new List<(string, double[])>();
            for (int i = start; i < end; i++)
            {
                if (visible[i])
                {
                    seg.Add((chars[i], boxes[i]));
                }
            }

            start = end;
            var ink = seg.Where(s => !IsSpace(s.Item1)).Select(s => s.Item2).ToList();
            if (ink.Count == 0)
            {
                continue;
            }

            double[] box = [ink.Min(v => v[0]), ink.Min(v => v[1]), ink.Max(v => v[2]), ink.Max(v => v[3])];
            runs.Add(new TextRun(box, Py.CollapseSpaces(string.Concat(seg.Select(s => s.Item1))), seg));
        }

        return (runs, stats);
    }

    /// <summary>Python's <c>str.isspace()</c> on a one-character string.</summary>
    internal static bool IsSpace(string c) => c.Length > 0 && Py.CodePoints(c).All(Py.IsSpace);

    /// <summary><c>invisible_text_frac</c>: the share of text objects drawn invisibly (an OCR layer).</summary>
    public static double InvisibleTextFrac(PdfPage page)
    {
        int n = 0, inv = 0;

        void Walk(IntPtr parent, bool form, int level)
        {
            int count = form ? Pdfium.FPDFFormObj_CountObjects(parent) : Pdfium.FPDFPage_CountObjects(parent);
            for (int i = 0; i < count; i++)
            {
                IntPtr obj = form ? Pdfium.FPDFFormObj_GetObject(parent, (uint)i) : Pdfium.FPDFPage_GetObject(parent, i);
                if (obj == IntPtr.Zero)
                {
                    continue;
                }

                int type = Pdfium.FPDFPageObj_GetType(obj);
                if (type == Pdfium.FPDF_PAGEOBJ_TEXT)
                {
                    n++;
                    inv += Pdfium.FPDFTextObj_GetTextRenderMode(obj) == Pdfium.FPDF_TEXTRENDERMODE_INVISIBLE ? 1 : 0;
                }

                if (type == Pdfium.FPDF_PAGEOBJ_FORM && level < 4 - 1)
                {
                    Walk(obj, true, level + 1);
                }
            }
        }

        lock (Pdfium.Lock)
        {
            Walk(page.Handle, false, 0);
        }

        return n > 0 ? (double)inv / n : 0.0;
    }

    /// <summary><c>no_ink_frac</c>: the share of runs whose box holds no dark pixel in <paramref name="gray"/>.</summary>
    public static double NoInkFrac(GrayImage gray, IReadOnlyList<TextRun> runs, double scale)
    {
        int empty = 0;
        foreach (TextRun ln in runs)
        {
            double x0 = ln.BBox[0], y0 = ln.BBox[1], x1 = ln.BBox[2], y1 = ln.BBox[3];
            int r0 = Math.Max(0, (int)(y0 * scale)), r1 = Math.Min(gray.Height, (int)(y1 * scale) + 1);
            int c0 = Math.Max(0, (int)(x0 * scale)), c1 = Math.Min(gray.Width, (int)(x1 * scale) + 1);
            bool any = false;
            for (int y = r0; y < r1 && !any; y++)
            {
                for (int x = c0; x < c1; x++)
                {
                    if (gray.Pixels[(y * gray.Width) + x] < Dark)
                    {
                        any = true;
                        break;
                    }
                }
            }

            empty += any ? 0 : 1;
        }

        return (double)empty / Math.Max(runs.Count, 1);
    }

    /// <summary><c>read_text_layer</c>: the runs of a page and the verdict on them.</summary>
    /// <param name="page">The open page.</param>
    /// <param name="gray">A render at <paramref name="scale"/> pixels per point (low resolution is enough).</param>
    /// <param name="scale">Pixels per point of <paramref name="gray"/>.</param>
    public static TextLayerResult Read(PdfPage page, GrayImage gray, double scale)
    {
        double inv = InvisibleTextFrac(page);
        if (inv > MaxInvisibleFrac)
        {
            return new TextLayerResult([], false, $"invisible text {Percent(inv)} (OCR layer)", []);
        }

        (List<TextRun> runs, Dictionary<string, double> st) = ExtractRuns(page);
        if (runs.Count == 0)
        {
            return new TextLayerResult([], false, "no text layer", st);
        }

        if (st["bad"] > MaxBadCharFrac * Math.Max(st["chars"], 1))
        {
            return new TextLayerResult(runs, false, BrokenFontMap, st);
        }

        double empty = NoInkFrac(gray, runs, scale);
        st["no_ink"] = Py.Round(empty, 4);
        if (empty > MaxNoInkLineFrac)
        {
            return new TextLayerResult(runs, false, $"{Percent(empty)} of text runs have no ink", st);
        }

        return new TextLayerResult(runs, true, "", st);
    }

    /// <summary>Python's <c>f"{x:.0%}"</c>.</summary>
    private static string Percent(double x) => Py.Round(x * 100).ToString(CultureInfo.InvariantCulture) + "%";

    /// <summary><c>order_backtracks</c>: upward moves in content order that do not go to a column on the right.</summary>
    public static int OrderBacktracks(IReadOnlyList<LineBox> boxes)
    {
        if (boxes.Count < 3)
        {
            return 0;
        }

        double h = Py.Median(boxes.Select(b => b.Y1 - b.Y0).ToArray());
        int bt = 0;
        for (int i = 0; i + 1 < boxes.Count; i++)
        {
            LineBox a = boxes[i], b = boxes[i + 1];
            if (b.Y0 < a.Y0 - (float)(0.5 * h) && b.X0 < a.X1 - (float)h)
            {
                bt++;
            }
        }

        return bt;
    }

    /// <summary>
    /// <c>pdf_order_coherent</c>: is the content order a reading order? Few upward moves that are
    /// not column changes, and the order model agrees with most consecutive pairs.
    /// </summary>
    public static bool PdfOrderCoherent(IReadOnlyList<LineBox> boxes, float[]? pairLogits)
    {
        if (OrderBacktracks(boxes) > Math.Max(1, (int)(0.02 * boxes.Count)))
        {
            return false;
        }

        if (pairLogits is null || boxes.Count < 3)
        {
            return true;
        }

        int n = boxes.Count;
        int agree = 0;
        for (int i = 0; i + 1 < n; i++)
        {
            agree += pairLogits[(i * n) + i + 1] > 0 ? 1 : 0;
        }

        return (double)agree / (n - 1) >= MinModelAgreement;
    }
}
