using NanoOcr.Pipeline;

namespace NanoOcr.Routing;

/// <summary>What kind of page a <see cref="PageClassifier"/> takes a page for.</summary>
public enum PageCategory
{
    /// <summary>Printed text NanoOCR reads confidently: paragraphs, headings, lists, columns.</summary>
    PrintedText,

    /// <summary>Most of the page is text NanoOCR cannot read, written large and in long strokes.</summary>
    Handwritten,

    /// <summary>Printed, but most of its lines sit in tables or forms.</summary>
    TableOrComplex,

    /// <summary>The signals disagree or sit near a threshold; also pages with almost no text.</summary>
    Mixed,
}

/// <summary>
/// What a page's NanoOCR reading says about it; every figure is a share or relative to the page,
/// so it does not depend on the resolution.
/// </summary>
/// <param name="Lines">Text lines found.</param>
/// <param name="Confidence">Mean line confidence, weighted by characters read.</param>
/// <param name="LowConfidenceShare">Share of lines below <see cref="PageClassifierOptions.LowConfidence"/>.</param>
/// <param name="TableShare">Share of lines in <c>Table</c> or <c>Form</c> blocks.</param>
/// <param name="Tables">Tables of at least two rows and two columns.</param>
/// <param name="LineHeight">Median line height over the page height.</param>
/// <param name="WideLineShare">Share of lines wider than <see cref="PageClassifierOptions.WideLine"/> of the page width.</param>
/// <param name="TextLayer">The page was read from a PDF text layer (born digital).</param>
public sealed record PageSignals(
    int Lines,
    double Confidence,
    double LowConfidenceShare,
    double TableShare,
    int Tables,
    double LineHeight,
    double WideLineShare,
    bool TextLayer);

/// <summary>A page's category, the signals behind it, and the rule that decided.</summary>
/// <param name="Category">The category.</param>
/// <param name="Signals">The measured signals.</param>
/// <param name="Reason">The deciding rule, in words.</param>
public sealed record PageClassification(PageCategory Category, PageSignals Signals, string Reason);

/// <summary>The thresholds of <see cref="PageClassifier"/>.</summary>
public sealed record PageClassifierOptions
{
    /// <summary>The calibrated defaults (see <see cref="PageClassifier"/>).</summary>
    public static PageClassifierOptions Default { get; } = new();

    /// <summary>Fewer lines than this: <see cref="PageCategory.Mixed"/>.</summary>
    public int MinLines { get; init; } = 3;

    /// <summary>A line below this confidence counts as unread.</summary>
    public double LowConfidence { get; init; } = 0.6;

    /// <summary>A page is mostly unread when this share of its lines is…</summary>
    public double UnreadShare { get; init; } = 0.45;

    /// <summary>…and its confidence is below this.</summary>
    public double UnreadConfidence { get; init; } = 0.75;

    /// <summary>An unread page is handwritten when its median line is this tall (of the page height)…</summary>
    public double HandwritingLineHeight { get; init; } = 0.012;

    /// <summary>…or this share of its lines are wide.</summary>
    public double HandwritingWideShare { get; init; } = 0.08;

    /// <summary>A line wider than this share of the page width is wide.</summary>
    public double WideLine { get; init; } = 0.25;

    /// <summary>This share of lines in tables or forms: <see cref="PageCategory.TableOrComplex"/>.</summary>
    public double TableShare { get; init; } = 0.25;

    /// <summary>Printed text needs at least this confidence…</summary>
    public double PrintedConfidence { get; init; } = 0.85;

    /// <summary>…and at most this share of unread lines.</summary>
    public double PrintedUnreadShare { get; init; } = 0.35;
}

/// <summary>
/// Sorts a page into a <see cref="PageCategory"/> from its NanoOCR reading alone — no extra model,
/// so classifying costs nothing beyond reading the page.
/// </summary>
/// <remarks>
/// <para>
/// The recognizer is trained on print, so handwriting comes back mostly unread or at low
/// confidence; what tells it from a degraded print or a fax is its size — tall lines, and long
/// connected strokes the detector takes for wide lines. Tables and forms are the order model's
/// <c>Table</c> and <c>Form</c> classes. A page between the thresholds is
/// <see cref="PageCategory.Mixed"/>, which a router sends to its fallback.
/// </para>
/// <para>
/// Calibrated on RVL-CDIP (160 pages, ten per class) and checked on 400 other pages
/// (<c>jordyvl/rvl_cdip_100_examples_per_class</c>, test split), taking <c>handwritten</c> as
/// handwritten; <c>form</c>, <c>advertisement</c>, <c>specification</c>, <c>budget</c>,
/// <c>invoice</c> and <c>questionnaire</c> as tables or complex; <c>file folder</c> as mixed; the
/// rest as printed. On the 400: 80% of handwritten pages found (20 of 25), 56% of handwriting
/// calls right (mostly adverts in large type otherwise); 83% of printed calls printed (116 of
/// 140, the rest RVL forms and invoices with little ruling); 65% of table and form pages called
/// tables (98 of 150), while 57 printed pages are too — memos and resumes whose key-value
/// headers the order model reads as tables. No handwritten page is ever called printed — the
/// error that would hand a page NanoOCR cannot read to NanoOCR. RVL-CDIP is 1990s fax-quality
/// scans, the hard case; a born-digital page (a PDF text layer) is never handwritten.
/// A trained classifier can replace this one in <see cref="PageRouter{TOutput}"/>.
/// </para>
/// </remarks>
public static class PageClassifier
{
    /// <summary>Classifies a page from its reading.</summary>
    /// <param name="result">The page's NanoOCR result.</param>
    /// <param name="options">Thresholds; <see cref="PageClassifierOptions.Default"/> when omitted.</param>
    public static PageClassification Classify(OcrResult result, PageClassifierOptions? options = null) =>
        Classify(result.Page, options);

    /// <summary>Classifies a page from its reading.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Thresholds; <see cref="PageClassifierOptions.Default"/> when omitted.</param>
    public static PageClassification Classify(Page page, PageClassifierOptions? options = null)
    {
        PageClassifierOptions o = options ?? PageClassifierOptions.Default;
        PageSignals s = Signals(page, o);
        if (page.Source == PageSources.Error)
        {
            return new(PageCategory.Mixed, s, "the page failed: " + page.Note);
        }

        if (s.Lines < o.MinLines)
        {
            return new(PageCategory.Mixed, s, $"{s.Lines} lines of text");
        }

        if (!s.TextLayer && s.LowConfidenceShare >= o.UnreadShare && s.Confidence < o.UnreadConfidence)
        {
            return s.LineHeight >= o.HandwritingLineHeight || s.WideLineShare >= o.HandwritingWideShare
                ? new(PageCategory.Handwritten, s, "mostly unread, in large or long strokes")
                : new(PageCategory.Mixed, s, "mostly unread, in small type: degraded print or noise");
        }

        if (s.TableShare >= o.TableShare)
        {
            return new(PageCategory.TableOrComplex, s, $"{s.TableShare:P0} of the lines in tables or forms");
        }

        if (s.Confidence < o.PrintedConfidence || s.LowConfidenceShare >= o.PrintedUnreadShare)
        {
            return new(PageCategory.Mixed, s, "printed, but read with low confidence");
        }

        return new(PageCategory.PrintedText, s, "printed text read confidently");
    }

    /// <summary>The signals <see cref="Classify(Page, PageClassifierOptions?)"/> decides on.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Thresholds (the low-confidence and wide-line limits).</param>
    public static PageSignals Signals(Page page, PageClassifierOptions? options = null)
    {
        PageClassifierOptions o = options ?? PageClassifierOptions.Default;
        List<Line> lines = page.Lines;
        bool textLayer = page.Source is PageSources.Text;
        int n = lines.Count;
        if (n == 0)
        {
            return new(0, 0, 0, 0, 0, 0, 0, textLayer);
        }

        double weighted = 0, weights = 0;
        int low = 0, wide = 0;
        var heights = new double[n];
        for (int i = 0; i < n; i++)
        {
            Line l = lines[i];
            double w = Math.Max(l.Text.Length, 1);
            weighted += l.Conf * w;
            weights += w;
            low += l.Conf < o.LowConfidence ? 1 : 0;
            wide += page.Width > 0 && l.Box[2] - l.Box[0] > o.WideLine * page.Width ? 1 : 0;
            heights[i] = l.Box[3] - l.Box[1];
        }

        Array.Sort(heights);
        double median = n % 2 == 1 ? heights[n / 2] : (heights[(n / 2) - 1] + heights[n / 2]) / 2;
        int inTables = page.Blocks.Where(b => b.Label is "Table" or "Form").Sum(b => b.Text.Split('\n').Length);
        int tables = page.Elements?.Count(e => e.Table is { Columns: >= 2, OwnRows: >= 2 }) ?? 0;
        return new(
            n,
            weighted / weights,
            (double)low / n,
            Math.Min(1.0, (double)inTables / n),
            tables,
            page.Height > 0 ? median / page.Height : 0,
            (double)wide / n,
            textLayer);
    }
}
