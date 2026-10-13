namespace IndraOcr.Vlm.Surya;

/// <summary>
/// <c>surya/inference/prompts.py</c> and <c>surya/layout/label.py</c>: the prompts (the model's
/// training-time contract — not to be paraphrased), the label vocabulary and its canonical names.
/// </summary>
public static class SuryaPrompts
{
    /// <summary><c>LAYOUT_PROMPT</c>.</summary>
    public const string Layout =
        "Output the layout of this image as JSON. Each entry is a dict with "
        + "\"label\", \"bbox\", and \"count\" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.";

    /// <summary><c>BLOCK_PROMPT</c>.</summary>
    public const string Block = "OCR this block image to HTML.";

    /// <summary><c>TABLE_REC_PROMPT</c>.</summary>
    public const string TableRec =
        "Output the table rows then columns as JSON. Each entry is a dict with "
        + "\"label\" (\"Row\" or \"Col\") and \"bbox\" (x0 y0 x1 y1, normalized 0-1000).";

    /// <summary><c>HIGH_ACCURACY_BBOX_PROMPT</c>.</summary>
    public const string HighAccuracyBbox =
        "OCR this image to HTML. Each block is a div with data-label and data-bbox "
        + "(x0 y0 x1 y1, normalized 0-1000).";

    /// <summary><c>LAYOUT_LABEL_SET</c>.</summary>
    public static readonly string[] LayoutLabels =
    [
        "Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer", "Image", "Section-Header",
        "Table", "Text", "Complex-Block", "Code-Block", "Form", "Table-Of-Contents", "Figure", "Chemical-Block", "Diagram",
        "Bibliography", "Blank-Page",
    ];

    /// <summary><c>SKIP_OCR_LABELS</c>: block labels not sent for OCR.</summary>
    public static readonly HashSet<string> SkipOcrLabels = ["Figure", "Image", "Diagram", "Blank-Page"];

    /// <summary><c>LAYOUT_PRED_RELABEL</c>: model labels to Surya's public names.</summary>
    public static readonly Dictionary<string, string> Relabel = new()
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

    /// <summary><c>TEXT_LABELS</c>: canonical labels eligible for blank-region filtering.</summary>
    public static readonly HashSet<string> TextLabels =
        ["Text", "SectionHeader", "PageHeader", "PageFooter", "Caption", "Footnote", "Code", "Bibliography"];

    /// <summary><c>SKIP_CANON_LABELS</c>.</summary>
    public static readonly HashSet<string> SkipCanonicalLabels = [.. SkipOcrLabels.Select(Canonical)];

    /// <summary><c>LAYOUT_PRED_RELABEL.get(label, label)</c>.</summary>
    public static string Canonical(string label) => Relabel.TryGetValue(label, out string? c) ? c : label;

    /// <summary><c>image_token_budget</c>: <c>count + 100</c> clamped to <c>[floor, ceiling]</c>.</summary>
    public static int ImageTokenBudget(int blockCount, int ceiling = 4096, int floor = 64) =>
        Math.Min(Math.Max(blockCount + 100, floor), ceiling);
}
