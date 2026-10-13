using System.Text.Json.Serialization;

namespace IndraOcr.Pipeline;

/// <summary>
/// A recognized token: a vocabulary id (CTC recognizer) or a string (CJK recognizer, math reader,
/// PDF text layer), the page x of its centre, and — for text-layer tokens — its index in the PDF
/// content order.
/// </summary>
public readonly record struct Token(int Id, string? Text, double X, int ContentIndex = -1)
{
    /// <summary>Whether this token carries text rather than a vocabulary id.</summary>
    public bool IsText => Text is not null;

    /// <summary>Whether the token has a PDF content index.</summary>
    public bool HasContentIndex => ContentIndex >= 0;
}

/// <summary>One line's recognition: text, mean CTC peak probability, and tokens.</summary>
public sealed record LineReading(string Text, double Conf, IReadOnlyList<Token> Tokens)
{
    /// <summary>A line with no reading.</summary>
    public static LineReading Empty { get; } = new("", 0.0, []);
}

/// <summary>A text line of the output: its box (<c>x0, y0, x1, y1</c> in image pixels), text and confidence.</summary>
public sealed class Line
{
    /// <summary>Creates a line.</summary>
    public Line(double[] box, string text, double conf)
    {
        Box = box;
        Text = text;
        Conf = conf;
    }

    /// <summary><c>x0, y0, x1, y1</c> in image pixels.</summary>
    [JsonPropertyName("box")]
    public double[] Box { get; set; }

    /// <summary>The line's text.</summary>
    [JsonPropertyName("text")]
    public string Text { get; set; }

    /// <summary>Mean CTC peak probability (1 for text-layer lines).</summary>
    [JsonPropertyName("conf")]
    public double Conf { get; set; }
}

/// <summary>A layout block of the output.</summary>
public sealed class Block
{
    /// <summary>One of <see cref="ReadingOrder.BlockClasses"/>.</summary>
    [JsonPropertyName("label")]
    public string Label { get; set; } = "";

    /// <summary>The union of the block's line boxes, in image pixels.</summary>
    [JsonPropertyName("bbox")]
    public double[] BBox { get; set; } = [];

    /// <summary>The block's lines, newline-joined.</summary>
    [JsonPropertyName("text")]
    public string Text { get; set; } = "";
}

/// <summary>Where a page's text came from.</summary>
public static class PageSources
{
    /// <summary>Recognized from the image.</summary>
    public const string Ocr = "ocr";

    /// <summary>Read from the PDF text layer.</summary>
    public const string Text = "text";

    /// <summary>Text layer plus OCR of text inside images.</summary>
    public const string Hybrid = "hybrid";

    /// <summary>The page failed; <see cref="Page.Note"/> says why.</summary>
    public const string Error = "error";
}

/// <summary>One page of output (port of the Python <c>Page</c> dataclass).</summary>
public sealed class Page
{
    /// <summary>Lines in reading order.</summary>
    [JsonPropertyName("lines")]
    public List<Line> Lines { get; set; } = [];

    /// <summary>Per-step milliseconds, with <c>total_ms</c>.</summary>
    [JsonPropertyName("timings")]
    public Dictionary<string, double> Timings { get; set; } = [];

    /// <summary>Markdown: headings, paragraphs, lists, and tables as HTML.</summary>
    [JsonPropertyName("markdown")]
    public string? Markdown { get; set; }

    /// <summary><c>ocr</c>, <c>text</c>, <c>hybrid</c> or <c>error</c> (see <see cref="PageSources"/>).</summary>
    [JsonPropertyName("source")]
    public string Source { get; set; } = PageSources.Ocr;

    /// <summary>Why the page took its path.</summary>
    [JsonPropertyName("note")]
    public string Note { get; set; } = "";

    /// <summary>Layout blocks in reading order.</summary>
    [JsonPropertyName("blocks")]
    public List<Block> Blocks { get; set; } = [];

    /// <summary>Degrees counter-clockwise the input was turned; boxes are in the upright page.</summary>
    [JsonPropertyName("rotation")]
    public int Rotation { get; set; }

    /// <summary>Degrees of small tilt straightened.</summary>
    [JsonPropertyName("skew")]
    public double Skew { get; set; }

    /// <summary>One line per detected line, in reading order.</summary>
    [JsonIgnore]
    public string Text => string.Join("\n", Lines.Select(l => l.Text));
}
