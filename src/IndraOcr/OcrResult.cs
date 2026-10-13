using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using IndraOcr.Imaging;
using IndraOcr.Pipeline;
using SkiaSharp;

namespace IndraOcr;

/// <summary>
/// The result for one page, shaped like PaddleOCR 3.x's result objects: <c>rec_texts</c>,
/// <c>rec_scores</c>, <c>rec_boxes</c>, <c>dt_polys</c>, plus Indra's markdown and layout blocks, and
/// <see cref="Print"/>, <see cref="SaveToJson"/>, <see cref="SaveToMarkdown"/> and <see cref="SaveToImg"/>.
/// </summary>
public sealed class OcrResult
{
    internal OcrResult(string inputPath, int pageIndex, Page page, GrayImage? image)
    {
        InputPath = inputPath;
        PageIndex = pageIndex;
        Page = page;
        Image = image;
    }

    /// <summary>The input file (empty for in-memory images).</summary>
    public string InputPath { get; }

    /// <summary>Zero-based page (or frame) index in the input.</summary>
    public int PageIndex { get; }

    /// <summary>The full Indra page: lines, blocks, markdown, source, timings.</summary>
    public Page Page { get; }

    /// <summary>The page image as processed (kept only when <see cref="IndraOcrOptions.KeepImages"/> is on).</summary>
    public GrayImage? Image { get; }

    /// <summary>Recognized text per line, in reading order (PaddleOCR's <c>rec_texts</c>).</summary>
    public IReadOnlyList<string> RecTexts => Page.Lines.Select(l => l.Text).ToList();

    /// <summary>Confidence per line (<c>rec_scores</c>).</summary>
    public IReadOnlyList<double> RecScores => Page.Lines.Select(l => l.Conf).ToList();

    /// <summary>Axis-aligned line boxes <c>[x0, y0, x1, y1]</c> in integer pixels (<c>rec_boxes</c>).</summary>
    public IReadOnlyList<int[]> RecBoxes => Page.Lines.Select(l => l.Box.Select(v => (int)Math.Round(v)).ToArray()).ToList();

    /// <summary>Line boxes as four corner points, clockwise from top-left (<c>dt_polys</c>).</summary>
    public IReadOnlyList<float[][]> DtPolys => Page.Lines.Select(l => new[]
    {
        new[] { (float)l.Box[0], (float)l.Box[1] },
        new[] { (float)l.Box[2], (float)l.Box[1] },
        new[] { (float)l.Box[2], (float)l.Box[3] },
        new[] { (float)l.Box[0], (float)l.Box[3] },
    }).ToList();

    /// <summary>Markdown of the page (headings, paragraphs, lists, tables as HTML).</summary>
    public string Markdown => Page.Markdown ?? Page.Text;

    /// <summary>Plain text, one line per detected line.</summary>
    public string Text => Page.Text;

    /// <summary>Layout blocks in reading order.</summary>
    public IReadOnlyList<Block> Blocks => Page.Blocks;

    /// <summary>The page as JSON, in the Python CLI's <c>--format json</c> layout.</summary>
    public string ToJson(bool indented = false)
    {
        var record = new PageRecord
        {
            File = InputPath,
            Page = PageIndex,
            Source = Page.Source,
            Rotation = Page.Rotation,
            Skew = Page.Skew,
            Note = Page.Note,
            Lines = Page.Lines,
            Markdown = Page.Markdown,
            Blocks = Page.Blocks,
            Timings = Page.Timings,
        };
        return JsonSerializer.Serialize(record, (indented ? JsonContext.Indented : JsonContext.Compact).PageRecord);
    }

    /// <summary>Writes a short summary of the page to the console (PaddleOCR's <c>res.print()</c>).</summary>
    public void Print(TextWriter? writer = null)
    {
        writer ??= Console.Out;
        writer.WriteLine($"{InputPath} page {PageIndex + 1}: {Page.Source}, {Page.Lines.Count} lines, {Page.Blocks.Count} blocks, {Page.Timings.GetValueOrDefault("total_ms"):0} ms ({Page.Note})");
        foreach (Line l in Page.Lines)
        {
            writer.WriteLine($"  [{string.Join(", ", l.Box.Select(v => v.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture)))}] {l.Conf:0.000} {l.Text}");
        }
    }

    private string BaseName(string directory)
    {
        string stem = InputPath.Length > 0 ? Path.GetFileNameWithoutExtension(InputPath) : "image";
        return Path.Combine(directory, $"{stem}_{PageIndex}");
    }

    /// <summary>Saves the page as <c>&lt;stem&gt;_&lt;page&gt;.json</c> in <paramref name="directory"/>; returns the path.</summary>
    public string SaveToJson(string directory)
    {
        Directory.CreateDirectory(directory);
        string path = BaseName(directory) + ".json";
        File.WriteAllText(path, ToJson(indented: true), new UTF8Encoding(false));
        return path;
    }

    /// <summary>Saves the markdown as <c>&lt;stem&gt;_&lt;page&gt;.md</c>; returns the path.</summary>
    public string SaveToMarkdown(string directory)
    {
        Directory.CreateDirectory(directory);
        string path = BaseName(directory) + ".md";
        File.WriteAllText(path, Markdown + "\n", new UTF8Encoding(false));
        return path;
    }

    /// <summary>
    /// Saves a PNG of the page with its layout blocks (coloured, labelled) and lines (faint blue)
    /// drawn over it; needs the page image (<see cref="IndraOcrOptions.KeepImages"/>).
    /// </summary>
    public string SaveToImg(string directory)
    {
        if (Image is null)
        {
            throw new InvalidOperationException("The page image was not kept; set IndraOcrOptions.KeepImages.");
        }

        Directory.CreateDirectory(directory);
        string path = BaseName(directory) + ".png";
        using var bitmap = new SKBitmap(new SKImageInfo(Image.Width, Image.Height, SKColorType.Rgba8888, SKAlphaType.Premul));
        Span<byte> px = bitmap.GetPixelSpan();
        for (int i = 0; i < Image.Width * Image.Height; i++)
        {
            byte g = Image.Pixels[i];
            px[i * 4] = g;
            px[(i * 4) + 1] = g;
            px[(i * 4) + 2] = g;
            px[(i * 4) + 3] = 255;
        }

        using (var canvas = new SKCanvas(bitmap))
        {
            using var linePaint = new SKPaint { Color = new SKColor(60, 120, 255, 90), Style = SKPaintStyle.Stroke, StrokeWidth = 1.5f };
            foreach (Line l in Page.Lines)
            {
                canvas.DrawRect(SKRect.Create((float)l.Box[0], (float)l.Box[1], (float)(l.Box[2] - l.Box[0]), (float)(l.Box[3] - l.Box[1])), linePaint);
            }

            using var font = new SKFont { Size = Math.Max(12, Image.Height / 90f) };
            int order = 0;
            foreach (Block b in Page.Blocks)
            {
                SKColor color = LabelColor(b.Label);
                using var paint = new SKPaint { Color = color, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
                using var text = new SKPaint { Color = color };
                canvas.DrawRect(SKRect.Create((float)b.BBox[0], (float)b.BBox[1], (float)(b.BBox[2] - b.BBox[0]), (float)(b.BBox[3] - b.BBox[1])), paint);
                canvas.DrawText($"{++order} {b.Label}", (float)b.BBox[0], (float)b.BBox[1] - 4, font, text);
            }
        }

        using SKData png = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        File.WriteAllBytes(path, png.ToArray());
        return path;
    }

    private static SKColor LabelColor(string label) => label switch
    {
        "Section-Header" => new SKColor(220, 50, 47),
        "Page-Header" or "Page-Footer" => new SKColor(150, 150, 150),
        "List-Group" => new SKColor(38, 139, 210),
        "Table" => new SKColor(133, 153, 0),
        "Caption" => new SKColor(211, 54, 130),
        "Footnote" => new SKColor(108, 113, 196),
        "Table-Of-Contents" => new SKColor(42, 161, 152),
        "Form" => new SKColor(203, 75, 22),
        _ => new SKColor(181, 137, 0),
    };
}

/// <summary>The JSON layout of one page (the Python CLI's <c>--format json</c> object).</summary>
public sealed class PageRecord
{
    /// <summary>Input file.</summary>
    [JsonPropertyName("file")]
    public string File { get; set; } = "";

    /// <summary>Page index.</summary>
    [JsonPropertyName("page")]
    public int Page { get; set; }

    /// <summary>Text source.</summary>
    [JsonPropertyName("source")]
    public string Source { get; set; } = "";

    /// <summary>Quarter turns undone, in degrees.</summary>
    [JsonPropertyName("rotation")]
    public int Rotation { get; set; }

    /// <summary>Skew undone, in degrees.</summary>
    [JsonPropertyName("skew")]
    public double Skew { get; set; }

    /// <summary>Why the page took its path.</summary>
    [JsonPropertyName("note")]
    public string Note { get; set; } = "";

    /// <summary>Lines in reading order.</summary>
    [JsonPropertyName("lines")]
    public List<Line> Lines { get; set; } = [];

    /// <summary>Markdown.</summary>
    [JsonPropertyName("markdown")]
    public string? Markdown { get; set; }

    /// <summary>Layout blocks.</summary>
    [JsonPropertyName("blocks")]
    public List<Block> Blocks { get; set; } = [];

    /// <summary>Timings in milliseconds.</summary>
    [JsonPropertyName("timings")]
    public Dictionary<string, double> Timings { get; set; } = [];
}

/// <summary>Source-generated JSON metadata (no reflection, so native AOT works).</summary>
[JsonSerializable(typeof(PageRecord))]
[JsonSerializable(typeof(Page))]
internal sealed partial class JsonContext : JsonSerializerContext
{
    /// <summary>Indented output, non-ASCII written as is (Python's <c>ensure_ascii=False</c>).</summary>
    public static JsonContext Indented { get; } = new(new JsonSerializerOptions
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    });

    /// <summary>One-line output, non-ASCII written as is.</summary>
    public static JsonContext Compact { get; } = new(new JsonSerializerOptions
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    });
}
