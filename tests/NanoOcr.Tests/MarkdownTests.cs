using NanoOcr.Pipeline;
using Xunit;

namespace NanoOcr.Tests;

/// <summary>
/// The PaddleOCR-style markdown (<see cref="MarkdownWriter"/>) and the document-level passes
/// (<see cref="MarkdownDocument"/>, <see cref="TitleLevels"/>, <see cref="TableMerger"/>) on
/// synthetic pages. The Python-identical markdown is pinned by <c>PipelineParityTests</c>.
/// </summary>
public class MarkdownTests
{
    private const int Text = 0, Section = 1, Footer = 3, ListGroup = 4, Table = 5, Caption = 6;

    private const string Td = "<td style='text-align: center; word-wrap: break-word;'>";
    private const string StyledTable = "<table border=1 style='margin: auto; word-wrap: break-word;'>";

    private static (LineBox, string) L(float x0, float y0, float x1, float y1, string text) => (new LineBox(x0, y0, x1, y1), text);

    /// <summary>One page's elements from blocks of (class, lines), in reading order.</summary>
    private static List<PageElement> Elements(params (int Cls, (LineBox Box, string Text)[] Lines)[] blocks)
    {
        var lines = blocks.SelectMany((b, bi) => b.Lines.Select(l => (Block: bi, b.Cls, l.Box, l.Text))).ToList();
        int n = lines.Count, classes = ReadingOrder.BlockClasses.Length;
        float[] cls = new float[n * classes];
        float[] same = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            cls[(i * classes) + lines[i].Cls] = 5f;
            for (int j = 0; j < n; j++)
            {
                same[(i * n) + j] = lines[i].Block == lines[j].Block ? 5f : -5f;
            }
        }

        return Layout.PageElements(lines.Select(l => l.Box).ToList(), lines.Select(l => l.Text).ToList(), Enumerable.Range(0, n).ToList(), cls, same);
    }

    /// <summary>A three-row, two-column table block whose rows start at <paramref name="y"/>.</summary>
    private static (int, (LineBox, string)[]) TableBlock(float y, params (string A, string B)[] rows) =>
        (Table, rows.SelectMany((r, i) => new[] { L(0, y + (i * 20), 60, y + (i * 20) + 10, r.A), L(100, y + (i * 20), 160, y + (i * 20) + 10, r.B) }).ToArray());

    private static string Row(params string[] cells) => "<tr>" + string.Concat(cells.Select(c => Td + c + "</td>")) + "</tr>";

    [Fact]
    public void PaddleStyleWritesWhatPaddleOcrWrites()
    {
        List<PageElement> page = Elements(
            (Section, [L(0, 0, 300, 20, "1. Introduction")]),
            (Text, [L(0, 30, 300, 40, "Optical character recog-"), L(0, 45, 300, 55, "nition turns pages.")]),
            (ListGroup, [L(0, 60, 300, 70, "1. Detect."), L(0, 75, 300, 85, "2. Read.")]),
            (Caption, [L(0, 100, 300, 110, "Table 1: Speed.")]),
            TableBlock(200, ("Engine", "Pages/s"), ("Nano", "49"), ("Other", "3")),
            (Footer, [L(0, 900, 50, 910, "Page 1")]));

        string expected =
            "### 1. Introduction\n\n"
            + "Optical character recognition turns pages.\n\n"
            + "1. Detect.\n\n2. Read.\n\n"
            + "<div style=\"text-align: center;\">Table 1: Speed.</div>\n\n\n\n"
            + StyledTable + Row("Engine", "Pages/s") + Row("Nano", "49") + Row("Other", "3") + "</table>";
        Assert.Equal(expected, MarkdownWriter.Write(page, MarkdownOptions.Default));

        // The plain variant: no styles, the caption as text, the table two lines down.
        Assert.Equal(
            "Table 1: Speed.\n\n\n\n<table><tr><td>Engine</td><td>Pages/s</td></tr><tr><td>Nano</td><td>49</td></tr><tr><td>Other</td><td>3</td></tr></table>",
            MarkdownWriter.Write(page.Skip(3).ToList(), MarkdownOptions.Default with { Pretty = false }));

        // The same blocks in the Python original's markdown.
        Assert.Equal(
            "## 1. Introduction\n\nOptical character recognition turns pages.\n\n1. Detect.\n2. Read.\n\nTable 1: Speed.\n\n"
            + "<table>\n<tr><th>Engine</th><th>Pages/s</th></tr>\n<tr><td>Nano</td><td>49</td></tr>\n<tr><td>Other</td><td>3</td></tr>\n</table>",
            MarkdownWriter.Write(page, MarkdownOptions.Reference));
    }

    [Theory]
    [InlineData("1. Introduction", "### 1. Introduction")]
    [InlineData("2.Method", "### 2. Method")]
    [InlineData("1.2 Scope", "### 1.2 Scope")]
    [InlineData("1.2.3 Detail", "#### 1.2.3 Detail")]
    [InlineData("Results", "## Results")]
    [InlineData("（三）结果", "## （三） 结果")]
    public void HeadingDepthFollowsNumbering(string title, string expected) =>
        Assert.Equal(expected, MarkdownWriter.FormatTitle(title));

    [Fact]
    public void TitleLevelsVoteOnNumberingStyleAndSize()
    {
        int[] levels = TitleLevels.Levels(
            ["1 Introduction", "1.1 Scope", "2 Method", "Abstract", "Appendix notes"],
            [30, 22, 31, 22, 22]);
        Assert.Equal([1, 2, 1, 1, 2], levels);

        // One type size measured a pixel apart is one level, not two.
        IReadOnlyDictionary<int, int> map = TitleLevels.ClusterHeights([20, 21, 20, 21]);
        Assert.Equal(1, map[20]);
        Assert.Equal(1, map[21]);
        map = TitleLevels.ClusterHeights([16, 24, 32]);
        Assert.Equal([3, 2, 1], new[] { map[16], map[24], map[32] });
    }

    [Fact]
    public void TheOpeningHeadingIsTheTitleOnlyWhenClearlyLargest()
    {
        static Page Of(List<PageElement> e) => new() { Elements = e };
        var big = Of(Elements(
            (Section, [L(0, 0, 400, 30, "Fast Line Recognition")]),
            (Section, [L(0, 50, 300, 70, "1. Introduction")]),
            (Text, [L(0, 80, 300, 90, "Body.")])));
        MarkdownDocument.Finish([big], MarkdownOptions.Default);
        Assert.StartsWith("# Fast Line Recognition\n\n### 1. Introduction\n\n", big.Markdown);

        var even = Of(Elements(
            (Section, [L(0, 0, 400, 20, "Section 1: Digital documents")]),
            (Section, [L(0, 50, 300, 70, "1. Introduction")])));
        MarkdownDocument.Finish([even], MarkdownOptions.Default);
        Assert.StartsWith("## Section 1", even.Markdown);

        var later = Of(Elements(
            (Section, [L(0, 0, 400, 30, "Fast Line Recognition")]),
            (Section, [L(0, 50, 300, 70, "1. Introduction")])));
        MarkdownDocument.Finish([later], MarkdownOptions.Default, first: false);
        Assert.StartsWith("## Fast Line Recognition", later.Markdown);
    }

    [Fact]
    public void ATableSplitByAPageBreakIsRejoined()
    {
        Page First() => new()
        {
            Elements = Elements(
                (Text, [L(0, 0, 300, 10, "Before the table.")]),
                TableBlock(100, ("Engine", "Pages/s"), ("Nano", "49"), ("Other", "3")),
                (Footer, [L(0, 900, 50, 910, "Page 1")])),
        };
        Page Second() => new()
        {
            Elements = Elements(
                TableBlock(100, ("Engine", "Pages/s"), ("Third", "7"), ("Fourth", "1")),
                (Text, [L(0, 300, 300, 310, "After the table.")])),
        };

        Page a = First(), b = Second();
        var merge = MarkdownOptions.Default with { MergeTables = true };
        MarkdownDocument.Finish([a, b], merge);
        Assert.Equal(
            "Before the table.\n\n\n" + StyledTable
            + Row("Engine", "Pages/s") + Row("Nano", "49") + Row("Other", "3") + Row("Third", "7") + Row("Fourth", "1") + "</table>",
            a.Markdown);
        Assert.Equal("After the table.", b.Markdown);

        // Again: the same result, not the rows twice. Without merging, the pages as recognized.
        MarkdownDocument.Finish([a, b], merge);
        Assert.Equal(5, a.Elements![1].Table!.Rows.Count);
        MarkdownDocument.Finish([a, b], MarkdownOptions.Default);
        Assert.Equal(3, a.Elements![1].Table!.Rows.Count);
        Assert.StartsWith("\n" + StyledTable, b.Markdown);

        // Text after the first table ends it.
        Page c = new()
        {
            Elements = Elements(
                TableBlock(100, ("Engine", "Pages/s"), ("Nano", "49"), ("Other", "3")),
                (Text, [L(0, 300, 300, 310, "Closing words.")])),
        };
        Page d = Second();
        MarkdownDocument.Finish([c, d], merge);
        Assert.Contains("<table", d.Markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void RepeatedHeadersAreDetectedRowByRow()
    {
        var a = new TableGrid(2, [new([new("A"), new("B")], true), new([new("1"), new("2")], false)]);
        var b = new TableGrid(2, [new([new("Ａ"), new(" B ")], true), new([new("3"), new("4")], false)]);
        Assert.Equal((1, true), TableMerger.HeaderRows(a, b));
        var c = new TableGrid(2, [new([new("x"), new("y")], true)]);
        Assert.Equal((0, false), TableMerger.HeaderRows(a, c));
    }
}
