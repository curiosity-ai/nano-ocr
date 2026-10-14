using System.Text;
using System.Text.RegularExpressions;

namespace NanoOcr.Pipeline;

/// <summary>Which markdown a page is written as.</summary>
public enum MarkdownStyle
{
    /// <summary>
    /// The markdown PaddleOCR's document parser writes (PaddleX's <c>MarkdownConverter</c>, as the
    /// sibling port <c>PaddleOcrSharp</c> reproduces it), so the two engines' output can be
    /// consumed the same way. The default.
    /// </summary>
    PaddleOcr,

    /// <summary>
    /// The Python original's <c>layout_post.page_markdown</c>, byte for byte: what the parity
    /// fixtures pin.
    /// </summary>
    Reference,
}

/// <summary>Markdown settings (<see cref="NanoOcrEngine"/>, <see cref="MarkdownDocument"/>).</summary>
public sealed record MarkdownOptions
{
    /// <summary>The PaddleOCR-style markdown, decorated, with no document-level restructuring.</summary>
    public static MarkdownOptions Default { get; } = new();

    /// <summary>The Python original's markdown (<see cref="MarkdownStyle.Reference"/>).</summary>
    public static MarkdownOptions Reference { get; } = new() { Style = MarkdownStyle.Reference };

    /// <summary>The markdown flavour.</summary>
    public MarkdownStyle Style { get; init; } = MarkdownStyle.PaddleOcr;

    /// <summary>
    /// PaddleOCR's <c>pretty</c> rendering (the default there too): tables carry border and
    /// centring styles and captions are centred in a <c>div</c>. Off gives its plain variant:
    /// unstyled table HTML, captions as text. Ignored by <see cref="MarkdownStyle.Reference"/>.
    /// </summary>
    public bool Pretty { get; init; } = true;

    /// <summary>
    /// Set every heading's depth from the whole document — its numbering, its numbering style's
    /// first appearance and its text size against every other heading's — as PaddleOCR's
    /// <c>AssignTitleLevels</c> does (<see cref="TitleLevels"/>). Off, a heading's depth comes from
    /// its numbering alone (<c>1.2 …</c> is deeper than <c>1 …</c>). PaddleOCR style only.
    /// </summary>
    public bool TitleLevels { get; init; }

    /// <summary>
    /// Rejoin a table that a page break split in two, as PaddleOCR's
    /// <c>MergeTablesAcrossPages</c> does (<see cref="TableMerger"/>). Needs the whole document;
    /// <see cref="NanoOCR"/> then holds a file's pages until its last one is done.
    /// </summary>
    public bool MergeTables { get; init; }

    /// <summary>
    /// Take a document's first heading for its title (written <c># </c>, as PaddleOCR writes a
    /// <c>doc_title</c>) when it opens the first page and its text is clearly the largest of the
    /// document's headings. The layout model has no title class; this is the size test that
    /// stands in for one. PaddleOCR style only.
    /// </summary>
    public bool DocTitle { get; init; } = true;

    /// <summary>Whether rendering a page needs the document's other pages.</summary>
    public bool NeedsDocument => Style == MarkdownStyle.PaddleOcr && (TitleLevels || MergeTables);
}

/// <summary>
/// Renders <see cref="PageElement"/>s as markdown, in either <see cref="MarkdownStyle"/>.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Reference"/> is <c>page_markdown</c>'s rendering. <see cref="Write"/> in
/// <see cref="MarkdownStyle.PaddleOcr"/> follows PaddleX's <c>MarkdownConverter</c> and its
/// <c>build_handle_funcs_dict</c> formatters, mapping this layout model's classes onto PaddleOCR's
/// labels: <c>Section-Header</c> → <c>paragraph_title</c> (or <c>doc_title</c>), <c>Caption</c> →
/// <c>table_title</c>, <c>Table</c> → <c>table</c>, a block of display equations →
/// <c>display_formula</c>, everything else → <c>text</c>. What that means on the page:
/// </para>
/// <list type="bullet">
/// <item>Blocks are separated by a blank line, as in both.</item>
/// <item>A heading's hashes come from its numbering (<c>format_title</c>: <c>1. Intro</c> and
/// <c>1.2 Intro</c> are <c>###</c>, an unnumbered one <c>##</c>), from
/// <see cref="PageElement.TitleLevel"/> when <see cref="MarkdownOptions.TitleLevels"/> set one,
/// and the document title is <c>#</c>.</item>
/// <item>A table is one line of HTML, every cell a <c>&lt;td&gt;</c>, after an empty line, and
/// with PaddleOCR's border and centring styles when <see cref="MarkdownOptions.Pretty"/>; a title
/// row lifted out of it is a caption before it.</item>
/// <item>List items are separated by blank lines (PaddleOCR's text blocks put each line of the
/// model's output in a paragraph of its own).</item>
/// <item>A caption is centred in a <c>div</c> when pretty.</item>
/// </list>
/// <para>
/// Two things stay NanoOCR's own because they are content rather than format: inline math is
/// <c>$…$</c> (PaddleOCR's <c> $ … $</c> padding comes from rewriting its model's <c>\(…\)</c>,
/// and the padded form does not render as math in every renderer), and table cells are escaped as
/// <c>html.escape(quote=False)</c>.
/// </para>
/// </remarks>
public static partial class MarkdownWriter
{
    // markdown_format_funcs.format_title's numbering pattern, as PaddleOcrSharp.MarkdownWriter has it.
    [GeneratedRegex(
        @"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|[\(\（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、\.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)$")]
    private static partial Regex TitleNumbering();

    private const string PrettyTable = "<table border=1 style='margin: auto; word-wrap: break-word;'>";
    private const string PrettyCell = "<td style='text-align: center; word-wrap: break-word;'>";

    /// <summary>Renders one page's elements in the style <paramref name="options"/> names.</summary>
    /// <param name="elements">The page's elements, in reading order.</param>
    /// <param name="options">Markdown settings.</param>
    /// <param name="keepFurniture">Keep page headers and footers.</param>
    public static string Write(IReadOnlyList<PageElement> elements, MarkdownOptions options, bool keepFurniture = false)
    {
        if (options.Style == MarkdownStyle.Reference)
        {
            return Reference(elements, keepFurniture);
        }

        var parts = new List<string>();
        foreach (PageElement e in elements)
        {
            if ((e.Furniture && !keepFurniture) || e.MergedIntoPrevious)
            {
                continue;
            }

            if (e.Furniture && e.Kind != ElementKind.Equation)
            {
                // PaddleOCR's header and footer handlers pass the content through.
                parts.Add(Layout.JoinLines(e.Lines));
                continue;
            }

            switch (e.Kind)
            {
                case ElementKind.Heading:
                    parts.Add(e.DocTitle ? $"# {e.Text}"
                        : e.TitleLevel is int level ? $"{new string('#', level + 1)} {e.Text}"
                        : FormatTitle(e.Text));
                    break;
                case ElementKind.Table:
                    foreach (string title in e.TableTitles)
                    {
                        parts.Add(Caption(title, options));
                    }

                    parts.Add(PaddleTable(e.Table!, options.Pretty));
                    break;
                case ElementKind.List:
                    parts.Add(string.Join("\n\n", e.Items));
                    break;
                case ElementKind.Caption:
                    parts.Add(Caption(e.Text, options));
                    break;
                default:
                    parts.Add(e.Text);
                    break;
            }
        }

        return string.Join("\n\n", parts.Where(p => p.Length > 0));
    }

    /// <summary><c>page_markdown</c>'s rendering of the elements, identical to the Python.</summary>
    /// <param name="elements">The page's elements, in reading order.</param>
    /// <param name="keepFurniture">Keep page headers and footers.</param>
    public static string Reference(IReadOnlyList<PageElement> elements, bool keepFurniture = false)
    {
        var parts = new List<string>();
        foreach (PageElement e in elements)
        {
            if (e.Furniture && !keepFurniture)
            {
                continue;
            }

            parts.Add(e.Kind switch
            {
                ElementKind.Heading => "## " + e.Text,
                ElementKind.Table => ReferenceTable(e.TableTitles, e.Table!),
                ElementKind.List => string.Join("\n", e.Items),
                _ => e.Text,
            });
        }

        return string.Join("\n\n", parts.Where(p => p.Length > 0));
    }

    /// <summary>
    /// <c>table_html</c>'s markup: the titles, then one <c>&lt;tr&gt;</c> per line, the header row
    /// in <c>&lt;th&gt;</c>.
    /// </summary>
    /// <param name="titles">Title rows lifted out of the table, in order.</param>
    /// <param name="grid">The table.</param>
    public static string ReferenceTable(IReadOnlyList<string> titles, TableGrid grid)
    {
        var html = new List<string> { "<table>" };
        foreach (TableRow row in grid.Rows)
        {
            string tag = row.Header ? "th" : "td";
            var tds = new StringBuilder("<tr>");
            foreach (TableCell cell in row.Cells)
            {
                string attr = cell.ColSpan > 1 ? $" colspan=\"{cell.ColSpan}\"" : "";
                tds.Append($"<{tag}{attr}>{Py.HtmlEscape(cell.Text)}</{tag}>");
            }

            html.Add(tds.Append("</tr>").ToString());
        }

        html.Add("</table>");
        string table = string.Join("\n", html);
        return titles.Count == 0 ? table : string.Join("\n\n", titles) + "\n\n" + table;
    }

    /// <summary>
    /// A table as PaddleOCR writes it: <c>convert_otsl_to_html</c>'s single-line markup, every
    /// cell a <c>&lt;td&gt;</c>, behind the newline the table handler prepends — styled
    /// (<c>pretty</c>) or with the extra newline of its plain variant.
    /// </summary>
    /// <param name="grid">The table.</param>
    /// <param name="pretty">PaddleOCR's styled rendering.</param>
    public static string PaddleTable(TableGrid grid, bool pretty)
    {
        var sb = new StringBuilder("<table>");
        foreach (TableRow row in grid.Rows)
        {
            sb.Append("<tr>");
            foreach (TableCell cell in row.Cells)
            {
                sb.Append("<td");
                if (cell.ColSpan > 1)
                {
                    sb.Append($" colspan=\"{cell.ColSpan}\"");
                }

                sb.Append('>').Append(Py.HtmlEscape(cell.Text)).Append("</td>");
            }

            sb.Append("</tr>");
        }

        string html = sb.Append("</table>").ToString();
        return pretty
            ? "\n" + html.Replace("<table>", PrettyTable, StringComparison.Ordinal).Replace("<td>", PrettyCell, StringComparison.Ordinal)
            : "\n\n" + html;
    }

    /// <summary>A caption (<c>table_title</c>): centred in a <c>div</c> when pretty.</summary>
    private static string Caption(string text, MarkdownOptions options) =>
        options.Pretty ? $"<div style=\"text-align: center;\">{text}</div>\n" : text;

    /// <summary>
    /// <c>format_title</c>: a heading whose depth follows its numbering — <c>## </c> plus one
    /// <c>#</c> per dot in the numbered title.
    /// </summary>
    /// <param name="content">The heading's text.</param>
    public static string FormatTitle(string content)
    {
        string title = content;
        Match match = TitleNumbering().Match(title);
        if (match.Success)
        {
            title = match.Groups[1].Value.Trim() + " " + match.Groups[3].Value.TrimStart();
        }

        title = title.TrimEnd('.');
        int level = title.Contains('.', StringComparison.Ordinal) ? title.Count(c => c == '.') + 1 : 1;
        return $"{new string('#', level + 1)} {title}";
    }
}
