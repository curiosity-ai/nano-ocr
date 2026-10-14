namespace NanoOcr.Pipeline;

/// <summary>What a <see cref="PageElement"/> is rendered as.</summary>
public enum ElementKind
{
    /// <summary>A paragraph: the lines joined (<c>join_lines</c>).</summary>
    Paragraph,

    /// <summary>A <c>Section-Header</c> block.</summary>
    Heading,

    /// <summary>A <c>List-Group</c> block: one item per bullet or number.</summary>
    List,

    /// <summary>A <c>Table</c> block of more than one line, with its cell grid.</summary>
    Table,

    /// <summary>A <c>Caption</c> block.</summary>
    Caption,

    /// <summary>A block whose every line is one formula: display equations.</summary>
    Equation,
}

/// <summary>One cell of a <see cref="TableGrid"/>; an empty cell has empty text.</summary>
/// <param name="Text">The cell's text, unescaped.</param>
/// <param name="ColSpan">Columns the cell covers.</param>
public sealed record TableCell(string Text, int ColSpan = 1);

/// <summary>One row of a <see cref="TableGrid"/>.</summary>
/// <param name="Cells">The row's cells, left to right, covering every column.</param>
/// <param name="Header">Whether the row is the header (<c>&lt;th&gt;</c> in the reference markdown).</param>
public sealed record TableRow(IReadOnlyList<TableCell> Cells, bool Header);

/// <summary>
/// A table's cells as <c>table_html</c> and <c>table_html_heads</c> lay them out, before any
/// markup: rows top to bottom, each covering <see cref="Columns"/> columns.
/// </summary>
public sealed class TableGrid
{
    /// <summary>Creates a grid.</summary>
    public TableGrid(int columns, List<TableRow> rows)
    {
        Columns = columns;
        Rows = rows;
        OwnRows = rows.Count;
    }

    /// <summary>The rows the table was assembled with, before any merge.</summary>
    public int OwnRows { get; }

    /// <summary>Drops rows a merge appended.</summary>
    public void Unmerge() => Rows.RemoveRange(OwnRows, Rows.Count - OwnRows);

    /// <summary>The number of columns.</summary>
    public int Columns { get; }

    /// <summary>The rows; a table merged across a page break gains the next page's rows.</summary>
    public List<TableRow> Rows { get; }
}

/// <summary>
/// One layout block of a page in the form the markdown writers consume: <c>page_markdown</c>'s
/// decision about each block (class, whole-math test, table assembly, list items) taken once, so
/// that the Python-identical markdown and the PaddleOCR-style markdown render the same blocks, and
/// the document-level passes (<see cref="MarkdownDocument"/>) can rework them across pages.
/// </summary>
public sealed class PageElement
{
    /// <summary>The block class, one of <see cref="ReadingOrder.BlockClasses"/>.</summary>
    public string Label { get; init; } = "";

    /// <summary>How the block renders.</summary>
    public ElementKind Kind { get; init; }

    /// <summary>A <c>Page-Header</c> or <c>Page-Footer</c>: left out unless furniture is kept.</summary>
    public bool Furniture { get; init; }

    /// <summary>The block's lines, in reading order.</summary>
    public IReadOnlyList<string> Lines { get; init; } = [];

    /// <summary>
    /// The rendered text of a paragraph, heading or caption (<c>join_lines</c>), or the
    /// <c>$$…$$</c> lines of an equation.
    /// </summary>
    public string Text { get; init; } = "";

    /// <summary>A list's items, each joined.</summary>
    public IReadOnlyList<string> Items { get; init; } = [];

    /// <summary>
    /// Title rows <c>table_html</c> lifted out of the table (a first row that is one line across
    /// most of it), in order; they precede the table.
    /// </summary>
    public IReadOnlyList<string> TableTitles { get; init; } = [];

    /// <summary>The table's cells, for <see cref="ElementKind.Table"/>.</summary>
    public TableGrid? Table { get; init; }

    /// <summary>The union of the block's line boxes (<c>x0, y0, x1, y1</c>, upright page pixels).</summary>
    public float[] Box { get; init; } = [0, 0, 0, 0];

    /// <summary>The median height of the block's lines, in pixels: the size of its text.</summary>
    public float LineHeight { get; init; }

    /// <summary>
    /// Set by <see cref="MarkdownDocument"/> on the heading it takes for the document's title
    /// (written <c># </c>, as PaddleOCR writes a <c>doc_title</c>).
    /// </summary>
    public bool DocTitle { get; set; }

    /// <summary>
    /// The heading's depth, 1 being a top-level section, when <see cref="MarkdownOptions.TitleLevels"/>
    /// assigned one (written with <c>level + 1</c> hashes, as PaddleOCR does).
    /// </summary>
    public int? TitleLevel { get; set; }

    /// <summary>
    /// Set on a table whose rows <see cref="TableMerger"/> moved into the same table on the
    /// previous page; it then renders as nothing.
    /// </summary>
    public bool MergedIntoPrevious { get; set; }
}
