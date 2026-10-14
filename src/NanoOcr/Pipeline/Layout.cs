using System.Text;
using System.Text.RegularExpressions;

namespace NanoOcr.Pipeline;

/// <summary>A layout block: its class label, the union of its lines' boxes, and its text.</summary>
public sealed record LayoutBlock(string Label, double[] BBox, string Text);

/// <summary>
/// Page text from the ordered lines and the order model's layout heads (port of
/// <c>indra_ocr/layout_post.py</c>). Walking the reading order, a block starts wherever the
/// same-block logit between consecutive lines is not positive; each block is rendered by class.
/// </summary>
public static partial class Layout
{
    private static readonly int Header = Array.IndexOf(ReadingOrder.BlockClasses, "Page-Header");
    private static readonly int Footer = Array.IndexOf(ReadingOrder.BlockClasses, "Page-Footer");
    private static readonly int Section = Array.IndexOf(ReadingOrder.BlockClasses, "Section-Header");
    private static readonly int ListGroup = Array.IndexOf(ReadingOrder.BlockClasses, "List-Group");
    private static readonly int Table = Array.IndexOf(ReadingOrder.BlockClasses, "Table");
    private static readonly int Caption = Array.IndexOf(ReadingOrder.BlockClasses, "Caption");

    [GeneratedRegex(@"^\$[^$]+\$$")]
    private static partial Regex WholeMath();

    [GeneratedRegex(@"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)")]
    private static partial Regex Item();

    [GeneratedRegex(@"\G [A-Za-z]{3,}\b(?![_^{(])")]
    private static partial Regex PlainWord();

    [GeneratedRegex(@"[\\_^{]")]
    private static partial Regex LatexAhead();

    /// <summary>
    /// <c>balance_math</c>: closes <c>$</c> math spans the recognizer left open. A <c>$</c> opens
    /// math only when LaTeX or another <c>$</c> follows within 15 characters.
    /// </summary>
    public static string BalanceMath(string text)
    {
        if (!text.Contains('$', StringComparison.Ordinal))
        {
            return text;
        }

        int[] cp = Py.CodePoints(text);
        var sb = new StringBuilder();
        bool inMath = false;
        string full = text;
        for (int i = 0; i < cp.Length; i++)
        {
            int c = cp[i];
            if (c == '$')
            {
                if (inMath)
                {
                    inMath = false;
                }
                else
                {
                    string ahead = Py.FromCodePoints(cp.Skip(i + 1).Take(15));
                    inMath = LatexAhead().IsMatch(ahead) || ahead.Contains('$', StringComparison.Ordinal);
                }

                sb.Append('$');
                continue;
            }

            if (inMath && c == ' ' && PlainWord().IsMatch(Py.FromCodePoints(cp.Skip(i))))
            {
                sb.Append('$');
                inMath = false;
            }

            sb.Append(char.ConvertFromUtf32(c));
        }

        if (inMath)
        {
            sb.Append('$');
        }

        return sb.ToString();
    }

    /// <summary>Han, kana, CJK punctuation and fullwidth forms: written without spaces.</summary>
    internal static bool IsCjk(int o) =>
        (o >= 0x3000 && o <= 0x30FF) || (o >= 0x3400 && o <= 0x9FFF) || (o >= 0xF900 && o <= 0xFAFF)
        || (o >= 0xFF00 && o <= 0xFFEF) || (o >= 0x20000 && o <= 0x2FA1F);

    /// <summary>
    /// <c>join_lines</c>: the lines of one paragraph joined; a word hyphenated at a line end is
    /// rejoined, and a break between two CJK characters takes no space.
    /// </summary>
    public static string JoinLines(IEnumerable<string> texts)
    {
        var outCp = new List<int>();
        foreach (string raw in texts)
        {
            string t = Py.Strip(raw);
            if (t.Length == 0)
            {
                continue;
            }

            int[] tc = Py.CodePoints(t);
            if (outCp.Count > 1 && outCp[^1] == '-' && Py.IsAlpha(outCp[^2]) && Py.IsAlpha(tc[0]) && Py.IsLower(tc[0]))
            {
                outCp.RemoveAt(outCp.Count - 1);
                outCp.AddRange(tc);
            }
            else if (outCp.Count > 0 && IsCjk(outCp[^1]) && IsCjk(tc[0]))
            {
                outCp.AddRange(tc);
            }
            else
            {
                if (outCp.Count > 0)
                {
                    outCp.Add(' ');
                }

                outCp.AddRange(tc);
            }
        }

        return Py.FromCodePoints(outCp);
    }

    /// <summary><c>split_blocks</c>: runs of the reading order with a positive same-block logit between neighbours.</summary>
    public static List<List<int>> SplitBlocks(IReadOnlyList<int> order, float[] same, int n)
    {
        var blocks = new List<List<int>>();
        foreach (int i in order)
        {
            if (blocks.Count > 0 && same[(blocks[^1][^1] * n) + i] > 0)
            {
                blocks[^1].Add(i);
            }
            else
            {
                blocks.Add([i]);
            }
        }

        return blocks;
    }

    private static List<List<int>> Rows(IReadOnlyList<LineBox> boxes)
    {
        int n = boxes.Count;
        var cy = new float[n];
        var hs = new float[n];
        for (int i = 0; i < n; i++)
        {
            cy[i] = (boxes[i].Y0 + boxes[i].Y1) / 2f;
            hs[i] = boxes[i].Y1 - boxes[i].Y0;
        }

        double h = Py.Median(hs);
        var rows = new List<List<int>>();
        foreach (int i in Py.ArgSort(cy))
        {
            if (rows.Count > 0)
            {
                float mean = Py.Mean(rows[^1].Select(k => cy[k]).ToArray());
                if (MathF.Abs(cy[i] - mean) < (float)(0.5 * h))
                {
                    rows[^1].Add(i);
                    continue;
                }
            }

            rows.Add([i]);
        }

        return rows.Select(r => r.OrderBy(k => boxes[k].X0).ToList()).ToList();
    }

    private static List<List<int>> MergeWrapped(List<List<int>> rows, IReadOnlyList<LineBox> boxes)
    {
        if (rows.Count < 3)
        {
            return rows;
        }

        double h = Py.Median(boxes.Select(b => b.Y1 - b.Y0).ToArray());
        var gaps = new List<double>();
        for (int r = 0; r + 1 < rows.Count; r++)
        {
            float top = rows[r + 1].Min(k => boxes[k].Y0);
            float bottom = rows[r].Max(k => boxes[k].Y1);
            gaps.Add(top - bottom);
        }

        double med = Py.Median(gaps.ToArray());
        var result = new List<List<int>> { new(rows[0]) };
        List<int> prev = rows[0];
        for (int r = 1; r < rows.Count; r++)
        {
            double g = gaps[r - 1];
            List<int> row = rows[r];
            if (g < 0.5 * med && g < 0.6 * h && row.Count < prev.Count)
            {
                result[^1].AddRange(row);
            }
            else
            {
                result.Add(new List<int>(row));
            }

            prev = row;
        }

        return result;
    }

    /// <summary>
    /// <c>table_html</c>: an HTML table from cell lines. Rows by vertical centre; column bands
    /// from the fullest rows; a cell spans every band it covers by more than half; a first row that
    /// is one line across most of the table is a title placed before it; the first row is the header.
    /// </summary>
    public static string TableHtml(IReadOnlyList<LineBox> boxes, IReadOnlyList<string> texts)
    {
        var titles = new List<string>();
        TableGrid grid = TableGridOf(boxes, texts, titles);
        return MarkdownWriter.ReferenceTable(titles, grid);
    }

    /// <summary>
    /// The cell grid of <see cref="TableHtml"/>; title rows lifted out of the table are appended
    /// to <paramref name="titles"/>.
    /// </summary>
    public static TableGrid TableGridOf(IReadOnlyList<LineBox> boxes, IReadOnlyList<string> texts, List<string> titles)
    {
        List<List<int>> rows = Rows(boxes);
        float spanX = boxes.Max(b => b.X1) - boxes.Min(b => b.X0);
        if (rows.Count > 2 && rows[0].Count == 1 && rows.Max(r => r.Count) > 1
            && boxes[rows[0][0]].X1 - boxes[rows[0][0]].X0 > 0.6f * spanX)
        {
            int title = rows[0][0];
            var keep = Enumerable.Range(0, texts.Count).Where(k => k != title).ToList();
            titles.Add(texts[title]);
            return TableGridOf(keep.Select(k => boxes[k]).ToList(), keep.Select(k => texts[k]).ToList(), titles);
        }

        int width = rows.Max(r => r.Count);
        var full = rows.Where(r => r.Count == width).ToList();
        rows = MergeWrapped(rows, boxes);
        var bands = new float[width, 2];
        for (int c = 0; c < width; c++)
        {
            bands[c, 0] = Py.Median(full.Select(r => boxes[r[c]].X0).ToArray());
            bands[c, 1] = Py.Median(full.Select(r => boxes[r[c]].X1).ToArray());
        }

        var edges = new double[width + 1];
        edges[0] = double.NegativeInfinity;
        for (int c = 0; c + 1 < width; c++)
        {
            edges[c + 1] = (bands[c, 1] + bands[c + 1, 0]) / 2f;
        }

        edges[width] = double.PositiveInfinity;
        var gridRows = new List<TableRow>();
        for (int ri = 0; ri < rows.Count; ri++)
        {
            var cells = new Dictionary<int, (int Span, string Text)>();
            foreach (int k in rows[ri].OrderBy(k => boxes[k].Y0).ThenBy(k => boxes[k].X0))
            {
                float x0 = boxes[k].X0, x1 = boxes[k].X1;
                int first = SearchSortedRight(edges, x0) - 1;
                int last = SearchSortedRight(edges, x1) - 1;
                var span = new List<int>();
                for (int c = Math.Max(first, 0); c <= Math.Min(last, width - 1); c++)
                {
                    if (MathF.Min(x1, bands[c, 1]) - MathF.Max(x0, bands[c, 0]) > 0.5f * (bands[c, 1] - bands[c, 0]))
                    {
                        span.Add(c);
                    }
                }

                if (span.Count == 0)
                {
                    span.Add(Math.Min(Math.Max(first, 0), width - 1));
                }

                int c0 = span[0];
                if (cells.TryGetValue(c0, out var prev))
                {
                    cells[c0] = (Math.Max(prev.Span, span.Count), prev.Text + " " + texts[k]);
                }
                else
                {
                    cells[c0] = (span.Count, texts[k]);
                }
            }

            gridRows.Add(GridRow(cells, width, ri == 0 && rows.Count > 1));
        }

        return new TableGrid(width, gridRows);
    }

    /// <summary>
    /// One row of the grid: the cells placed at their first column, an empty cell for every column
    /// no cell starts at or spans (the loop both <c>table_html</c> functions end with).
    /// </summary>
    private static TableRow GridRow(Func<int, (bool Found, int Span, string Text)> cellAt, int width, bool header)
    {
        var cells = new List<TableCell>();
        int col = 0;
        while (col < width)
        {
            var (found, span, text) = cellAt(col);
            if (found)
            {
                cells.Add(new TableCell(text, span));
                col += span;
            }
            else
            {
                cells.Add(new TableCell(""));
                col++;
            }
        }

        return new TableRow(cells, header);
    }

    private static TableRow GridRow(Dictionary<int, (int Span, string Text)> cells, int width, bool header) =>
        GridRow(c => cells.TryGetValue(c, out var v) ? (true, v.Span, v.Text) : (false, 0, ""), width, header);

    /// <summary><c>np.searchsorted(a, v, side="right")</c>.</summary>
    private static int SearchSortedRight(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (a[mid] <= v)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }

    /// <summary>The <c>[3, n, n]</c> table logits restricted to the lines of one block.</summary>
    private static float[] SubTab(float[] tab, int n, IReadOnlyList<int> blk)
    {
        int m = blk.Count;
        var r = new float[3 * m * m];
        for (int k = 0; k < 3; k++)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    r[(((k * m) + i) * m) + j] = tab[(((k * n) + blk[i]) * n) + blk[j]];
                }
            }
        }

        return r;
    }

    /// <summary>
    /// <c>heads_plausible</c>: the lines of one cell are stacked; when the cell head joins
    /// side-by-side lines in more than 10% of its cells, the geometric assembly is used instead.
    /// </summary>
    public static bool HeadsPlausible(IReadOnlyList<LineBox> boxes, float[] tab)
    {
        int n = boxes.Count;
        List<List<int>> cells = ReadingOrder.Cluster(tab.AsSpan(2 * n * n, n * n).ToArray(), n);
        int bad = 0;
        foreach (List<int> c in cells)
        {
            if (c.Count < 2)
            {
                continue;
            }

            float minOverlap = float.PositiveInfinity;
            foreach (int p in c)
            {
                foreach (int q in c)
                {
                    minOverlap = MathF.Min(minOverlap, MathF.Min(boxes[p].X1, boxes[q].X1) - MathF.Max(boxes[p].X0, boxes[q].X0));
                }
            }

            if (minOverlap < 0)
            {
                bad++;
            }
        }

        return bad <= 0.1 * cells.Count;
    }

    /// <summary><c>table_html_heads</c>: an HTML table from the order model's row / column / cell heads.</summary>
    public static string TableHtmlHeads(IReadOnlyList<LineBox> boxes, IReadOnlyList<string> texts, float[] tab) =>
        MarkdownWriter.ReferenceTable([], TableGridHeads(boxes, texts, tab));

    /// <summary>The cell grid of <see cref="TableHtmlHeads"/>.</summary>
    public static TableGrid TableGridHeads(IReadOnlyList<LineBox> boxes, IReadOnlyList<string> texts, float[] tab)
    {
        int n = boxes.Count;
        List<List<int>> cells = ReadingOrder.Cluster(tab.AsSpan(2 * n * n, n * n).ToArray(), n);
        int m = cells.Count;
        var cb = cells.Select(c => new[] { c.Min(k => boxes[k].X0), c.Min(k => boxes[k].Y0), c.Max(k => boxes[k].X1), c.Max(k => boxes[k].Y1) }).ToArray();
        var ctext = cells.Select(c => string.Join(" ", c.OrderBy(k => boxes[k].Y0).ThenBy(k => boxes[k].X0).Select(k => texts[k]))).ToArray();

        float[] CellLogits(int k)
        {
            var o = new float[m * m];
            for (int a = 0; a < m; a++)
            {
                for (int b = a + 1; b < m; b++)
                {
                    float sum = 0;
                    var vals = new List<float>();
                    foreach (int i in cells[a])
                    {
                        foreach (int j in cells[b])
                        {
                            vals.Add(tab[(((k * n) + i) * n) + j]);
                        }
                    }

                    sum = Py.Mean(vals.ToArray());
                    o[(a * m) + b] = o[(b * m) + a] = sum;
                }
            }

            return o;
        }

        var rows = ReadingOrder.Cluster(CellLogits(0), m).OrderBy(g => g.Min(c => cb[c][1])).ToList();
        var cols = ReadingOrder.Cluster(CellLogits(1), m).OrderBy(g => g.Min(c => cb[c][0])).ToList();
        var rowOf = new Dictionary<int, int>();
        var colOf = new Dictionary<int, int>();
        for (int r = 0; r < rows.Count; r++)
        {
            foreach (int c in rows[r])
            {
                rowOf[c] = r;
            }
        }

        for (int k = 0; k < cols.Count; k++)
        {
            foreach (int c in cols[k])
            {
                colOf[c] = k;
            }
        }

        float[] centers = cols.Select(g => Py.Median(g.Select(c => (cb[c][0] + cb[c][2]) / 2f).ToArray())).ToArray();
        float h = Py.Median(boxes.Select(b => b.Y1 - b.Y0).ToArray());
        var grid = new Dictionary<(int, int), (int Span, string Text)>();
        for (int c = 0; c < m; c++)
        {
            int c0 = colOf[c];
            int span = 1;
            while (c0 + span < cols.Count && cb[c][0] - h <= centers[c0 + span] && centers[c0 + span] <= cb[c][2] + h)
            {
                span++;
            }

            var key = (rowOf[c], c0);
            grid[key] = grid.TryGetValue(key, out var prev) ? (Math.Max(prev.Span, span), prev.Text + " " + ctext[c]) : (span, ctext[c]);
        }

        var gridRows = new List<TableRow>();
        for (int r = 0; r < rows.Count; r++)
        {
            int row = r;
            gridRows.Add(GridRow(
                c => grid.TryGetValue((row, c), out var v) ? (true, v.Span, v.Text) : (false, 0, ""),
                cols.Count,
                r == 0 && rows.Count > 1));
        }

        return new TableGrid(cols.Count, gridRows);
    }

    /// <summary><c>page_blocks</c>: the layout blocks behind <see cref="PageMarkdown"/>, in reading order.</summary>
    public static List<LayoutBlock> PageBlocks(IReadOnlyList<LineBox> boxes, IReadOnlyList<string> texts, IReadOnlyList<int> order, float[] cls, float[] same)
    {
        int n = boxes.Count;
        int classes = ReadingOrder.BlockClasses.Length;
        var result = new List<LayoutBlock>();
        foreach (List<int> blk in SplitBlocks(order, same, n))
        {
            int label = ReadingOrder.BlockClass(cls, classes, blk);
            double[] bbox =
            [
                Py.Round(blk.Min(i => boxes[i].X0), 1),
                Py.Round(blk.Min(i => boxes[i].Y0), 1),
                Py.Round(blk.Max(i => boxes[i].X1), 1),
                Py.Round(blk.Max(i => boxes[i].Y1), 1),
            ];
            result.Add(new LayoutBlock(ReadingOrder.BlockClasses[label], bbox, string.Join("\n", blk.Select(i => texts[i]))));
        }

        return result;
    }

    /// <summary>
    /// <c>page_markdown</c>: markdown for one page. Section headers become <c>## </c>, tables HTML,
    /// lists one item per line, everything else a paragraph; page headers and footers are dropped
    /// unless <paramref name="keepFurniture"/>. Identical to the Python; see
    /// <see cref="MarkdownWriter"/> for the PaddleOCR-style rendering of the same blocks.
    /// </summary>
    public static string PageMarkdown(
        IReadOnlyList<LineBox> boxes,
        IReadOnlyList<string> texts,
        IReadOnlyList<int> order,
        float[] cls,
        float[] same,
        bool keepFurniture = false,
        IReadOnlyDictionary<int, List<(LineBox Box, string Text)>>? cells = null,
        float[]? tab = null) =>
        MarkdownWriter.Reference(PageElements(boxes, texts, order, cls, same, cells, tab), keepFurniture);

    /// <summary>
    /// The blocks of <see cref="PageMarkdown"/>, each with the decisions <c>page_markdown</c>
    /// takes about it — class, whole-math test, table cells, list items — before any markup.
    /// </summary>
    public static List<PageElement> PageElements(
        IReadOnlyList<LineBox> boxes,
        IReadOnlyList<string> texts,
        IReadOnlyList<int> order,
        float[] cls,
        float[] same,
        IReadOnlyDictionary<int, List<(LineBox Box, string Text)>>? cells = null,
        float[]? tab = null)
    {
        int n = boxes.Count;
        int classes = ReadingOrder.BlockClasses.Length;
        var elements = new List<PageElement>();
        foreach (List<int> blk in SplitBlocks(order, same, n))
        {
            int cls0 = ReadingOrder.BlockClass(cls, classes, blk);
            var textsB = blk.Select(i => texts[i]).ToList();
            var element = new PageElement
            {
                Label = ReadingOrder.BlockClasses[cls0],
                Furniture = cls0 == Header || cls0 == Footer,
                Lines = textsB,
                Box = [blk.Min(i => boxes[i].X0), blk.Min(i => boxes[i].Y0), blk.Max(i => boxes[i].X1), blk.Max(i => boxes[i].Y1)],
                LineHeight = Py.Median(blk.Select(i => boxes[i].Y1 - boxes[i].Y0).ToArray()),
            };

            if (textsB.All(t => WholeMath().IsMatch(Py.Strip(t))))
            {
                elements.Add(Copy(element, ElementKind.Equation, text: string.Join("\n", textsB.Select(t =>
                {
                    int[] cp = Py.CodePoints(Py.Strip(t));
                    return "$$" + Py.FromCodePoints(cp[1..^1]) + "$$";
                }))));
            }
            else if (cls0 == Section)
            {
                elements.Add(Copy(element, ElementKind.Heading, text: JoinLines(textsB)));
            }
            else if (cls0 == Table && blk.Count > 1 && tab is not null
                && HeadsPlausible(blk.Select(i => boxes[i]).ToList(), SubTab(tab, n, blk)))
            {
                TableGrid grid = TableGridHeads(blk.Select(i => boxes[i]).ToList(), textsB, SubTab(tab, n, blk));
                elements.Add(Copy(element, ElementKind.Table, table: grid));
            }
            else if (cls0 == Table && blk.Count > 1)
            {
                var cb = new List<LineBox>();
                var ct = new List<string>();
                foreach (int i in blk)
                {
                    if (cells is not null && cells.TryGetValue(i, out var parts2) && parts2.Count > 0)
                    {
                        foreach (var (bx, tx) in parts2)
                        {
                            cb.Add(bx);
                            ct.Add(tx);
                        }
                    }
                    else
                    {
                        cb.Add(boxes[i]);
                        ct.Add(texts[i]);
                    }
                }

                var titles = new List<string>();
                TableGrid grid = TableGridOf(cb, ct, titles);
                elements.Add(Copy(element, ElementKind.Table, table: grid, titles: titles));
            }
            else if (cls0 == ListGroup)
            {
                var items = new List<List<string>>();
                foreach (string t in textsB)
                {
                    if (items.Count == 0 || Item().IsMatch(t))
                    {
                        items.Add([t]);
                    }
                    else
                    {
                        items[^1].Add(t);
                    }
                }

                elements.Add(Copy(element, ElementKind.List, items: items.Select(JoinLines).ToList()));
            }
            else
            {
                elements.Add(Copy(element, cls0 == Caption ? ElementKind.Caption : ElementKind.Paragraph, text: JoinLines(textsB)));
            }
        }

        return elements;
    }

    private static PageElement Copy(
        PageElement e,
        ElementKind kind,
        string text = "",
        IReadOnlyList<string>? items = null,
        TableGrid? table = null,
        IReadOnlyList<string>? titles = null) => new()
        {
            Label = e.Label,
            Kind = kind,
            Furniture = e.Furniture,
            Lines = e.Lines,
            Box = e.Box,
            LineHeight = e.LineHeight,
            Text = text,
            Items = items ?? [],
            Table = table,
            TableTitles = titles ?? [],
        };
}
