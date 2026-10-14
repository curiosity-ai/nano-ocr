using System.Text;

namespace NanoOcr.Pipeline;

/// <summary>
/// Rejoins a table that a page break split in two — PaddleOCR's <c>MergeTablesAcrossPages</c>
/// (PaddleX <c>layout_parsing/merge_table.py</c>, <c>merge_tables_across_pages</c>) on NanoOCR's
/// table grids.
/// </summary>
/// <remarks>
/// <para>
/// The last table of one page and the first of the next are one table when only page furniture
/// and footnotes follow the first and only page furniture precedes the second, their widths agree
/// within 10%, and either their column counts agree or the first's last row lines up with the
/// second's first data row. Rows the second repeats from the top of the first (a header printed
/// again) are dropped; the rest move into the first, and the second renders as nothing.
/// </para>
/// <para>
/// PaddleOCR compares HTML it re-parses; NanoOCR keeps the grid the table was assembled into, so
/// the same tests run on cells directly. One test is NanoOCR's own: a second table with a title
/// row of its own (<see cref="PageElement.TableTitles"/>) starts a new table.
/// </para>
/// </remarks>
public static class TableMerger
{
    /// <summary>Header rows compared at most (<c>max_header_rows</c>).</summary>
    public const int MaxHeaderRows = 5;

    /// <summary>Merges tables split across consecutive pages.</summary>
    /// <param name="pages">Each page's elements, in document order.</param>
    /// <returns>How many tables were merged into the one before.</returns>
    public static int Apply(IReadOnlyList<IReadOnlyList<PageElement>> pages)
    {
        int merged = 0;

        // Backwards, so a table running over three pages collapses into the first.
        for (int i = pages.Count - 1; i > 0; i--)
        {
            IReadOnlyList<PageElement> previous = pages[i - 1];
            IReadOnlyList<PageElement> current = pages[i];
            int before = LastIndex(previous, e => e.Kind == ElementKind.Table && !e.MergedIntoPrevious);
            int after = FirstIndex(current, e => e.Kind == ElementKind.Table);
            if (before < 0 || after < 0 || !CanMerge(previous, before, current, after))
            {
                continue;
            }

            TableGrid first = previous[before].Table!, second = current[after].Table!;
            int skip = HeaderRows(first, second).Rows;
            for (int r = skip; r < second.Rows.Count; r++)
            {
                first.Rows.Add(second.Rows[r] with { Header = false });
            }

            current[after].MergedIntoPrevious = true;
            merged++;
        }

        return merged;
    }

    private static bool CanMerge(IReadOnlyList<PageElement> previousPage, int before, IReadOnlyList<PageElement> currentPage, int after)
    {
        PageElement a = previousPage[before], b = currentPage[after];
        if (b.TableTitles.Count > 0)
        {
            return false;
        }

        int wa = (int)a.Box[2] - (int)a.Box[0], wb = (int)b.Box[2] - (int)b.Box[0];
        if (wa == 0 || wb == 0 || Math.Abs(wb - wa) / (double)Math.Min(wa, wb) >= 0.1)
        {
            return false;
        }

        for (int i = before + 1; i < previousPage.Count; i++)
        {
            if (!previousPage[i].Furniture && previousPage[i].Label != "Footnote")
            {
                return false;
            }
        }

        for (int i = 0; i < after; i++)
        {
            if (!currentPage[i].Furniture)
            {
                return false;
            }
        }

        TableGrid ga = a.Table!, gb = b.Table!;
        if (ga.Rows.Count == 0 || gb.Rows.Count == 0)
        {
            return false;
        }

        return ga.Columns == gb.Columns || RowsLineUp(ga, gb);
    }

    /// <summary>
    /// How many rows at the top of both tables are the same (text with whitespace removed and
    /// full-width forms folded, and spans), and whether they count as a header.
    /// </summary>
    public static (int Rows, bool Match) HeaderRows(TableGrid first, TableGrid second)
    {
        int limit = Math.Min(Math.Min(first.Rows.Count, second.Rows.Count), MaxHeaderRows);
        int rows = 0;
        bool match = true;
        for (int i = 0; i < limit; i++)
        {
            IReadOnlyList<TableCell> a = first.Rows[i].Cells, b = second.Rows[i].Cells;
            bool same = a.Count == b.Count
                && a.Zip(b).All(p => Normalize(p.First.Text) == Normalize(p.Second.Text) && p.First.ColSpan == p.Second.ColSpan);
            if (!same)
            {
                match = rows > 0;
                break;
            }

            rows++;
        }

        return (rows, rows != 0 && match);
    }

    /// <summary>Whether the first table's last row lines up with the second's first data row.</summary>
    private static bool RowsLineUp(TableGrid first, TableGrid second)
    {
        int header = HeaderRows(first, second).Rows;
        if (second.Rows.Count <= header)
        {
            return false;
        }

        TableRow last = first.Rows[^1], data = second.Rows[header];
        return last.Cells.Sum(c => c.ColSpan) == data.Cells.Sum(c => c.ColSpan) || last.Cells.Count == data.Cells.Count;
    }

    /// <summary><c>full_to_half</c> with whitespace removed: a full-width header matches its ASCII twin.</summary>
    public static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (!char.IsWhiteSpace(c))
            {
                sb.Append(c is >= '！' and <= '～' ? (char)(c - 0xFEE0) : c);
            }
        }

        return sb.ToString();
    }

    private static int FirstIndex(IReadOnlyList<PageElement> list, Func<PageElement, bool> f)
    {
        for (int i = 0; i < list.Count; i++)
        {
            if (f(list[i]))
            {
                return i;
            }
        }

        return -1;
    }

    private static int LastIndex(IReadOnlyList<PageElement> list, Func<PageElement, bool> f)
    {
        for (int i = list.Count - 1; i >= 0; i--)
        {
            if (f(list[i]))
            {
                return i;
            }
        }

        return -1;
    }
}
