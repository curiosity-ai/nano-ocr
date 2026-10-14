using System.Text.RegularExpressions;

namespace NanoOcr.Pipeline;

/// <summary>
/// How deep each heading of a document sits — PaddleOCR's <c>AssignTitleLevels</c> (PaddleX
/// <c>layout_parsing/title_level.py</c>, <c>assign_levels_to_parsing_res</c>) on NanoOCR's
/// headings.
/// </summary>
/// <remarks>
/// <para>
/// A numbered heading gets its depth from three signals and takes whichever two agree: the depth
/// its numbering implies (<c>1.2.1</c> is three deep), the rank of its numbering style by first
/// appearance in the document, and the rank of its text size among the headings'. A recognised
/// section name (Abstract, References, 参考文献 …) is level one; any other heading takes its size
/// rank alone.
/// </para>
/// <para>
/// The size is where NanoOCR differs from PaddleOCR's input: PaddleOCR divides a title block's
/// height by its line count; NanoOCR has every line's box, and uses the heading's median line
/// height (<see cref="PageElement.LineHeight"/>), rounded to a pixel. The sizes are grouped into at
/// most four levels by an exact one-dimensional k-means (as PaddleOcrSharp does in place of
/// scikit-learn's seeded <c>KMeans</c>), the tallest text the shallowest — and, because box
/// heights scatter by a pixel or two for one type size, only into levels whose sizes differ by
/// <see cref="DistinctSize"/>.
/// </para>
/// </remarks>
public static partial class TitleLevels
{
    [GeneratedRegex(@"^\s*([IVX]+)(?:[\.．\)\s]|$)", RegexOptions.IgnoreCase)]
    private static partial Regex Roman();

    [GeneratedRegex(@"^\s*([A-Z])(?:[\.．\)\s])", RegexOptions.IgnoreCase)]
    private static partial Regex Letter();

    [GeneratedRegex(@"^\s*(\d+(?:\.\d+)*)(?![）)])(?:[\.]?\s*|(?=[A-Z]))")]
    private static partial Regex NumberList();

    [GeneratedRegex(@"^\s*(?:[\(（])?(\d+(?:\.\d+)*)[\)）]")]
    private static partial Regex BracketedNumberList();

    [GeneratedRegex(@"^\s*(?:第|[（\(])?([一二三四五六七八九十]{1,2})(?:[章节篇卷部条题讲课回）\)]|(?![a-zA-Z一-龥]))", RegexOptions.IgnoreCase)]
    private static partial Regex ChineseNumber();

    /// <summary>
    /// Two heading sizes are different levels only when the larger is at least this much larger
    /// (PaddleOCR's k-means takes every distinct height as a candidate level).
    /// </summary>
    public const double DistinctSize = 1.12;

    /// <summary>Section names that are level one however they are set (<c>SPECIAL_KEYWORDS</c>).</summary>
    private static readonly HashSet<string> SpecialKeywords = new(StringComparer.Ordinal)
    {
        "ABSTRACT", "SUMMARY", "RESUME", "绪论", "引言", "CONTENTS", "REFERENCES", "REFERENCE",
        "参考文献", "APPENDIX", "APPENDICES", "附录", "ACKNOWLEDGMENTS", "INTRODUCTION",
        "BACKGROUNDANDRELATEDWORK", "BACKGROUND", "RELATEDWORK", "THEORETICALMODELS", "DATA",
        "METHOD", "METHODS", "METHODOLOGY", "TOPICANALYSIS", "RESULT", "RESULTS", "DISCUSSION",
        "CONCLUSIONS", "CONCLUSION", "LIMITATIONS", "研究背景", "相关工作", "研究方法", "实验结果",
        "讨论", "结论", "致谢", "目录",
    };

    /// <summary>
    /// Sets <see cref="PageElement.TitleLevel"/> on every heading of the document except its title.
    /// </summary>
    /// <param name="pages">Each page's elements, in document order.</param>
    public static void Apply(IReadOnlyList<IReadOnlyList<PageElement>> pages)
    {
        var headings = pages.SelectMany(p => p).Where(e => e.Kind == ElementKind.Heading && !e.DocTitle && !e.Furniture).ToList();
        if (headings.Count == 0)
        {
            return;
        }

        int[] levels = Levels(
            headings.Select(e => e.Text).ToList(),
            headings.Select(e => (int)MathF.Round(e.LineHeight)).ToList());
        for (int i = 0; i < headings.Count; i++)
        {
            headings[i].TitleLevel = levels[i];
        }
    }

    /// <summary>The depth of each heading, given its text and its text height, in document order.</summary>
    /// <param name="contents">Each heading's text.</param>
    /// <param name="heights">Each heading's text height in pixels.</param>
    public static int[] Levels(IReadOnlyList<string> contents, IReadOnlyList<int> heights)
    {
        int count = contents.Count;
        var symbols = contents.Select(SymbolAndLevel).ToArray();
        IReadOnlyDictionary<int, int> clusterMap = ClusterHeights(heights);

        // Numbering styles rank by first appearance: the style a document opens with outranks one
        // it only reaches later.
        var order = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach ((string? symbol, int level) in symbols)
        {
            if (level > 0 && symbol is not null && !order.ContainsKey(symbol))
            {
                order[symbol] = order.Count + 1;
            }
        }

        int[] levels = new int[count];
        int firstNumberLevel = 0;
        for (int i = 0; i < count; i++)
        {
            (string? symbol, int level) = symbols[i];
            int clusterLevel = clusterMap[heights[i]];
            if (level > 0 && symbol is not null)
            {
                int relative;
                if (symbol == "NUM_LIST")
                {
                    if (firstNumberLevel != 0)
                    {
                        relative = order[symbol] + (level - firstNumberLevel);
                    }
                    else
                    {
                        firstNumberLevel = level;
                        relative = order[symbol];
                    }
                }
                else
                {
                    relative = order[symbol];
                }

                levels[i] = Vote(level, relative, clusterLevel) ?? relative;
            }
            else if (SpecialKeywords.Contains(Keyword(contents[i])))
            {
                levels[i] = 1;
            }
            else
            {
                levels[i] = clusterLevel;
            }
        }

        return levels;
    }

    /// <summary>The value at least two of the three votes share, if any.</summary>
    private static int? Vote(int first, int second, int third) =>
        first == second || first == third ? first : second == third ? second : null;

    /// <summary>
    /// The numbering style a heading opens with and the depth it implies (upstream's order of
    /// tests: a bracketed number before a bare one, a Roman numeral before a letter).
    /// </summary>
    /// <param name="content">The heading's text.</param>
    public static (string? Symbol, int Level) SymbolAndLevel(string content)
    {
        string text = content.Trim();
        if (BracketedNumberList().IsMatch(text))
        {
            return ("NUM_LIST_BRACKET", 4);
        }

        if (Roman().IsMatch(text))
        {
            return ("ROMAN", 1);
        }

        if (ChineseNumber().IsMatch(text))
        {
            return ("CHINESE_NUM", 1);
        }

        if (Letter().IsMatch(text))
        {
            return ("LETTER", 2);
        }

        Match numbers = NumberList().Match(text);
        return numbers.Success ? ("NUM_LIST", numbers.Groups[1].Value.Count(c => c == '.') + 1) : (null, -1);
    }

    private static string Keyword(string content) =>
        content.ToUpperInvariant().Trim().TrimEnd('：', ':', ' ').Replace(" ", string.Empty, StringComparison.Ordinal);

    /// <summary>
    /// Groups heights into at most <paramref name="clusters"/> levels, tallest first: an exact
    /// one-dimensional k-means by dynamic programming over the sorted heights (every cluster of an
    /// optimum is a contiguous run), with the most clusters whose centres differ by
    /// <see cref="DistinctSize"/>, each height then mapped to its nearest centre.
    /// </summary>
    /// <param name="heights">Every heading's height, repeats included.</param>
    /// <param name="clusters">The most levels to use.</param>
    public static IReadOnlyDictionary<int, int> ClusterHeights(IReadOnlyList<int> heights, int clusters = 4)
    {
        int[] unique = [.. heights.Distinct().Order()];
        if (unique.Length == 0)
        {
            return new Dictionary<int, int>();
        }

        int k = Math.Min(clusters, unique.Length);
        double[] values = [.. heights.Order().Select(h => (double)h)];
        int n = values.Length;
        double[] sum = new double[n + 1];
        double[] squares = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            sum[i + 1] = sum[i] + values[i];
            squares[i + 1] = squares[i] + (values[i] * values[i]);
        }

        double Cost(int from, int to)
        {
            int length = to - from;
            if (length <= 0)
            {
                return 0;
            }

            double total = sum[to] - sum[from];
            return squares[to] - squares[from] - (total * total / length);
        }

        double[,] best = new double[k + 1, n + 1];
        int[,] split = new int[k + 1, n + 1];
        for (int i = 1; i <= n; i++)
        {
            best[0, i] = double.PositiveInfinity;
        }

        for (int j = 1; j <= k; j++)
        {
            for (int i = j; i <= n; i++)
            {
                best[j, i] = double.PositiveInfinity;
                for (int m = j - 1; m < i; m++)
                {
                    double candidate = best[j - 1, m] + Cost(m, i);
                    if (candidate < best[j, i])
                    {
                        best[j, i] = candidate;
                        split[j, i] = m;
                    }
                }
            }
        }

        // The most clusters whose centres all differ by DistinctSize: a detector's boxes put the
        // same type size a pixel or two apart, which exact k-means would otherwise call levels.
        List<double> centres = [];
        for (int kk = k; kk >= 1; kk--)
        {
            centres = Centres(kk);
            if (kk == 1 || centres.Zip(centres.Skip(1)).All(p => p.Second >= p.First * DistinctSize))
            {
                break;
            }
        }

        List<double> Centres(int kk)
        {
            var c = new List<double>(kk);
            int end = n;
            for (int j = kk; j >= 1; j--)
            {
                int start = split[j, end];
                c.Add((sum[end] - sum[start]) / (end - start));
                end = start;
            }

            c.Reverse();
            return c;
        }

        var map = new Dictionary<int, int>(unique.Length);
        foreach (int height in unique)
        {
            int nearest = 0;
            double closest = Math.Abs(height - centres[0]);
            for (int c = 1; c < centres.Count; c++)
            {
                double distance = Math.Abs(height - centres[c]);
                if (distance < closest)
                {
                    closest = distance;
                    nearest = c;
                }
            }

            map[height] = centres.Count - nearest;
        }

        return map;
    }
}
