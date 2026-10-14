namespace NanoOcr.Pipeline;

/// <summary>
/// The reading-order stage's torch-free parts (port of <c>indra_ocr/order_post.py</c>): line
/// features for the order model, decoding its pairwise logits into an order, the XY-cut and
/// top-down baselines, and the average-linkage clustering behind blocks.
/// </summary>
public static class ReadingOrder
{
    /// <summary>Geometric features per line.</summary>
    public const int GeomFeatures = 10;

    /// <summary>Text features per line.</summary>
    public const int TextFeatures = 6;

    /// <summary>Features per line.</summary>
    public const int Features = GeomFeatures + TextFeatures;

    /// <summary>Block classes of the layout heads, in class-id order (chandra block labels).</summary>
    public static readonly string[] BlockClasses =
    [
        "Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group", "Table", "Caption", "Footnote",
        "Table-Of-Contents", "Form",
    ];

    /// <summary>Blocks whose lines read row by row.</summary>
    public static readonly HashSet<string> GridLabels = ["Table", "Form", "Table-Of-Contents"];

    private const string Bullets = "•-–▪◦*·→";

    /// <summary><c>line_features</c>: <c>[N, 16]</c> float32 features of line boxes on a page.</summary>
    public static float[] LineFeatures(IReadOnlyList<LineBox> boxes, double pageW, double pageH, IReadOnlyList<string>? texts)
    {
        int n = boxes.Count;
        var f = new float[n * Features];
        if (n == 0)
        {
            return f;
        }

        var hs = new float[n];
        for (int i = 0; i < n; i++)
        {
            hs[i] = MathF.Max(boxes[i].Y1 - boxes[i].Y0, 1e-3f);
        }

        double medH = Py.Median(hs);
        float fw = (float)pageW, fh = (float)pageH, fw2 = (float)(2 * pageW), fh2 = (float)(2 * pageH);
        float medHf = (float)medH;
        for (int i = 0; i < n; i++)
        {
            LineBox b = boxes[i];
            float w = MathF.Max(b.X1 - b.X0, 1e-3f);
            float h = hs[i];
            int o = i * Features;
            f[o] = b.X0 / fw;
            f[o + 1] = b.Y0 / fh;
            f[o + 2] = b.X1 / fw;
            f[o + 3] = b.Y1 / fh;
            f[o + 4] = (b.X0 + b.X1) / fw2;
            f[o + 5] = (b.Y0 + b.Y1) / fh2;
            f[o + 6] = w / fw;
            f[o + 7] = MathF.Log(h / medHf);
            f[o + 8] = MathF.Log(w / h) / 4f;
            f[o + 9] = (float)(medH / pageH * 20.0);
            if (texts is null)
            {
                continue;
            }

            string t = Py.Strip(texts[i]);
            if (t.Length == 0)
            {
                continue;
            }

            int[] cp = Py.CodePoints(t);
            f[o + 10] = (float)(Math.Log(1 + cp.Length) / 5.0);
            f[o + 11] = (float)((double)cp.Count(Py.IsDigit) / cp.Length);
            bool bullet = Bullets.Contains(char.ConvertFromUtf32(cp[0]), StringComparison.Ordinal);
            string head = Py.FromCodePoints(cp.Take(3)).TrimEnd('.', ')');
            bool numbered = head.Length > 0 && Py.CodePoints(head).All(Py.IsDigit) && cp.Length > 3;
            f[o + 12] = bullet || numbered ? 1f : 0f;
            f[o + 13] = cp[^1] == '-' ? 1f : 0f;
            f[o + 14] = ".!?:;".Contains((char)Math.Min(cp[^1], 0xFFFF)) && cp[^1] < 0x10000 ? 1f : 0f;
            f[o + 15] = Py.IsUpper(cp[0]) ? 1f : 0f;
        }

        return f;
    }

    /// <summary>
    /// <c>decode_order</c>: an order from a pairwise logit matrix (<c>a[i, j] &gt; 0</c>: i before j).
    /// Starts from the row-sum ranking, then moves single lines to the position that most increases
    /// the summed log-probability of the pairwise relations.
    /// </summary>
    public static List<int> DecodeOrder(float[] a, int n, int sweeps = 4)
    {
        if (n <= 1)
        {
            return Enumerable.Range(0, n).ToList();
        }

        // lp = -logaddexp(0, -a) = log P(i before j), float32.
        var lp = new float[n * n];
        for (int i = 0; i < lp.Length; i++)
        {
            lp[i] = -LogAddExp0(-a[i]);
        }

        var score = new float[n];
        var colSum = new float[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                colSum[j] += lp[(i * n) + j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            score[i] = -Py.PairwiseSum(lp.AsSpan(i * n, n)) + colSum[i];
        }

        List<int> order = Py.ArgSort(score).ToList();
        var gains = new float[n];
        var before = new float[n - 1];
        var after = new float[n - 1];
        for (int s = 0; s < sweeps; s++)
        {
            bool improved = false;
            foreach (int x in order.ToArray())
            {
                int cur = order.IndexOf(x);
                var rest = new List<int>(order);
                rest.RemoveAt(cur);
                for (int k = 0; k < n - 1; k++)
                {
                    before[k] = lp[(rest[k] * n) + x];
                    after[k] = lp[(x * n) + rest[k]];
                }

                // gains[k] = sum(before[:k]) + sum(after[k:]), as float32 cumulative sums.
                var head = new float[n];
                float acc = 0f;
                head[0] = 0f;
                for (int k = 0; k < n - 1; k++)
                {
                    acc += before[k];
                    head[k + 1] = acc;
                }

                var tail = new float[n];
                acc = 0f;
                tail[n - 1] = 0f;
                for (int k = n - 2; k >= 0; k--)
                {
                    acc += after[k];
                    tail[k] = acc;
                }

                for (int k = 0; k < n; k++)
                {
                    gains[k] = head[k] + tail[k];
                }

                int best = Py.ArgMax(gains);
                if (gains[best] > gains[cur] + 1e-6f)
                {
                    rest.Insert(best, x);
                    order = rest;
                    improved = true;
                }
            }

            if (!improved)
            {
                break;
            }
        }

        return order;
    }

    /// <summary><c>np.logaddexp(0, y)</c> in float32 (numpy's <c>npy_logaddexpf</c>).</summary>
    private static float LogAddExp0(float y)
    {
        if (y == 0f)
        {
            return 0.6931472f;
        }

        float tmp = 0f - y;
        return tmp > 0 ? (float)Log1p(MathF.Exp(-tmp)) : y + (float)Log1p(MathF.Exp(tmp));
    }

    /// <summary>An accurate <c>log(1 + x)</c>.</summary>
    internal static double Log1p(double x)
    {
        double u = 1.0 + x;
        return u == 1.0 ? x : Math.Log(u) * x / (u - 1.0);
    }

    /// <summary><c>order_top_down</c>: rows top to bottom (centres within half a line height share a row), left to right.</summary>
    public static List<int> TopDown(IReadOnlyList<LineBox> boxes)
    {
        int n = boxes.Count;
        if (n == 0)
        {
            return [];
        }

        var cy = new float[n];
        var hs = new float[n];
        for (int i = 0; i < n; i++)
        {
            cy[i] = (boxes[i].Y0 + boxes[i].Y1) / 2f;
            hs[i] = boxes[i].Y1 - boxes[i].Y0;
        }

        // round(cy / max(h * 0.5, 1e-3)): numpy divides in float32 either way.
        float half = Py.Median(hs) * 0.5f;
        float div = 1e-3 > half ? 1e-3f : half;
        return Enumerable.Range(0, n)
            .OrderBy(i => Py.Round(cy[i] / div))
            .ThenBy(i => boxes[i].X0)
            .ToList();
    }

    /// <summary>
    /// <c>order_xycut</c>: recursive XY-cut at the widest whitespace gap on either axis (upper or
    /// left part first), falling back to <see cref="TopDown"/> where no gap exists.
    /// </summary>
    public static List<int> XyCut(IReadOnlyList<LineBox> boxes)
    {
        List<int> Rec(List<int> ids)
        {
            if (ids.Count <= 1)
            {
                return [.. ids];
            }

            var b = ids.Select(i => boxes[i]).ToList();
            (float gy, int ky, int[] oy) = BestCut(b, vertical: true);
            (float gx, int kx, int[] ox) = BestCut(b, vertical: false);
            if (ky < 0 && kx < 0)
            {
                return TopDown(b).Select(i => ids[i]).ToList();
            }

            (int k, int[] o) = gy >= gx ? (ky, oy) : (kx, ox);
            var first = Rec(o.Take(k).Select(i => ids[i]).ToList());
            first.AddRange(Rec(o.Skip(k).Select(i => ids[i]).ToList()));
            return first;
        }

        return Rec(Enumerable.Range(0, boxes.Count).ToList());
    }

    private static (float Gap, int K, int[] Order) BestCut(List<LineBox> b, bool vertical)
    {
        float[] lo = b.Select(x => vertical ? x.Y0 : x.X0).ToArray();
        float[] hi = b.Select(x => vertical ? x.Y1 : x.X1).ToArray();
        int[] o = Py.ArgSort(lo);
        float end = hi[o[0]];
        float bestGap = 0f;
        int bestK = -1;
        for (int k = 1; k < o.Length; k++)
        {
            float gap = lo[o[k]] - end;
            if (gap > bestGap)
            {
                bestGap = gap;
                bestK = k;
            }

            end = MathF.Max(end, hi[o[k]]);
        }

        return (bestGap, bestK, o);
    }

    /// <summary>
    /// <c>_cluster</c>: average-linkage clustering of pairwise "same group" logits. Merges the two
    /// groups with the highest mean logit while that mean is positive. Returns the groups, each
    /// sorted, in order of their smallest member.
    /// </summary>
    public static List<List<int>> Cluster(float[] logits, int n)
    {
        if (n == 0)
        {
            return [];
        }

        var s = new double[n * n];
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = logits[i];
        }

        for (int i = 0; i < n; i++)
        {
            s[(i * n) + i] = 0;
        }

        var size = new double[n];
        Array.Fill(size, 1.0);
        var alive = new bool[n];
        Array.Fill(alive, true);
        var members = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            members[i] = [i];
        }

        var m = (double[])s.Clone();
        for (int i = 0; i < n; i++)
        {
            m[(i * n) + i] = double.NegativeInfinity;
        }

        var best = new int[n];
        var bestV = new double[n];
        for (int i = 0; i < n; i++)
        {
            best[i] = Py.ArgMax(m.AsSpan(i * n, n));
            bestV[i] = m[(i * n) + best[i]];
        }

        var row = new double[n];
        for (int step = 0; step < n - 1; step++)
        {
            int a = Py.ArgMax(bestV);
            int b = best[a];
            if (bestV[a] <= 0)
            {
                break;
            }

            (a, b) = (Math.Min(a, b), Math.Max(a, b));
            for (int j = 0; j < n; j++)
            {
                s[(a * n) + j] += s[(b * n) + j];
            }

            for (int j = 0; j < n; j++)
            {
                s[(j * n) + a] += s[(j * n) + b];
            }

            s[(a * n) + a] = 0;
            size[a] += size[b];
            alive[b] = false;
            members[a].AddRange(members[b]);
            for (int j = 0; j < n; j++)
            {
                row[j] = alive[j] && j != a ? s[(a * n) + j] / (size[a] * size[j]) : double.NegativeInfinity;
            }

            for (int j = 0; j < n; j++)
            {
                m[(a * n) + j] = row[j];
                m[(j * n) + a] = row[j];
            }

            for (int j = 0; j < n; j++)
            {
                m[(b * n) + j] = double.NegativeInfinity;
                m[(j * n) + b] = double.NegativeInfinity;
            }

            bestV[b] = double.NegativeInfinity;
            best[a] = Py.ArgMax(row);
            bestV[a] = row[best[a]];
            for (int j = 0; j < n; j++)
            {
                if (alive[j] && j != a && (best[j] == a || best[j] == b))
                {
                    best[j] = Py.ArgMax(m.AsSpan(j * n, n));
                    bestV[j] = m[(j * n) + best[j]];
                }
            }

            for (int j = 0; j < n; j++)
            {
                if (alive[j] && j != a && row[j] > bestV[j])
                {
                    best[j] = a;
                    bestV[j] = row[j];
                }
            }
        }

        var groups = new List<List<int>>();
        for (int i = 0; i < n; i++)
        {
            if (alive[i])
            {
                groups.Add(members[i].Order().ToList());
            }
        }

        return groups;
    }

    /// <summary>The block class of a group of lines: the argmax of the summed class logits.</summary>
    public static int BlockClass(float[] cls, int classes, IReadOnlyList<int> lines)
    {
        var sum = new float[classes];
        foreach (int i in lines)
        {
            for (int c = 0; c < classes; c++)
            {
                sum[c] += cls[(i * classes) + c];
            }
        }

        return Py.ArgMax(sum);
    }

    /// <summary>
    /// <c>decode_blocks_order</c>: blocks from clustering the same-block logits; lines inside a
    /// block by XY-cut (row by row in grid blocks); blocks by the mean pairwise order logit.
    /// </summary>
    public static List<int> DecodeBlocksOrder(float[] a, float[] same, IReadOnlyList<LineBox> boxes, float[]? cls, int classes)
    {
        int n = boxes.Count;
        if (n <= 1)
        {
            return Enumerable.Range(0, n).ToList();
        }

        List<List<int>> blocks = Cluster(same, n);
        var grid = new HashSet<int>(GridLabels.Select(l => Array.IndexOf(BlockClasses, l)).Where(i => i >= 0));
        var inner = new List<List<int>>();
        foreach (List<int> b in blocks)
        {
            var bx = b.Select(i => boxes[i]).ToList();
            List<int> local = cls is not null && grid.Contains(BlockClass(cls, classes, b)) ? TopDown(bx) : XyCut(bx);
            inner.Add(local.Select(k => b[k]).ToList());
        }

        // Mean order logit between blocks: G a Gᵀ with G the membership matrix scaled by 1/size.
        int mb = blocks.Count;
        var ga = new double[mb * n];
        for (int k = 0; k < mb; k++)
        {
            double wk = 1.0 / blocks[k].Count;
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                foreach (int i in blocks[k])
                {
                    acc += wk * a[(i * n) + j];
                }

                ga[(k * n) + j] = acc;
            }
        }

        var g = new double[mb * mb];
        for (int p = 0; p < mb; p++)
        {
            for (int q = 0; q < mb; q++)
            {
                double wq = 1.0 / blocks[q].Count;
                double acc = 0;
                foreach (int j in blocks[q])
                {
                    acc += ga[(p * n) + j] * wq;
                }

                g[(p * mb) + q] = acc;
            }
        }

        var ab = new float[mb * mb];
        for (int p = 0; p < mb; p++)
        {
            for (int q = 0; q < mb; q++)
            {
                double up = p < q ? g[(p * mb) + q] : 0;
                double down = q < p ? g[(q * mb) + p] : 0;
                ab[(p * mb) + q] = (float)(up - down);
            }
        }

        var result = new List<int>(n);
        foreach (int k in DecodeOrder(ab, mb))
        {
            result.AddRange(inner[k]);
        }

        return result;
    }
}
