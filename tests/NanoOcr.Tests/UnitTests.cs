using NanoOcr.Imaging;
using NanoOcr.Models;
using NanoOcr.Pdf;
using NanoOcr.Pipeline;
using Xunit;

namespace NanoOcr.Tests;

/// <summary>
/// The Python package's own unit tests (<c>.reference/tests</c>), ported, plus checks of the
/// Python-semantics helpers the port leans on.
/// </summary>
public class UnitTests
{
    private static readonly Lazy<NanoOcrEngine> Engine = new(() => new NanoOcrEngine());

    private static string Inputs => Path.Combine(TestData.Documents, "inputs");

    private static LineBox Box(float x0, float y0, float x1, float y1) => new(x0, y0, x1, y1);

    // ------------------------------------------------------------------ Python semantics

    [Fact]
    public void PythonRoundingRoundsTheBinaryValue()
    {
        Assert.Equal(2.67, Py.Round(2.675, 2));
        Assert.Equal(0.1, Py.Round(0.05, 1));
        Assert.Equal(0.1, Py.Round(0.15, 1)); // 0.15 is 0.1499999… in binary
        Assert.Equal(2, Py.Round(2.5));
        Assert.Equal(4, Py.Round(3.5));
        Assert.Equal(-1.2, Py.Round(-1.25, 1)); // an exact tie: to even
        Assert.Equal(123.4, Py.Round(123.4499999, 1));
    }

    [Fact]
    public void PairwiseSumMatchesNumpy()
    {
        // float32 values whose sum depends on the summation order.
        var rng = new Random(3);
        float[] a = Enumerable.Range(0, 1000).Select(_ => (float)rng.NextDouble() * 1000f).ToArray();
        float sequential = 0f;
        foreach (float v in a)
        {
            sequential += v;
        }

        float pairwise = Py.PairwiseSum(a);
        Assert.Equal(a.Sum(v => (double)v), pairwise, 1);
        Assert.Equal(Py.PairwiseSum(a.AsSpan(0, 7)), a.Take(7).Aggregate(0f, (s, v) => s + v));
        _ = sequential;
    }

    // ------------------------------------------------------------------ test_textlayer.py

    [Fact]
    public void BadCharsCountReplacementAndPrivateUse()
    {
        Assert.Equal(0, TextLayer.BadCharCount("ok"));
        Assert.Equal(2, TextLayer.BadCharCount("a�"));
    }

    private static List<LineBox> Column(float x0, float x1) =>
        Enumerable.Range(0, 20).Select(i => Box(x0, 100 + (15 * i), x1, 110 + (15 * i))).ToList();

    [Fact]
    public void TwoColumnsInReadingOrderAreCoherent()
    {
        var boxes = Column(50, 290).Concat(Column(310, 550)).ToList();
        Assert.Equal(0, TextLayer.OrderBacktracks(boxes));
        Assert.True(TextLayer.PdfOrderCoherent(boxes, null));
    }

    [Fact]
    public void InterleavedColumnsAreNotCoherent()
    {
        List<LineBox> left = Column(50, 290), right = Column(310, 550);
        var boxes = left.Zip(right).SelectMany(p => new[] { p.First, p.Second }).ToList();
        Assert.Equal(0, TextLayer.OrderBacktracks(boxes));
        int n = boxes.Count;
        int[] key = Enumerable.Range(0, n).Select(i => ((i % 2) * 100) + (i / 2)).ToArray();
        var logits = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                logits[(i * n) + j] = Math.Sign(key[j] - key[i]) * 4f;
            }
        }

        Assert.False(TextLayer.PdfOrderCoherent(boxes, logits));
        int[] good = Enumerable.Range(0, n).OrderBy(i => key[i]).ToArray();
        var sub = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                sub[(i * n) + j] = logits[(good[i] * n) + good[j]];
            }
        }

        Assert.True(TextLayer.PdfOrderCoherent(good.Select(i => boxes[i]).ToList(), sub));
    }

    [Fact]
    public void LinesWrittenBottomUpAreNotCoherent()
    {
        var boxes = Enumerable.Range(0, 20).Select(i => Box(50, 400 - (15 * i), 500, 410 - (15 * i))).ToList();
        Assert.False(TextLayer.PdfOrderCoherent(boxes, null));
    }

    private static TextRun Run(string text, double x0, double y0, double y1, double w = 4.0)
    {
        int[] cp = Py.CodePoints(text);
        var chars = cp.Select((c, i) => (char.ConvertFromUtf32(c), new[] { x0 + (i * w), y0, x0 + ((i + 1) * w), y1 })).ToList();
        return new TextRun([x0, y0, x0 + (w * cp.Length), y1], text, chars);
    }

    private static string Joined(List<Token> toks) => string.Concat(toks.Select(t => t.Text));

    [Fact]
    public void RunsGoToTheSmallestContainingLineInXOrder()
    {
        var boxes = new List<LineBox> { Box(0, 0, 100, 20), Box(70, 2, 95, 18), Box(0, 30, 100, 50) };
        var runs = new List<TextRun> { Run("ab c", 10, 5, 15), Run("α", 2, 3, 13), Run("xy", 75, 5, 15), Run("far", 500, 5, 15) };
        (List<List<Token>> toks, int lost) = DocumentConverter.AssignRuns(runs, boxes, 1.0);
        Assert.Equal("α ab c", Joined(toks[0]));
        Assert.Equal(1, toks[0][0].ContentIndex);
        Assert.Equal("xy", Joined(toks[1]));
        Assert.Empty(toks[2]);
        Assert.Equal(3, lost);
    }

    [Fact]
    public void ARunAcrossTwoCellsSplitsBetweenThem()
    {
        string text = "12 34";
        var chars = text.Select((c, i) => (c.ToString(), new[] { i * 10.0, 0.0, (i * 10.0) + 8, 10.0 })).ToList();
        var run = new TextRun([0, 0, 48, 10], text, chars);
        var boxes = new List<LineBox> { Box(0, 0, 19, 10), Box(29, 0, 50, 10) };
        (List<List<Token>> toks, int lost) = DocumentConverter.AssignRuns([run], boxes, 1.0);
        Assert.Equal(["12", "34"], toks.Select(t => Joined(t).Trim()).ToArray());
        Assert.Equal(0, lost);
    }

    [Fact]
    public void RunsOfARightToLeftLineGoRightToLeft()
    {
        static TextRun RtlRun(string text, double x0)
        {
            var chars = text.Select((c, i) => (c.ToString(), new[] { x0 - ((i + 1) * 4.0), 5.0, x0 - (i * 4.0), 15.0 })).ToList();
            return new TextRun([x0 - (4.0 * text.Length), 5, x0, 15], text, chars);
        }

        (List<List<Token>> toks, _) = DocumentConverter.AssignRuns([RtlRun("עולם", 30), RtlRun("שלום", 90)], [Box(0, 0, 100, 20)], 1.0);
        Assert.Equal("שלום עולם", Joined(toks[0]));
    }

    [Fact]
    public void OcrOnlyReadsScriptsInItsVocab()
    {
        var conv = new DocumentConverter(Engine.Value);
        static TextRun L(string t) => new([0, 0, 1, 1], t, []);
        Assert.True(conv.Readable([L("hello world 123")]));
        Assert.False(conv.Readable([L("مرحبا بالعالم"), L("ok")]));
        Assert.True(conv.Readable([L("1234 ...")]));
        Assert.True(conv.Readable([L("中文文本")]));
    }

    [Fact]
    public void FastAverageLinkageMatchesTheBruteForce()
    {
        var rng = new Random(1);
        for (int trial = 0; trial < 100; trial++)
        {
            int n = rng.Next(1, 40);
            int[] g = Enumerable.Range(0, n).Select(_ => rng.Next(0, Math.Max(1, n / 4))).ToArray();
            var l = new float[n * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    l[(i * n) + j] = (g[i] == g[j] ? 2f : -2f) + (float)(Normal(rng) * 1.5);
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    float m = (l[(i * n) + j] + l[(j * n) + i]) / 2;
                    l[(i * n) + j] = l[(j * n) + i] = m;
                }
            }

            var fast = ReadingOrder.Cluster(l, n).Select(c => string.Join(",", c)).Order().ToList();
            Assert.Equal(Brute(l, n), fast);
        }

        static double Normal(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());

        static List<string> Brute(float[] logits, int n)
        {
            var s = logits.Select(v => (double)v).ToArray();
            for (int i = 0; i < n; i++)
            {
                s[(i * n) + i] = 0;
            }

            var size = Enumerable.Repeat(1.0, n).ToArray();
            var alive = Enumerable.Repeat(true, n).ToArray();
            var members = Enumerable.Range(0, n).Select(i => new List<int> { i }).ToArray();
            for (int step = 0; step < n - 1; step++)
            {
                double best = double.NegativeInfinity;
                int ba = 0, bb = 0;
                for (int a = 0; a < n; a++)
                {
                    for (int b = 0; b < n; b++)
                    {
                        if (a == b || !alive[a] || !alive[b])
                        {
                            continue;
                        }

                        double mean = s[(a * n) + b] / (size[a] * size[b]);
                        if (mean > best)
                        {
                            best = mean;
                            ba = a;
                            bb = b;
                        }
                    }
                }

                if (best <= 0)
                {
                    break;
                }

                for (int j = 0; j < n; j++)
                {
                    s[(ba * n) + j] += s[(bb * n) + j];
                }

                for (int j = 0; j < n; j++)
                {
                    s[(j * n) + ba] += s[(j * n) + bb];
                }

                s[(ba * n) + ba] = 0;
                size[ba] += size[bb];
                alive[bb] = false;
                members[ba].AddRange(members[bb]);
            }

            return Enumerable.Range(0, n).Where(i => alive[i]).Select(i => string.Join(",", members[i].Order())).Order().ToList();
        }
    }

    [Fact]
    public void SkewEstimateRecoversATilt()
    {
        var rng = new Random(0);
        var page = new GrayImage(1700, 2200);
        Array.Fill(page.Pixels, (byte)255);
        for (int y = 150; y < 2050; y += 45)
        {
            int x = 150;
            while (x < 1550)
            {
                int w = rng.Next(20, 90);
                for (int yy = y; yy < y + 18; yy++)
                {
                    for (int xx = x; xx < Math.Min(1700, x + w); xx++)
                    {
                        page[xx, yy] = 30;
                    }
                }

                x += w + rng.Next(10, 25);
            }
        }

        Assert.True(Math.Abs(NanoOcrEngine.EstimateSkew(page)) < 0.15);
        foreach (double tilt in new[] { 2.0, -1.3, 0.6 })
        {
            double est = NanoOcrEngine.EstimateSkew(NanoOcrEngine.RotatePage(page, tilt));
            Assert.True(Math.Abs(est + tilt) <= 0.15, $"tilt {tilt}: estimated {est}");
        }

        var blank = new GrayImage(600, 800);
        Array.Fill(blank.Pixels, (byte)255);
        Assert.Equal(0.0, NanoOcrEngine.EstimateSkew(blank));
    }

    [Fact]
    public void CjkLinesJoinWithoutASpaceAndBareBulletsStartItems()
    {
        Assert.Equal("或其中的任何部分，在 2020 年", Layout.JoinLines(["或其中的", "任何部分，在 2020 年"]));
        Assert.Equal("説明します。次に", Layout.JoinLines(["説明します。", "次に"]));
        Assert.Equal("한국어 문장 다음 줄", Layout.JoinLines(["한국어 문장", "다음 줄"]));
        Assert.Equal("tumorigenesis is", Layout.JoinLines(["tumori-", "genesis is"]));
        Assert.Equal("first line second", Layout.JoinLines(["first line", "second"]));

        // _ITEM through the list rendering: a bare bullet or "12." starts an item, "•text" does not.
        float[] cls = new float[4 * 10];
        float[] same = new float[16];
        Array.Fill(same, 5f);
        for (int i = 0; i < 4; i++)
        {
            cls[(i * 10) + 4] = 5f; // List-Group
        }

        var boxes = Enumerable.Range(0, 4).Select(i => Box(0, i * 20, 100, (i * 20) + 10)).ToList();
        string md = Layout.PageMarkdown(boxes, ["•", "first", "•text", "12. second"], [0, 1, 2, 3], cls, same);
        Assert.Equal("• first •text\n12. second", md);
    }

    [Fact]
    public void UnrotateBoxMapsADeskewedBoxBack()
    {
        var page = new GrayImage(600, 400);
        Array.Fill(page.Pixels, (byte)255);
        for (int y = 180; y < 220; y++)
        {
            for (int x = 100; x < 500; x++)
            {
                page[x, y] = 0;
            }
        }

        GrayImage tilted = NanoOcrEngine.RotatePage(page, -3.0);
        int minX = int.MaxValue, maxX = 0, minY = int.MaxValue, maxY = 0;
        for (int y = 0; y < 400; y++)
        {
            for (int x = 0; x < 600; x++)
            {
                if (tilted[x, y] < 128)
                {
                    minX = Math.Min(minX, x);
                    maxX = Math.Max(maxX, x);
                    minY = Math.Min(minY, y);
                    maxY = Math.Max(maxY, y);
                }
            }
        }

        double[] got = NanoOcrEngine.UnrotateBox([100, 180, 500, 220], 3.0, 600, 400);
        Assert.True(got[0] <= minX + 2 && got[2] >= maxX - 2);
        Assert.True(got[1] <= minY + 2 && got[3] >= maxY - 2);
        Assert.Equal([10.0, 10, 20, 20], NanoOcrEngine.UnrotateBox([10, 10, 20, 20], 0.0, 600, 400));
    }

    [Fact]
    public void MathTokensBackToText()
    {
        var toks = "we have ".Select(c => c.ToString()).Concat(["$", "x", "_", "{", "i", "}", "\\leq", "1", "$"]).Concat(" here".Select(c => c.ToString())).ToList();
        Assert.Equal("we have $x _ { i } \\leq 1$ here", MathReader.Detokenize(toks));
        Assert.Equal("$\\alpha$", MathReader.Detokenize(["$", "\\alpha"]));
        Assert.Equal("the bound", MathReader.Detokenize([" the", " bound"]));
    }

    [Fact]
    public void MathModelsAreOptional()
    {
        string dir = Directory.CreateTempSubdirectory().FullName;
        File.WriteAllBytes(Path.Combine(dir, "math_enc.onnx"), []);
        Assert.Null(MathReader.Load(ModelStore.FromDirectory(dir)));
        Assert.NotNull(MathReader.Load(ModelStore.Embedded));
    }

    // ------------------------------------------------------------------ test_inputs.py

    [Fact]
    public void FileTypeComesFromTheBytes()
    {
        Assert.True(PdfDocument.IsPdf(Path.Combine(Inputs, "five.pdf")));
        Assert.False(PdfDocument.IsPdf(Path.Combine(Inputs, "image_named.pdf")));
        var conv = new DocumentConverter(Engine.Value, mode: ConvertMode.Ocr);
        Assert.Single(conv.Convert(Path.Combine(Inputs, "image_named.pdf")));
    }

    [Fact]
    public void EveryTiffFrameIsAPage() => Assert.Equal(3, ImageIO.ReadPages(Path.Combine(Inputs, "three.tiff")).Count);

    [Fact]
    public void UnreadableFilesRaiseInputError()
    {
        var conv = new DocumentConverter(Engine.Value);
        foreach (string name in new[] { "garbage.pdf", "empty.png", "text.txt" })
        {
            Assert.Throws<InputException>(() => conv.Convert(Path.Combine(Inputs, name)).ToList());
        }

        var e = Assert.Throws<InputException>(() => conv.Convert(Path.Combine(Inputs, "missing.pdf")).ToList());
        Assert.Contains("no such file", e.Message);
    }

    [Fact]
    public void PasswordProtectedPdf()
    {
        var conv = new DocumentConverter(Engine.Value, mode: ConvertMode.Ocr);
        string locked = Path.Combine(Inputs, "locked.pdf");
        var e = Assert.Throws<InputException>(() => conv.Convert(locked).ToList());
        Assert.Contains("password", e.Message);
        Assert.Single(conv.Convert(locked, password: "pw"));
    }

    [Fact]
    public void AFailingPageDoesNotEndTheDocument()
    {
        Page page = DocumentConverter.Safe(() => throw new InvalidOperationException("stub failure"));
        Assert.Equal(PageSources.Error, page.Source);
        Assert.Contains("stub failure", page.Note);
        Assert.Equal("", page.Markdown);
        Assert.Equal(0.0, page.Timings["total_ms"]);
    }

    [Fact]
    public void HugePagesRenderBelowThePixelCap()
    {
        var conv = new DocumentConverter(Engine.Value, mode: ConvertMode.Ocr);
        using PdfDocument doc = PdfDocument.Open(Path.Combine(Inputs, "poster.pdf"));
        using PdfPage big = doc.GetPage(0);
        using PdfPage letter = doc.GetPage(1);
        double kb = conv.PageScale(big);
        Assert.True(Math.Pow(14400 * kb, 2) <= DocumentConverter.MaxPagePixels * 1.01);
        Assert.Equal(200 / 72.0, conv.PageScale(letter), 9);
    }

    [Fact]
    public void ManyFilesKeepInputOrderAndReportErrors()
    {
        using var ocr = new NanoOCR(new NanoOcrOptions { Mode = ConvertMode.Ocr });
        string[] paths = [Path.Combine(Inputs, "five.pdf"), Path.Combine(Inputs, "missing.pdf"), Path.Combine(Inputs, "garbage.pdf"), Path.Combine(Inputs, "three.tiff")];
        var results = ocr.PredictMany(paths, cores: 4, maxPages: 3).ToList();
        Assert.Equal(paths, results.Select(r => r.Path).ToArray());
        Assert.Equal([0, 1, 2], results[0].Results!.Select(r => r.PageIndex).ToArray());
        Assert.NotNull(results[1].Error);
        Assert.NotNull(results[2].Error);
        Assert.Equal(3, results[3].Results!.Count);
    }
}
