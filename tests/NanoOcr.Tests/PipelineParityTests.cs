using System.Text.Json;
using NanoOcr.Imaging;
using NanoOcr.Onnx;
using NanoOcr.Pdf;
using NanoOcr.Pipeline;
using Xunit;

namespace NanoOcr.Tests;

/// <summary>
/// The C# pipeline against the Python reference, stage by stage, on the test corpus (fixtures from
/// <c>tools/reference/dump_pipeline.py</c>). Each stage is fed the reference's own input where
/// that isolates it — recognition reads the reference's boxes — and the whole page is compared at
/// the end.
/// </summary>
public class PipelineParityTests(ITestOutputHelper output)
{
    private static readonly Lazy<NanoOcrEngine> Engine = new(() => new NanoOcrEngine(new EngineOptions { Markdown = MarkdownOptions.Reference }));
    private static readonly Lazy<NanoOcrEngine> MathEngine = new(() => new NanoOcrEngine(new EngineOptions { Math = true, Markdown = MarkdownOptions.Reference }));

    public static TheoryData<string> Images => ["page_two_column", "page_rotated", "page_skewed", "page_chinese", "page_math"];

    public static TheoryData<string> Pdfs => ["digital", "scanned"];

    private static JsonDocument? Reference(string name, string suffix = ".json")
    {
        string path = Path.Combine(TestData.Pipeline, name + suffix);
        return File.Exists(path) ? JsonDocument.Parse(File.ReadAllText(path)) : null;
    }

    private static GrayImage ToGray(Tensor t) =>
        new(t.Shape[1], t.Shape[0], t.ToLongArray().Select(v => (byte)v).ToArray());

    [Theory]
    [MemberData(nameof(Images))]
    public void ImageStagesMatch(string name)
    {
        using JsonDocument? reference = Reference(name);
        Assert.SkipWhen(reference is null, "Pipeline fixtures missing; run tools/reference/dump_pipeline.py.");
        JsonElement r = reference!.RootElement;
        Dictionary<string, Tensor> arrays = Npz.Load(Path.Combine(TestData.Pipeline, name + ".npz"));
        GrayImage gray = ImageIO.ReadPages(Path.Combine(TestData.Documents, name + ".png"))[0];
        NanoOcrEngine engine = Engine.Value;

        // Orientation and deskew: the upright page must be byte-identical.
        Assert.Equal(r.GetProperty("orientation").GetInt32(), engine.Orientation(gray));
        (GrayImage up, _, double skew) = engine.Upright(gray);
        Assert.Equal(r.GetProperty("skew").GetDouble(), skew, 9);
        GrayImage refUp = ToGray(arrays["upright"]);
        Assert.Equal(refUp.Width, up.Width);
        Assert.Equal(refUp.Height, up.Height);
        Assert.True(refUp.Pixels.AsSpan().SequenceEqual(up.Pixels), "upright page differs");

        // Detection.
        List<LineBox> boxes = engine.Detect(up);
        float[] refBoxes = arrays["boxes"].ToFloatArray();
        int n = refBoxes.Length / 5;
        Assert.Equal(n, boxes.Count);
        double worst = 0;
        for (int i = 0; i < n; i++)
        {
            float[] mine = [boxes[i].X0, boxes[i].Y0, boxes[i].X1, boxes[i].Y1, boxes[i].Score];
            for (int c = 0; c < 5; c++)
            {
                worst = Math.Max(worst, Math.Abs(mine[c] - refBoxes[(i * 5) + c]));
            }
        }

        output.WriteLine($"{name}: {n} boxes, worst coordinate difference {worst:E2} px");
        Assert.True(worst < 0.01, $"box coordinates differ by {worst}");

        // Recognition on the reference's own boxes.
        var refLineBoxes = Enumerable.Range(0, n).Select(i => new LineBox(refBoxes[i * 5], refBoxes[(i * 5) + 1], refBoxes[(i * 5) + 2], refBoxes[(i * 5) + 3], refBoxes[(i * 5) + 4])).ToList();
        LineReading[] rec = engine.Recognize(up, refLineBoxes);
        JsonElement[] refRec = r.GetProperty("rec").EnumerateArray().ToArray();
        Assert.Equal(refRec.Length, rec.Length);
        double worstConf = 0;
        for (int i = 0; i < rec.Length; i++)
        {
            Assert.Equal(refRec[i].GetProperty("text").GetString(), rec[i].Text);
            worstConf = Math.Max(worstConf, Math.Abs(refRec[i].GetProperty("conf").GetDouble() - rec[i].Conf));
            JsonElement[] toks = refRec[i].GetProperty("tokens").EnumerateArray().ToArray();
            Assert.Equal(toks.Length, rec[i].Tokens.Count);
            for (int k = 0; k < toks.Length; k++)
            {
                JsonElement[] t = toks[k].EnumerateArray().ToArray();
                if (t[0].ValueKind == JsonValueKind.String)
                {
                    Assert.Equal(t[0].GetString(), rec[i].Tokens[k].Text);
                }
                else
                {
                    Assert.Equal(t[0].GetInt32(), rec[i].Tokens[k].Id);
                }

                Assert.Equal(t[1].GetDouble(), rec[i].Tokens[k].X, 6);
            }
        }

        output.WriteLine($"{name}: {rec.Length} lines recognized identically; worst confidence difference {worstConf:E2}");
        Assert.True(worstConf < 1e-4);

        // The whole page.
        ComparePage(r.GetProperty("page"), engine.Process(gray), name);
    }

    [Fact]
    public void MathPageMatches()
    {
        using JsonDocument? reference = Reference("page_math", ".math.json");
        Assert.SkipWhen(reference is null, "Pipeline fixtures missing; run tools/reference/dump_pipeline.py.");
        GrayImage gray = ImageIO.ReadPages(Path.Combine(TestData.Documents, "page_math.png"))[0];
        ComparePage(reference!.RootElement.GetProperty("page"), MathEngine.Value.Process(gray), "page_math (math)");
    }

    [Theory]
    [MemberData(nameof(Pdfs))]
    public void PdfMatches(string name)
    {
        using JsonDocument? reference = Reference(name);
        Assert.SkipWhen(reference is null, "Pipeline fixtures missing; run tools/reference/dump_pipeline.py.");
        JsonElement r = reference!.RootElement;
        Dictionary<string, Tensor> renders = Npz.Load(Path.Combine(TestData.Pipeline, name + ".npz"));
        string path = Path.Combine(TestData.Documents, name + ".pdf");
        using (PdfDocument doc = PdfDocument.Open(path))
        {
            JsonElement[] layers = r.GetProperty("layers").EnumerateArray().ToArray();
            Assert.Equal(layers.Length, doc.PageCount);
            for (int i = 0; i < doc.PageCount; i++)
            {
                using PdfPage page = doc.GetPage(i);
                GrayImage render = page.Render(200 / 72.0);
                GrayImage refRender = ToGray(renders[$"render{i}"]);
                Assert.Equal(refRender.Width, render.Width);
                Assert.Equal(refRender.Height, render.Height);
                int diff = 0, worst = 0;
                for (int p = 0; p < render.Pixels.Length; p++)
                {
                    int d = Math.Abs(render.Pixels[p] - refRender.Pixels[p]);
                    diff += d > 0 ? 1 : 0;
                    worst = Math.Max(worst, d);
                }

                output.WriteLine($"{name} page {i + 1}: render differs from pypdfium2's on {diff} of {render.Pixels.Length} pixels (worst {worst})");

                TextLayerResult tl = TextLayer.Read(page, page.Render(1.0), 1.0);
                JsonElement layer = layers[i];
                Assert.Equal(layer.GetProperty("trusted").GetBoolean(), tl.Trusted);
                Assert.Equal(layer.GetProperty("reason").GetString(), tl.Reason);
                JsonElement[] runs = layer.GetProperty("runs").EnumerateArray().ToArray();
                Assert.Equal(runs.Length, tl.Lines.Count);
                for (int k = 0; k < runs.Length; k++)
                {
                    Assert.Equal(runs[k].GetProperty("text").GetString(), tl.Lines[k].Text);
                    double[] bbox = runs[k].GetProperty("bbox").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                    for (int c = 0; c < 4; c++)
                    {
                        Assert.Equal(bbox[c], tl.Lines[k].BBox[c], 3);
                    }
                }
            }
        }

        foreach (var (mode, key) in new[] { (ConvertMode.Auto, "auto"), (ConvertMode.Ocr, "ocr") })
        {
            var converter = new DocumentConverter(Engine.Value, mode: mode);
            var pages = converter.Convert(path).Select(p => p.Page).ToList();
            JsonElement[] refPages = r.GetProperty(key).EnumerateArray().ToArray();
            Assert.Equal(refPages.Length, pages.Count);
            for (int i = 0; i < pages.Count; i++)
            {
                ComparePage(refPages[i], pages[i], $"{name} {key} page {i + 1}");
            }
        }
    }

    private void ComparePage(JsonElement expected, Page actual, string label)
    {
        Assert.Equal(expected.GetProperty("source").GetString(), actual.Source);
        Assert.Equal(expected.GetProperty("rotation").GetInt32(), actual.Rotation);
        Assert.Equal(expected.GetProperty("skew").GetDouble(), actual.Skew);
        Assert.Equal(expected.GetProperty("note").GetString(), actual.Note);
        JsonElement[] lines = expected.GetProperty("lines").EnumerateArray().ToArray();
        Assert.Equal(lines.Length, actual.Lines.Count);
        double worstBox = 0, worstConf = 0;
        for (int i = 0; i < lines.Length; i++)
        {
            Assert.Equal(lines[i].GetProperty("text").GetString(), actual.Lines[i].Text);
            double[] box = lines[i].GetProperty("box").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            for (int c = 0; c < 4; c++)
            {
                worstBox = Math.Max(worstBox, Math.Abs(box[c] - actual.Lines[i].Box[c]));
            }

            worstConf = Math.Max(worstConf, Math.Abs(lines[i].GetProperty("conf").GetDouble() - actual.Lines[i].Conf));
        }

        Assert.Equal(expected.GetProperty("markdown").GetString(), actual.Markdown);
        JsonElement[] blocks = expected.GetProperty("blocks").EnumerateArray().ToArray();
        Assert.Equal(blocks.Length, actual.Blocks.Count);
        for (int i = 0; i < blocks.Length; i++)
        {
            Assert.Equal(blocks[i].GetProperty("label").GetString(), actual.Blocks[i].Label);
            Assert.Equal(blocks[i].GetProperty("text").GetString(), actual.Blocks[i].Text);
            double[] bbox = blocks[i].GetProperty("bbox").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            for (int c = 0; c < 4; c++)
            {
                worstBox = Math.Max(worstBox, Math.Abs(bbox[c] - actual.Blocks[i].BBox[c]));
            }
        }

        output.WriteLine($"{label}: {lines.Length} lines and {blocks.Length} blocks match; markdown identical; worst box difference {worstBox:0.###} px, worst confidence difference {worstConf:0.####}");
        Assert.True(worstBox <= 0.1 + 1e-9, $"boxes differ by {worstBox} px");
        Assert.True(worstConf <= 2e-4, $"confidences differ by {worstConf}");
    }
}
