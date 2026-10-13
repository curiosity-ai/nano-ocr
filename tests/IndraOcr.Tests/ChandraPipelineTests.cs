using System.Text.Json;
using IndraOcr.Vlm.Chandra;
using IndraOcr.Vlm.Imaging;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// Chandra 2 end to end against the <c>chandra</c> package's own pipeline
/// (<c>tools/reference/dump_chandra_pipeline.py</c>): the loaded image, the raw model output, the
/// markdown, HTML, chunks and figure names must be identical. Skipped without the dump or checkpoint.
/// </summary>
public class ChandraPipelineTests(ITestOutputHelper output)
{
    private static string Dump => Path.Combine(TestData.Root, "tests", "data", "reference", "chandra_pipeline", "chandra.json");

    private static string ModelDir => Environment.GetEnvironmentVariable("CHANDRA_MODEL_DIR")
        ?? Path.Combine(TestData.Root, "tests", "data", "models", "chandra-ocr-2");

    [Fact]
    public void PageMatchesChandra()
    {
        Assert.SkipUnless(File.Exists(Dump) && File.Exists(Path.Combine(ModelDir, "config.json")), "Chandra pipeline dump or checkpoint missing.");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Dump));
        JsonElement expected = document.RootElement;

        string input = expected.GetProperty("input").GetString()!;
        if (!Path.IsPathRooted(input))
        {
            input = Path.Combine(TestData.Root, input);
        }

        using ChandraOcr chandra = ChandraOcr.Load(ModelDir, new ChandraOptions { MaxOutputTokens = expected.GetProperty("max_tokens").GetInt32() });
        List<RgbImage> pages = chandra.LoadFile(input);
        Assert.Equal(expected.GetProperty("loaded_size").EnumerateArray().Select(v => v.GetInt32()), [pages[0].Width, pages[0].Height]);

        ChandraPage page = chandra.Predict(pages[0]);
        output.WriteLine(page.Raw);
        output.WriteLine("----");
        output.WriteLine(page.Markdown);
        output.WriteLine(string.Join(", ", page.Timings.Select(t => $"{t.Key} {t.Value:F0}")));

        Assert.Equal(expected.GetProperty("raw").GetString(), page.Raw);
        Assert.Equal(expected.GetProperty("markdown").GetString(), page.Markdown);
        Assert.Equal(expected.GetProperty("html").GetString(), page.Html);
        Assert.Equal(expected.GetProperty("token_count").GetInt32(), page.TokenCount);
        string chunks = string.Join("\n", expected.GetProperty("chunks").EnumerateArray().Select(c =>
            $"{c.GetProperty("label").GetString()} [{string.Join(",", c.GetProperty("bbox").EnumerateArray().Select(v => v.GetInt32()))}] {c.GetProperty("content").GetString()}"));
        Assert.Equal(chunks, string.Join("\n", page.Chunks.Select(c => $"{c.Label} [{string.Join(",", c.Bbox)}] {c.Content}")));
        Assert.Equal(
            expected.GetProperty("images").EnumerateObject().Select(p => $"{p.Name}={string.Join("x", p.Value.EnumerateArray().Select(v => v.GetInt32()))}"),
            page.Images.Select(p => $"{p.Key}={p.Value.Width}x{p.Value.Height}"));
    }
}
