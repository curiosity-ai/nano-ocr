using System.Text.Json;
using IndraOcr.Vlm.Chandra;
using IndraOcr.Vlm.Html;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Surya;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// The HTML stack against BeautifulSoup, markdownify 1.1.0 and the Chandra/Surya output parsers,
/// string for string, over the corpus <c>tools/reference/dump_html.py</c> writes. Skipped when the
/// dump is absent.
/// </summary>
public class HtmlParityTests(ITestOutputHelper output)
{
    private static string Dump => Path.Combine(TestData.Root, "tests", "data", "reference", "html", "html.json");

    [Fact]
    public void EveryCaseMatchesPython()
    {
        Assert.SkipUnless(File.Exists(Dump), "HTML reference dump missing; run tools/reference/dump_html.py.");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Dump));
        using RgbImage page = RgbImage.Wrap(new byte[1000 * 1400 * 3], 1000, 1400);

        var failures = new List<string>();
        int index = 0;
        foreach (JsonElement entry in document.RootElement.EnumerateArray())
        {
            string html = entry.GetProperty("html").GetString()!;
            void Expect(string what, string expected, string actual)
            {
                if (expected != actual)
                {
                    failures.Add($"case {index} {what}:\n  html     {Show(html)}\n  expected {Show(expected)}\n  actual   {Show(actual)}");
                }
            }

            Expect("soup", entry.GetProperty("soup").GetString()!, Soup.Parse(html).Decode());
            Expect("chandra_html", entry.GetProperty("chandra_html").GetString()!, ChandraOutput.ParseHtml(html));
            Expect("chandra_html_all", entry.GetProperty("chandra_html_all").GetString()!, ChandraOutput.ParseHtml(html, true, false));
            Expect("chandra_markdown", entry.GetProperty("chandra_markdown").GetString()!, ChandraOutput.ParseMarkdown(html));
            Expect("chandra_markdown_all", entry.GetProperty("chandra_markdown_all").GetString()!, ChandraOutput.ParseMarkdown(html, true, false));

            Expect("markdownify", entry.GetProperty("markdownify").GetString()!, new ChandraMarkdownConverter().Convert(html));

            List<LayoutBlock> chunks = ChandraOutput.ParseLayout(html, 1000, 1400);
            string expectedChunks = string.Join("\n", entry.GetProperty("chandra_chunks").EnumerateArray().Select(c =>
                $"{c.GetProperty("label").GetString()} [{string.Join(",", c.GetProperty("bbox").EnumerateArray().Select(v => v.GetInt32()))}] {c.GetProperty("content").GetString()}"));
            string actualChunks = string.Join("\n", chunks.Select(c => $"{c.Label} [{string.Join(",", c.Bbox)}] {c.Content}"));
            Expect("chandra_chunks", expectedChunks, actualChunks);

            string expectedImages = string.Join(";", entry.GetProperty("chandra_images").EnumerateObject()
                .Select(p => $"{p.Name}={string.Join("x", p.Value.EnumerateArray().Select(v => v.GetInt32()))}"));
            string actualImages = string.Join(";", ChandraOutput.ExtractImages(html, chunks, page).Select(p => $"{p.Key}={p.Value.Width}x{p.Value.Height}"));
            Expect("chandra_images", expectedImages, actualImages);

            if (entry.TryGetProperty("surya_full_page", out JsonElement full))
            {
                string expected = string.Join("\n", full.EnumerateArray().Select(b =>
                    $"{b[0].GetString()} [{string.Join(",", b[1].EnumerateArray().Select(v => v.GetDouble()))}] {b[2].GetString()}"));
                string actual = string.Join("\n", SuryaParsers.ParseFullPageHtml(html).Select(b =>
                    $"{b.Label} [{string.Join(",", b.Bbox)}] {b.Html}"));
                Expect("surya_full_page", expected, actual);
            }

            if (entry.TryGetProperty("surya_clean_block_html", out JsonElement clean))
            {
                Expect("surya_clean_block_html", clean.GetString()!, SuryaParsers.CleanBlockHtml(html));
            }

            index++;
        }

        output.WriteLine($"{index} cases, {failures.Count} mismatches");
        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(12)));
    }

    private static string Show(string s) => JsonSerializer.Serialize(s.Length > 400 ? s[..400] + "…" : s);
}
