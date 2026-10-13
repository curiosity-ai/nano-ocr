using System.Text.Json;
using IndraOcr.Vlm.Common;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Models;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>Small pieces of the VLM pipelines, with expectations computed by the Python originals.</summary>
public class VlmUnitTests
{
    /// <summary>
    /// <c>scale_to_fit</c> and <c>smart_resize</c> (Surya's server pixel bounds, then the checkpoint's)
    /// as Python computes them, including extreme aspect ratios.
    /// </summary>
    [Theory]
    [InlineData(1700, 2200, 1708, 2212, 2208, 1696, 2208, 1696)]
    [InlineData(800, 800, 812, 812, 800, 800, 800, 800)]
    [InlineData(100, 30, 420, 112, 128, 416, 160, 512)]
    [InlineData(5000, 7000, 2100, 2968, 2976, 2112, 2976, 2112)]
    [InlineData(3072, 2048, 3052, 2044, 2048, 3040, 2048, 3040)]
    [InlineData(28, 28, 224, 224, 224, 224, 256, 256)]
    [InlineData(2, 300, 28, 2744, 2752, 32, 2752, 32)]
    [InlineData(12000, 300, 12012, 308, 320, 12000, 320, 12000)]
    [InlineData(1596, 812, 1596, 812, 800, 1600, 800, 1600)]
    [InlineData(640, 480, 644, 476, 480, 640, 480, 640)]
    [InlineData(1536, 1536, 1540, 1540, 1536, 1536, 1536, 1536)]
    [InlineData(333, 777, 336, 784, 768, 320, 768, 320)]
    public void ResizePoliciesMatchPython(int w, int h, int sw, int sh, int h1, int w1, int h2, int w2)
    {
        Assert.Equal((sw, sh), ScaleToFit.TargetSize(w, h));
        Assert.Equal((h1, w1), ImageProcessor.SmartResize(sh, sw, 32, 3136, 6291456));
        Assert.Equal((h2, w2), ImageProcessor.SmartResize(sh, sw, 32, 65536, 16777216));
    }

    /// <summary><c>detect_repeat_token</c> on 80 seeded strings (expectations from Python).</summary>
    [Fact]
    public void RepeatDetectionMatchesPython()
    {
        string path = Path.Combine(TestData.Root, "tests", "IndraOcr.Tests", "Data", "repeat_cases.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        foreach (JsonElement c in document.RootElement.EnumerateArray())
        {
            string text = c[0].GetString()!;
            Assert.Equal(c[1].GetBoolean(), RepeatDetector.Detect(text));
            Assert.Equal(c[2].GetBoolean(), RepeatDetector.HasRepeat(text));
        }
    }

    [Theory]
    [InlineData("[]", true)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"56 121 743 173\", \"count\": 80}]", true)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"56 121 743 173\", \"count\": 80}, {\"label\": \"Table\", \"bbox\": \"0 0 1000 1000\", \"count\": 0}]", true)]
    [InlineData(" [ {\"label\":\"Text\",\"bbox\":\"1 2 3 4\",\"count\":10000} ] \n", true)]
    [InlineData("[{\"label\": \"Txt\", \"bbox\": \"1 2 3 4\", \"count\": 1}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3 45678\", \"count\": 1}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3\", \"count\": 1}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1  2 3 4\", \"count\": 1}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3 4\", \"count\": 10001}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3 4\", \"count\": 07}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3 4\", \"count\": -1}]", false)]
    [InlineData("[{\"bbox\": \"1 2 3 4\", \"label\": \"Text\", \"count\": 1}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3 4\"}]", false)]
    [InlineData("[{\"label\": \"Text\", \"bbox\": \"1 2 3 4\", \"count\": 1, \"x\": 2}]", false)]
    [InlineData("{\"label\": \"Text\"}", false)]
    public void LayoutGrammar(string json, bool valid)
    {
        var constraint = JsonArrayConstraint.SuryaLayout();
        bool accepted = true;
        foreach (char c in json)
        {
            if (!constraint.Accepts(c.ToString()))
            {
                accepted = false;
                break;
            }

            constraint.Advance(c.ToString());
        }

        Assert.Equal(valid, accepted && constraint.IsComplete);
    }

    [Fact]
    public void LayoutGrammarLimitsItemsAndAcceptsMultiCharacterTokens()
    {
        var constraint = JsonArrayConstraint.SuryaLayout();
        Assert.True(constraint.Accepts("[{\"label\": \""));
        constraint.Advance("[{\"label\": \"");
        Assert.True(constraint.Accepts("Section-Header"));
        Assert.False(constraint.Accepts("Section-Footer"));
        Assert.False(constraint.IsComplete);

        string item = "{\"label\": \"Text\", \"bbox\": \"1 2 3 4\", \"count\": 50}";
        var full = JsonArrayConstraint.SuryaLayout();
        full.Advance("[" + string.Join(", ", Enumerable.Repeat(item, 200)));
        Assert.False(full.Accepts(", "));
        Assert.True(full.Accepts("]"));
    }
}
