using System.Text.Json;
using IndraOcr.Vlm.Surya;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// Surya's OCR-error classifier against transformers (<c>tools/reference/dump_surya_ocr_error.py</c>):
/// token ids identical, logits within the layer tolerance, labels identical and scores within 1e-4.
/// </summary>
public class SuryaOcrErrorParityTests(ITestOutputHelper output)
{
    private static string Dump => Path.Combine(TestData.Root, "tests", "data", "reference", "surya_ocr_error", "ocr_error.json");

    private static string ModelDir => Environment.GetEnvironmentVariable("SURYA_OCR_ERROR_MODEL_DIR")
        ?? Path.Combine(TestData.Root, "tests", "data", "models", "surya_ocr_error");

    [Fact]
    public void ClassifierMatchesTransformers()
    {
        Assert.SkipUnless(File.Exists(Dump) && File.Exists(Path.Combine(ModelDir, "model.safetensors")), "OCR-error dump or checkpoint missing.");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Dump));
        using OcrErrorDetector detector = OcrErrorDetector.Load(ModelDir);
        var failures = new List<string>();
        var texts = new List<string>();
        foreach (JsonElement entry in document.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            texts.Add(text);
            int[] expectedIds = entry.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            float[] expectedLogits = entry.GetProperty("logits").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
            float[] logits = detector.Logits(text);
            output.WriteLine($"{logits[0],9:F5} {logits[1],9:F5}  vs {expectedLogits[0],9:F5} {expectedLogits[1],9:F5}  {text[..Math.Min(40, text.Length)]}");
            for (int i = 0; i < 2; i++)
            {
                if (Math.Abs(logits[i] - expectedLogits[i]) > (1e-3 * (Math.Abs(expectedLogits[i]) + 1)) + 1e-5)
                {
                    failures.Add($"logits of {JsonSerializer.Serialize(text[..Math.Min(40, text.Length)])}: {logits[i]} vs {expectedLogits[i]}");
                }
            }
        }

        OcrErrorResult result = detector.Predict(texts);
        int index = 0;
        foreach (JsonElement entry in document.RootElement.EnumerateArray())
        {
            if (result.Labels[index] != entry.GetProperty("label").GetString())
            {
                failures.Add($"label {index}: {result.Labels[index]} vs {entry.GetProperty("label").GetString()}");
            }

            if (Math.Abs(result.Scores[index] - entry.GetProperty("score").GetDouble()) > 1e-4)
            {
                failures.Add($"score {index}: {result.Scores[index]} vs {entry.GetProperty("score").GetDouble()}");
            }

            index++;
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void TokenizerMatchesTransformers()
    {
        Assert.SkipUnless(File.Exists(Dump) && File.Exists(Path.Combine(ModelDir, "tokenizer.json")), "OCR-error dump or tokenizer missing.");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Dump));
        var tokenizer = IndraOcr.Vlm.Text.WordPieceTokenizer.FromFile(Path.Combine(ModelDir, "tokenizer.json"));
        foreach (JsonElement entry in document.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            int[] expected = entry.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            Assert.Equal(expected, tokenizer.Encode(text, 512));
        }
    }
}
