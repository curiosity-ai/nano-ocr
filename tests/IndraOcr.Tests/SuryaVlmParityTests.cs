using System.Globalization;
using System.Text;
using System.Text.Json;
using IndraOcr.Vlm.Chandra;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Surya;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// Surya 2's layout, full-page OCR, block OCR and table predictors against Surya's own Python
/// predictors driven by transformers (<c>tools/reference/dump_surya_vlm.py</c>): the raw model
/// outputs must be identical, and the parsed results equal up to float formatting. Skipped when
/// the dump or the checkpoint is absent.
/// </summary>
public class SuryaVlmParityTests(ITestOutputHelper output)
{
    private static string Dump => Path.Combine(TestData.Root, "tests", "data", "reference", "surya_vlm", "surya_vlm.json");

    private static string ModelDir => Environment.GetEnvironmentVariable("SURYA_MODEL_DIR")
        ?? Path.Combine(TestData.Root, "tests", "data", "models", "surya-ocr-2");

    [Fact]
    public void PredictorsMatchSurya()
    {
        Assert.SkipUnless(File.Exists(Dump) && File.Exists(Path.Combine(ModelDir, "config.json")), "Surya VLM dump or checkpoint missing.");
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Dump));
        JsonElement root = reference.RootElement;

        using SuryaInference inference = SuryaInference.Load(ModelDir);
        using RgbImage page = RgbImageIO.Read(Path.Combine(TestData.Documents, "page_math.png"));
        int[] c = root.GetProperty("table_crop").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        using RgbImage source = RgbImageIO.Read(Path.Combine(TestData.Documents, "page_two_column.png"));
        using RgbImage table = ChandraOutput.Crop(source, c[0], c[1], c[2], c[3]);

        List<LayoutResult> layout = new SuryaLayoutPredictor(inference).Predict([page]);
        List<PageOcrResult> full = new SuryaRecognitionPredictor(inference).Predict([page], fullPage: true);
        List<PageOcrResult> blocks = new SuryaRecognitionPredictor(inference).Predict([page], layout, fullPage: false);
        List<TableResult> tables = new SuryaTableRecPredictor(inference).PredictSimple([table]);

        var failures = new List<string>();
        void Expect(string what, string expected, string actual)
        {
            output.WriteLine($"{what}:\n{actual}");
            if (expected != actual)
            {
                failures.Add($"{what}\n  expected {expected}\n  actual   {actual}");
            }
        }

        Expect("layout", Describe(root.GetProperty("layout")[0]), Describe(JsonSerializer.SerializeToElement(layout[0], SuryaTestJson.Default.LayoutResult)));
        Expect("full_page", Describe(root.GetProperty("full_page")[0]), Describe(JsonSerializer.SerializeToElement(full[0], SuryaTestJson.Default.PageOcrResult)));
        Expect("blocks", Describe(root.GetProperty("blocks")[0]), Describe(JsonSerializer.SerializeToElement(blocks[0], SuryaTestJson.Default.PageOcrResult)));
        Expect("tables", Describe(root.GetProperty("tables")[0]), Describe(JsonSerializer.SerializeToElement(tables[0], SuryaTestJson.Default.TableResult)));
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>A canonical rendering: strings verbatim, numbers to 4 decimals, confidences to 3.</summary>
    private static string Describe(JsonElement element, string key = "")
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                var builder = new StringBuilder("{");
                foreach (JsonProperty p in element.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
                {
                    if (p.Name is "page" or "raw_logprobs")
                    {
                        continue;
                    }

                    builder.Append(p.Name).Append(':').Append(Describe(p.Value, p.Name)).Append(',');
                }

                return builder.Append('}').ToString();
            case JsonValueKind.Array:
                return "[" + string.Join(",", element.EnumerateArray().Select(e => Describe(e, key))) + "]";
            case JsonValueKind.Number:
                return element.GetDouble().ToString(key == "confidence" ? "F3" : "F4", CultureInfo.InvariantCulture);
            case JsonValueKind.String:
                return JsonSerializer.Serialize(element.GetString());
            default:
                return element.GetRawText();
        }
    }
}

[System.Text.Json.Serialization.JsonSerializable(typeof(LayoutResult))]
[System.Text.Json.Serialization.JsonSerializable(typeof(PageOcrResult))]
[System.Text.Json.Serialization.JsonSerializable(typeof(TableResult))]
internal sealed partial class SuryaTestJson : System.Text.Json.Serialization.JsonSerializerContext;
