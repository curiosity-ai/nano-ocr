using System.Text.Json;
using IndraOcr.Onnx;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Models;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// The Qwen3.5 vision-language model (Surya 2, Chandra 2) against transformers, stage by stage:
/// the pre-resizes, the processor, the tokenizer, every vision block, every decoder layer, and the
/// greedy decode. The reference comes from <c>tools/reference/dump_vlm.py</c>; the checkpoint from
/// <c>SURYA_MODEL_DIR</c> / <c>CHANDRA_MODEL_DIR</c> or <c>tests/data/models/&lt;name&gt;</c>.
/// Both are skipped when absent.
/// </summary>
public class VlmParityTests(ITestOutputHelper output)
{
    public static TheoryData<string> Checkpoints => new() { "surya", "chandra" };

    private static (string Dump, string Model)? Locate(string name)
    {
        string dump = Path.Combine(TestData.Root, "tests", "data", "reference", "vlm", name, "vlm.npz");
        string model = Environment.GetEnvironmentVariable($"{name.ToUpperInvariant()}_MODEL_DIR")
            ?? Path.Combine(TestData.Root, "tests", "data", "models", $"{name}-ocr-2");
        return File.Exists(dump) && File.Exists(Path.Combine(model, "config.json")) ? (dump, model) : null;
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void PreprocessingIsByteExact(string name)
    {
        var found = Locate(name);
        Assert.SkipUnless(found is not null, $"Reference dump or checkpoint for {name} missing.");
        Dictionary<string, Tensor> reference = Npz.Load(found.Value.Dump);

        using RgbImage raw = ToImage(reference["image.raw"]);
        using RgbImage scaled = ScaleToFit.Apply(raw);
        AssertBytes(reference["image.scaled"], scaled, "scale_to_fit (PIL Lanczos)");

        Qwen35Config config = Qwen35Config.Load(found.Value.Model);
        var processor = new ImageProcessor(config, name == "surya" ? 3136 : null, name == "surya" ? 6291456 : null);
        using RgbImage resized = processor.Resize(scaled);
        AssertBytes(reference["image.resized"], resized, "smart_resize (torch bicubic, antialias)");

        PreprocessedImage pixels = processor.Patchify(resized);
        float[] expected = reference["pixel_values"].ToFloatArray();
        Assert.Equal(expected.Length, pixels.Pixels.Length);
        Assert.True(expected.AsSpan().SequenceEqual(pixels.Pixels), "pixel_values differ");
        long[] grid = reference["grid_thw"].ToLongArray();
        Assert.Equal([1L, pixels.GridH, pixels.GridW], grid);
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void ModelMatchesTransformers(string name)
    {
        var found = Locate(name);
        Assert.SkipUnless(found is not null, $"Reference dump or checkpoint for {name} missing.");
        Dictionary<string, Tensor> reference = Npz.Load(found.Value.Dump);
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.ChangeExtension(found.Value.Dump, ".json")));

        using Qwen35Vlm model = Qwen35Vlm.Load(
            found.Value.Model, name == "surya" ? 3136 : null, name == "surya" ? 6291456 : null);

        // The prompt the reference tokenized, and its ids.
        long[] expectedIds = reference["input_ids"].ToLongArray();
        long[] grid = reference["grid_thw"].ToLongArray();
        int merge = model.Config.SpatialMergeSize;
        int imageTokens = (int)(grid[1] * grid[2]) / (merge * merge);
        List<int> ids = model.Tokenizer.Encode(model.BuildPrompt(Prompt(meta), imageTokens));
        Assert.Equal(expectedIds, ids.Select(i => (long)i));

        (int[] positions, int _) = model.RopeIndex(ids, (int)grid[1] / merge, (int)grid[2] / merge);
        Assert.Equal(reference["position_ids"].ToLongArray(), positions.Select(p => (long)p));

        var failures = new List<string>();
        var worst = new List<(string Name, double Error)>();
        void Check(string tensor, ReadOnlySpan<float> actual, int width)
        {
            if (!reference.TryGetValue(tensor, out Tensor? expected))
            {
                return;
            }

            float[] e = expected.ToFloatArray();
            string? problem = Difference(e, actual, out double error);
            worst.Add((tensor, error));
            if (problem is not null)
            {
                failures.Add($"{tensor}: {problem}");
            }
        }

        int maxTokens = reference["gen.ids"].Length;
        using RgbImage raw = ToImage(reference["image.raw"]);
        using RgbImage scaled = ScaleToFit.Apply(raw);
        GenerationOutput result = model.Generate(
            scaled, Prompt(meta), new GenerationOptions { MaxNewTokens = maxTokens }, (n, v, w) =>
            {
                if (n.StartsWith("gen.logits", StringComparison.Ordinal))
                {
                    int step = int.Parse(n["gen.logits".Length..]);
                    float[] all = reference["gen.logits"].ToFloatArray();
                    string? p = Difference(all.AsSpan(step * w, w).ToArray(), v, out double err);
                    worst.Add((n, err));
                    if (p is not null)
                    {
                        failures.Add($"{n}: {p}");
                    }

                    return;
                }

                Check(n, v, w);
            });

        foreach ((string n, double e) in worst.OrderByDescending(x => x.Error).Take(8))
        {
            output.WriteLine($"{n}: worst normalized error {e:E2}");
        }

        output.WriteLine($"generated: {result.Text}");
        output.WriteLine(string.Join(", ", result.Timings.Select(t => $"{t.Key} {t.Value:F0}")));
        Assert.True(worst.Count > 30, $"only {worst.Count} tensors compared");
        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(10)));

        long[] expectedGen = reference["gen.ids"].ToLongArray();
        int stop = model.Tokenizer.TokenToId("<|im_end|>");
        long[] expectedTokens = expectedGen.TakeWhile(t => t != stop).ToArray();
        Assert.Equal(expectedTokens, result.TokenIds.Select(t => (long)t));
    }

    private static string Prompt(JsonDocument meta)
    {
        // The rendered prompt is "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>...".
        string rendered = meta.RootElement.GetProperty("prompt").GetString()!;
        int start = rendered.IndexOf("<|vision_end|>", StringComparison.Ordinal) + "<|vision_end|>".Length;
        return rendered[start..rendered.IndexOf("<|im_end|>", StringComparison.Ordinal)];
    }

    /// <summary>Same criterion as the ONNX layer tests: <c>|a − e| ≤ 1e-3·(|e| + rms(e)) + 1e-5</c>.</summary>
    private static string? Difference(float[] expected, ReadOnlySpan<float> actual, out double error)
    {
        error = 0;
        if (expected.Length != actual.Length)
        {
            error = double.PositiveInfinity;
            return $"length {actual.Length}, expected {expected.Length}";
        }

        double rms = Math.Sqrt(expected.Average(v => (double)v * v));
        int bad = 0;
        int first = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double tol = (1e-3 * (Math.Abs(expected[i]) + rms)) + 1e-5;
            double d = Math.Abs(expected[i] - actual[i]);
            if (double.IsNaN(actual[i]))
            {
                d = double.PositiveInfinity;
            }

            error = Math.Max(error, d / tol * 1e-3);
            if (d > tol)
            {
                bad++;
                first = first < 0 ? i : first;
            }
        }

        return bad == 0 ? null : $"{bad} of {expected.Length} differ; first at {first}: {actual[first]} vs {expected[first]}";
    }

    private static RgbImage ToImage(Tensor hwc)
    {
        long[] values = hwc.ToLongArray();
        var bytes = new byte[values.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = (byte)values[i];
        }

        return RgbImage.Wrap(bytes, hwc.Shape[1], hwc.Shape[0]);
    }

    private static void AssertBytes(Tensor expected, RgbImage actual, string what)
    {
        Assert.Equal([actual.Height, actual.Width, 3], expected.Shape);
        long[] e = expected.ToLongArray();
        ReadOnlySpan<byte> a = actual.Pixels;
        int differ = 0;
        int maxDiff = 0;
        for (int i = 0; i < e.Length; i++)
        {
            int d = Math.Abs((int)e[i] - a[i]);
            if (d != 0)
            {
                differ++;
                maxDiff = Math.Max(maxDiff, d);
            }
        }

        Assert.True(differ == 0, $"{what}: {differ} of {e.Length} bytes differ (max {maxDiff})");
    }
}
