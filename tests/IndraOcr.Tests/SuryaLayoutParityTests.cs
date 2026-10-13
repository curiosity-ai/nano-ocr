using System.Diagnostics;
using System.Text.Json;
using IndraOcr.Onnx;
using IndraOcr.Vlm.Formats;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Surya.Layout;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// Surya's fast layout (RF-DETR with a windowed DINOv2 backbone) and its reading-order head against
/// PyTorch, stage by stage: the preprocessing to the bit, every backbone layer, the projector, the
/// two-stage proposals, every decoder layer, the heads, post-processing, the order head's encoded
/// boxes and per-step logits, and the final <c>LayoutResult</c>. The reference comes from
/// <c>tools/reference/dump_surya_layout.py</c>; the checkpoint from <c>SURYA_LAYOUT_MODEL_DIR</c> or
/// <c>tests/data/models/surya_layout2</c>. Both are skipped when absent.
/// </summary>
public class SuryaLayoutParityTests(ITestOutputHelper output)
{
    public static TheoryData<string> Pages => new() { "page_two_column", "page_math" };

    private static string ModelDir =>
        Environment.GetEnvironmentVariable("SURYA_LAYOUT_MODEL_DIR")
        ?? Path.Combine(TestData.Root, "tests", "data", "models", "surya_layout2");

    private static string? Dump(string page)
    {
        string path = Path.Combine(TestData.Root, "tests", "data", "reference", "surya_layout", page + ".npz");
        return File.Exists(path) && File.Exists(Path.Combine(ModelDir, "config.json")) ? path : null;
    }

    private static readonly Lazy<FastLayoutPredictor> Predictor = new(() => FastLayoutPredictor.Load(ModelDir));

    [Fact]
    public void CheckpointReaderReadsBothFiles()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(ModelDir, "config.json")), "surya_layout2 checkpoint missing.");

        TorchCheckpoint layout = TorchCheckpoint.Load(Path.Combine(ModelDir, "rfdetr_layout.pth"));
        Dictionary<string, WeightTensor> state = layout.StateDict("model");
        Assert.Equal(510, state.Count);
        Assert.Equal([1, 1937, 384], state["backbone.0.encoder.encoder.embeddings.position_embeddings"].Shape);
        Assert.Equal([0, 0], state["_kp_active_mask"].Shape);
        Assert.Equal(Vlm.Core.DType.Bool, state["_kp_active_mask"].Dtype);

        TorchCheckpoint order = TorchCheckpoint.Load(Path.Combine(ModelDir, "order", "order_ar.pt"));
        Assert.Equal(128, order.GetInt("d", 0));
        Assert.Equal(3, order.GetInt("layers", 0));
        Assert.Equal(28, order.GetInt("feat_hw", 0));
        Assert.Equal(118, order.StateDict("model").Count);
    }

    [Theory]
    [MemberData(nameof(Pages))]
    public void PreprocessingIsExact(string page)
    {
        string? dump = Dump(page);
        Assert.SkipUnless(dump is not null, $"Reference dump or checkpoint for {page} missing.");
        Dictionary<string, Tensor> reference = Npz.Load(dump);

        using RgbImage image = RgbImageIO.Read(Path.Combine(TestData.Documents, page + ".png"));
        long[] raw = reference["image.raw"].ToLongArray();
        Assert.Equal([image.Height, image.Width, 3], reference["image.raw"].Shape);
        ReadOnlySpan<byte> px = image.Pixels;
        for (int i = 0; i < raw.Length; i++)
        {
            Assert.True(raw[i] == px[i], $"decoded pixel byte {i} differs: {px[i]} vs {raw[i]}");
        }

        float[] pixels = Predictor.Value.Detector.Preprocess(image);
        float[] expected = reference["pixel_values"].ToFloatArray();
        Assert.Equal(expected.Length, pixels.Length);
        int differ = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(pixels[i]))
            {
                differ++;
            }
        }

        Assert.True(differ == 0, $"{differ} of {expected.Length} pixel values differ (to_tensor/normalize/resize)");
    }

    [Theory]
    [MemberData(nameof(Pages))]
    public void ModelMatchesTorch(string page)
    {
        string? dump = Dump(page);
        Assert.SkipUnless(dump is not null, $"Reference dump or checkpoint for {page} missing.");
        Dictionary<string, Tensor> reference = Npz.Load(dump);
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.ChangeExtension(dump, ".json")));
        using RgbImage image = RgbImageIO.Read(Path.Combine(TestData.Documents, page + ".png"));
        FastLayoutPredictor predictor = Predictor.Value;

        var failures = new List<string>();
        var errors = new List<(string Name, double Error)>();
        var orderLogits = new List<float>();
        float[]? postLabels = null, postScores = null, postBoxes = null;
        void Trace(string name, ReadOnlySpan<float> values)
        {
            switch (name)
            {
                case "post.scores":
                    postScores = values.ToArray();
                    return;
                case "post.boxes":
                    postBoxes = values.ToArray();
                    return;
                case "order.logits":
                    orderLogits.AddRange(values.ToArray());
                    return;
                case "enc.topk":
                    return;
                case "post.labels":
                    postLabels = values.ToArray();
                    return;
            }

            Compare(name, values);
        }

        void Compare(string name, ReadOnlySpan<float> values)
        {
            if (!reference.TryGetValue(name, out Tensor? expected))
            {
                failures.Add($"{name}: not in the reference");
                return;
            }

            string? problem = Difference(expected.ToFloatArray(), values, out double error);
            errors.Add((name, error));
            if (problem is not null)
            {
                failures.Add($"{name}: {problem}");
            }
        }

        // The decoder's 300 queries are the top encoder proposals, and they are positional: query i
        // takes the learned refpoint_embed[i] and query_feat[i]. Which proposals are selected is
        // checked as a set; their rank order is then taken from the reference, because two
        // proposals whose scores differ by a few ulps can swap ranks between two float
        // implementations (page_two_column has such a pair, 2.4e-6 apart at ranks 144/145), and a
        // swap hands the two queries each other's seeds. The end-to-end run below has no override.
        long[] expectedTopk = reference["enc.topk"].ToLongArray();
        int[]? portTopk = null;
        predictor.Detector.ProposalOverride = mine =>
        {
            portTopk = mine;
            return expectedTopk.Select(v => (int)v).ToArray();
        };
        LayoutResult result;
        try
        {
            result = predictor.Predict(image, trace: Trace);
        }
        finally
        {
            predictor.Detector.ProposalOverride = null;
        }

        Compare("order.logits", orderLogits.ToArray());
        Assert.Equal(expectedTopk.Order(), portTopk!.Select(v => (long)v).Order());
        int swapped = expectedTopk.Zip(portTopk!).Count(p => p.First != p.Second);
        output.WriteLine($"enc.topk: the same 300 proposals; {swapped} ranks differ");

        foreach ((string n, double e) in errors)
        {
            output.WriteLine($"{n,-16} worst normalized error {e:E2}");
        }

        Assert.True(errors.Count >= 35, $"only {errors.Count} tensors compared");
        Assert.True(failures.Count == 0, string.Join("\n", failures));

        // Post-processing: the top 100 (query, class) pairs by score. Ranks whose scores are a
        // float-noise tie may come out in either order (page_two_column has one at ranks 75/76,
        // 0.01571948 against 0.01571941, far below the 0.4 threshold), so each reference entry is
        // matched to an entry of the port within two ranks with the same label, an equal score
        // and the same box.
        Compare("post.scores", postScores!);
        float[] refScores = reference["post.scores"].ToFloatArray(), refBoxes = reference["post.boxes"].ToFloatArray();
        long[] refLabels = reference["post.labels"].ToLongArray();
        var taken = new bool[refLabels.Length];
        int reordered = 0;
        for (int r = 0; r < refLabels.Length; r++)
        {
            int match = -1;
            for (int c = Math.Max(0, r - 2); c < Math.Min(refLabels.Length, r + 3) && match < 0; c++)
            {
                bool same = !taken[c] && (long)postLabels![c] == refLabels[r]
                    && Math.Abs(postScores![c] - refScores[r]) <= (1e-3 * refScores[r]) + 1e-6
                    && Enumerable.Range(0, 4).All(k => Math.Abs(postBoxes![(c * 4) + k] - refBoxes[(r * 4) + k]) <= (1e-3 * Math.Abs(refBoxes[(r * 4) + k])) + 0.05);
                match = same ? c : -1;
            }

            Assert.True(match >= 0, $"post-processing rank {r} (label {refLabels[r]}, score {refScores[r]}) has no counterpart");
            taken[match] = true;
            reordered += match != r ? 1 : 0;
        }

        output.WriteLine($"post: top {refLabels.Length} identical up to {reordered} ranks swapped at score ties");

        // The final LayoutResult.
        JsonElement expectedBoxes = meta.RootElement.GetProperty("result").GetProperty("bboxes");
        AssertSameLayout(expectedBoxes, result, 0.01, "tensor-parity run");

        // End to end, with the port's own proposal ranking.
        LayoutResult free = predictor.Predict(image);
        AssertSameLayout(expectedBoxes, free, swapped == 0 ? 0.01 : 0.05, "end-to-end run");

        output.WriteLine(string.Join("\n", result.Boxes.Select(b =>
            $"{b.Position,2} {b.Label,-16} {b.Confidence:F4} [{string.Join(", ", b.Bbox.Select(v => v.ToString("F2")))}]")));

        // Timing of a warm page.
        var times = new List<double>();
        for (int run = 0; run < 3; run++)
        {
            long mark = Stopwatch.GetTimestamp();
            predictor.Predict(image);
            times.Add(Stopwatch.GetElapsedTime(mark).TotalMilliseconds);
        }

        output.WriteLine($"warm page: best {times.Min():F0} ms of [{string.Join(", ", times.Select(t => t.ToString("F0")))}]; "
            + string.Join(", ", predictor.LastTimings.Select(t => $"{t.Key} {t.Value:F0}")));
    }

    /// <summary>
    /// Same boxes, labels, raw labels and reading positions, in the same order; confidences and
    /// corners within float noise (corners: <c>1e-4·|v| + pixels</c>).
    /// </summary>
    private void AssertSameLayout(JsonElement expectedBoxes, LayoutResult result, double pixels, string what)
    {
        Assert.Equal(expectedBoxes.GetArrayLength(), result.Boxes.Count);
        int i = 0;
        double worstCorner = 0, worstConfidence = 0;
        foreach (JsonElement e in expectedBoxes.EnumerateArray())
        {
            LayoutBox b = result.Boxes[i++];
            Assert.Equal(e.GetProperty("label").GetString(), b.Label);
            Assert.Equal(e.GetProperty("raw_label").GetString(), b.RawLabel);
            Assert.Equal(e.GetProperty("position").GetInt32(), b.Position);
            double dc = Math.Abs(e.GetProperty("confidence").GetDouble() - b.Confidence);
            worstConfidence = Math.Max(worstConfidence, dc);
            double[][] polygon = e.GetProperty("polygon").EnumerateArray()
                .Select(p => p.EnumerateArray().Select(v => v.GetDouble()).ToArray()).ToArray();
            for (int c = 0; c < 4; c++)
            {
                for (int k = 0; k < 2; k++)
                {
                    double d = Math.Abs(polygon[c][k] - b.Polygon[c][k]);
                    worstCorner = Math.Max(worstCorner, d);
                    Assert.True(d <= (1e-4 * Math.Abs(polygon[c][k])) + pixels, $"{what}: box {i} corner {c}: {b.Polygon[c][k]} vs {polygon[c][k]}");
                }
            }
        }

        output.WriteLine($"{what}: {result.Boxes.Count} boxes, labels and order identical; worst corner {worstCorner:E2} px, worst confidence {worstConfidence:E2}");
        Assert.True(worstConfidence < 1e-3, $"{what}: confidence differs by {worstConfidence}");
    }

    /// <summary>Same criterion as the other parity tests: <c>|a − e| ≤ 1e-3·(|e| + rms(e)) + 1e-5</c>.</summary>
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
}
