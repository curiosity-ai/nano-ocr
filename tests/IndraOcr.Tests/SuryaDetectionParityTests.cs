using System.Diagnostics;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Surya.Detection;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// Surya's text-line detector against the Python package, stage by stage: chunking and the two
/// Pillow resizes and the processor (exact), the EfficientViT model's stages and its maps (float
/// tolerance), the stitched heatmaps, and the heatmap post-processing with its OpenCV calls
/// (exact). The reference comes from <c>tools/reference/dump_surya_detection.py</c>; the
/// checkpoint from <c>SURYA_DET_MODEL_DIR</c> or <c>tests/data/models/surya_det</c>. Both are
/// skipped when absent.
/// </summary>
public partial class SuryaDetectionParityTests(ITestOutputHelper output)
{
    public static TheoryData<string> Documents => new() { "page_two_column", "page_math", "tall", "wide" };

    private static string DumpDirectory => Path.Combine(TestData.Root, "tests", "data", "reference", "surya_detection");

    private static string ModelDirectory =>
        Environment.GetEnvironmentVariable("SURYA_DET_MODEL_DIR") ?? Path.Combine(TestData.Root, "tests", "data", "models", "surya_det");

    private static bool HaveDump => File.Exists(Path.Combine(DumpDirectory, "detection.npz"));

    private static bool HaveModel =>
        File.Exists(Path.Combine(ModelDirectory, "config.json")) && File.Exists(Path.Combine(ModelDirectory, "model.safetensors"));

    private static readonly Lazy<TextDetector> Detector = new(() => TextDetector.Load(ModelDirectory));

    [Theory]
    [MemberData(nameof(Documents))]
    public void PreprocessingIsExact(string name)
    {
        Assert.SkipUnless(HaveDump && HaveModel, "Surya detection reference dump or checkpoint missing.");
        using var archive = new NpzArchive(Path.Combine(DumpDirectory, "detection.npz"));
        JsonElement meta = Meta(name);
        var preprocessor = new DetectionPreprocessor(DetectionConfig.Load(ModelDirectory));

        using RgbImage image = archive.Image($"{name}.image");
        List<ImageChunk> chunks = preprocessor.Split(image);
        try
        {
            Assert.Equal(meta.GetProperty("chunk_heights").EnumerateArray().Select(e => e.GetInt32()), chunks.Select(c => c.Height));
            for (int k = 0; k < chunks.Count; k++)
            {
                RgbImage chunk = chunks[k].Image;
                using RgbImage thumb = PilThumbnail.Thumbnail(chunk, 1200, 1200);
                AssertBytes(archive, $"{name}.chunk{k}.thumb", thumb);
                using RgbImage resized = preprocessor.Resize(chunk);
                AssertBytes(archive, $"{name}.chunk{k}.resized", resized);
                float[] pixels = preprocessor.Normalize(resized);
                float[] expected = archive.Floats($"{name}.chunk{k}.pixels", out _);
                Assert.True(expected.AsSpan().SequenceEqual(pixels), $"chunk {k}: pixel_values differ");
            }
        }
        finally
        {
            foreach (ImageChunk c in chunks)
            {
                c.Image.Dispose();
            }
        }
    }

    [Fact]
    public void ModelStagesMatchPyTorch()
    {
        string path = Path.Combine(DumpDirectory, "model.npz");
        Assert.SkipUnless(File.Exists(path) && HaveModel, "Surya detection reference dump or checkpoint missing.");
        using var archive = new NpzArchive(path);
        float[] pixels = archive.Floats("pixels", out int[] shape);
        var failures = new List<string>();
        int compared = 0;
        Detector.Value.Model.Forward(pixels, shape[1], shape[2], (stage, map) =>
        {
            if (!archive.Contains(stage))
            {
                return;
            }

            float[] expected = archive.Floats(stage, out int[] s);
            Assert.Equal([map.Channels, map.Height, map.Width], s);
            string? problem = Difference(expected, map.ToChannelsFirst(), out double error);
            output.WriteLine($"{stage}: worst normalized error {error:E2}");
            compared++;
            if (problem is not null)
            {
                failures.Add($"{stage}: {problem}");
            }
        });
        Assert.Equal(10, compared);
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Theory]
    [MemberData(nameof(Documents))]
    public void BoxesFromTheReferenceHeatmapAreIdentical(string name)
    {
        Assert.SkipUnless(HaveDump, "Surya detection reference dump missing.");
        using var archive = new NpzArchive(Path.Combine(DumpDirectory, "detection.npz"));
        JsonElement meta = Meta(name);
        float[] heatmap = archive.Floats($"{name}.heatmap", out int[] shape);
        int rows = shape[0], width = shape[1];

        (float text, float low) = HeatmapBoxes.DynamicThresholds(heatmap, HeatmapBoxes.TextThreshold, HeatmapBoxes.BlankThreshold);
        double[] thresholds = meta.GetProperty("thresholds").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        Assert.Equal(thresholds[0], (double)text);
        Assert.Equal(thresholds[1], (double)low);

        (List<float[]> raw, List<float> confidences) = HeatmapBoxes.DetectBoxes(heatmap, width, rows);
        JsonElement expectedRaw = meta.GetProperty("raw");
        double[][] expectedPolygons = expectedRaw.GetProperty("polygons").EnumerateArray()
            .Select(p => p.EnumerateArray().SelectMany(c => c.EnumerateArray()).Select(v => v.GetDouble()).ToArray()).ToArray();
        Assert.Equal(expectedPolygons.Length, raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            Assert.True(expectedPolygons[i].SequenceEqual(raw[i].Select(v => (double)v)),
                $"raw box {i}: {string.Join(",", raw[i])} vs {string.Join(",", expectedPolygons[i])}");
        }

        Assert.Equal(expectedRaw.GetProperty("confidences").EnumerateArray().Select(e => e.GetDouble()), confidences.Select(c => (double)c));

        int[] size = meta.GetProperty("size").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        List<PolygonBox> boxes = HeatmapBoxes.GetBoxes(heatmap, width, rows, size[0], size[1]);
        AssertBoxes(meta, boxes);
        output.WriteLine($"{name}: {boxes.Count} boxes, thresholds {text} / {low}");
    }

    [Theory]
    [MemberData(nameof(Documents))]
    public void DetectorMatchesSurya(string name)
    {
        Assert.SkipUnless(HaveDump && HaveModel, "Surya detection reference dump or checkpoint missing.");
        using var archive = new NpzArchive(Path.Combine(DumpDirectory, "detection.npz"));
        JsonElement meta = Meta(name);
        TextDetector detector = Detector.Value;
        var failures = new List<string>();
        var worst = new List<string>();

        detector.ChunkObserver = (k, maps) =>
        {
            float[] expected = archive.Floats($"{name}.chunk{k}.logits", out int[] s);
            Assert.Equal([maps.Labels, maps.Height, maps.Width], s);
            string? problem = Difference(expected, maps.Maps.AsSpan(0, expected.Length), out double error);
            worst.Add($"chunk{k}.logits {error:E2}");
            if (problem is not null)
            {
                failures.Add($"chunk {k} logits: {problem}");
            }
        };

        using RgbImage image = archive.Image($"{name}.image");
        TextDetectionResult result;
        try
        {
            result = detector.Detect(image, includeMaps: true);
        }
        finally
        {
            detector.ChunkObserver = null;
        }

        output.WriteLine($"{name}: {string.Join(", ", detector.Timings.Select(t => $"{t.Key} {t.Value:F0} ms"))}");
        foreach (string map in new[] { "heatmap", "affinity" })
        {
            float[] expected = archive.Floats($"{name}.{map}", out int[] s);
            Assert.Equal([result.HeatmapHeight, result.HeatmapWidth], s);
            string? problem = Difference(expected, map == "heatmap" ? result.Heatmap : result.AffinityMap, out double error);
            worst.Add($"{map} {error:E2}");
            if (problem is not null)
            {
                failures.Add($"{map}: {problem}");
            }
        }

        output.WriteLine($"{name}: worst normalized error {string.Join(", ", worst)}");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
        Assert.Equal(meta.GetProperty("image_bbox").EnumerateArray().Select(e => e.GetDouble()), result.ImageBbox);
        AssertBoxes(meta, result.Bboxes, confidenceTolerance: 2e-6);
    }

    [Fact]
    public void Timing()
    {
        Assert.SkipUnless(HaveDump && HaveModel, "Surya detection reference dump or checkpoint missing.");
        using var archive = new NpzArchive(Path.Combine(DumpDirectory, "detection.npz"));
        TextDetector detector = Detector.Value;
        foreach (string name in new[] { "page_two_column", "page_math" })
        {
            using RgbImage image = archive.Image($"{name}.image");
            var samples = new List<double>();
            for (int i = 0; i < 4; i++)
            {
                var clock = Stopwatch.StartNew();
                detector.Detect(image);
                samples.Add(clock.Elapsed.TotalMilliseconds);
            }

            output.WriteLine($"{name}: best {samples.Skip(1).Min():F0} ms of {string.Join(", ", samples.Select(s => s.ToString("F0")))} (first is cold)");
        }
    }

    /// <summary>
    /// Polygons and bboxes must be identical. A confidence is a heatmap peak over the page's
    /// largest one, so it is exact when the heatmap is the reference's and carries the model's
    /// float error otherwise: <paramref name="confidenceTolerance"/> is 0 or a few float32 ulps.
    /// </summary>
    private void AssertBoxes(JsonElement meta, IReadOnlyList<PolygonBox> boxes, double confidenceTolerance = 0)
    {
        JsonElement[] expected = meta.GetProperty("bboxes").EnumerateArray().ToArray();
        var lines = new List<string>();
        double worst = 0;
        for (int i = 0; i < Math.Max(expected.Length, boxes.Count); i++)
        {
            string e = i < expected.Length ? Describe(expected[i]) : "-";
            string a = i < boxes.Count ? Describe(boxes[i]) : "-";
            double confidenceError = i < expected.Length && i < boxes.Count
                ? Math.Abs(expected[i].GetProperty("confidence").GetDouble() - boxes[i].Confidence!.Value)
                : double.PositiveInfinity;
            worst = Math.Max(worst, confidenceError);
            if (e != a || confidenceError > confidenceTolerance)
            {
                lines.Add($"box {i}: expected {e} {(i < expected.Length ? expected[i].GetProperty("confidence").GetDouble() : 0):R}, " +
                    $"got {a} {(i < boxes.Count ? boxes[i].Confidence : 0):R}");
            }
        }

        output.WriteLine($"{boxes.Count} boxes, polygons identical: {lines.Count == 0}, worst confidence difference {worst:E2}");
        Assert.True(lines.Count == 0, $"{expected.Length} boxes expected, {boxes.Count} produced\n" + string.Join("\n", lines.Take(10)));
    }

    private static string Describe(JsonElement box) =>
        $"{Polygon(box.GetProperty("polygon").EnumerateArray().Select(p => p.EnumerateArray().Select(v => v.GetDouble()).ToArray()))} " +
        $"{string.Join(",", box.GetProperty("bbox").EnumerateArray().Select(v => v.GetDouble()))}";

    private static string Describe(PolygonBox box) =>
        $"{Polygon(box.Polygon)} {string.Join(",", box.Bbox)}";

    private static string Polygon(IEnumerable<double[]> polygon) => string.Join(" ", polygon.Select(p => $"({p[0]},{p[1]})"));

    private static JsonElement Meta(string name)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DumpDirectory, "detection.json")));
        return document.RootElement.GetProperty("documents").GetProperty(name).Clone();
    }

    private static void AssertBytes(NpzArchive archive, string name, RgbImage actual)
    {
        byte[] expected = archive.Bytes(name, out int[] shape);
        Assert.Equal([actual.Height, actual.Width, 3], shape);
        int differ = 0;
        ReadOnlySpan<byte> a = actual.Pixels;
        for (int i = 0; i < expected.Length; i++)
        {
            differ += expected[i] != a[i] ? 1 : 0;
        }

        Assert.True(differ == 0, $"{name}: {differ} of {expected.Length} bytes differ");
    }

    /// <summary>The criterion of the other parity tests: <c>|a − e| ≤ 1e-3·(|e| + rms(e)) + 1e-5</c>.</summary>
    private static string? Difference(float[] expected, ReadOnlySpan<float> actual, out double error)
    {
        error = 0;
        if (expected.Length != actual.Length)
        {
            error = double.PositiveInfinity;
            return $"length {actual.Length}, expected {expected.Length}";
        }

        double sum = 0;
        foreach (float v in expected)
        {
            sum += (double)v * v;
        }

        double rms = Math.Sqrt(sum / expected.Length);
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

    /// <summary>
    /// Reads single arrays out of an <c>.npz</c> on demand — the detection dump is a few hundred
    /// MB, and its uint8 images would be eight times that through <see cref="Npz"/>'s int64 tensors.
    /// </summary>
    private sealed partial class NpzArchive(string path) : IDisposable
    {
        private readonly ZipArchive _zip = ZipFile.OpenRead(path);

        public bool Contains(string name) => _zip.GetEntry(name + ".npy") is not null;

        public float[] Floats(string name, out int[] shape)
        {
            (string descr, byte[] bytes, int offset, int[] s) = Read(name);
            shape = s;
            Assert.Equal("<f4", descr);
            var values = new float[(bytes.Length - offset) / 4];
            Buffer.BlockCopy(bytes, offset, values, 0, values.Length * 4);
            return values;
        }

        public byte[] Bytes(string name, out int[] shape)
        {
            (string descr, byte[] bytes, int offset, int[] s) = Read(name);
            shape = s;
            Assert.Equal("|u1", descr);
            return bytes.AsSpan(offset).ToArray();
        }

        public RgbImage Image(string name)
        {
            byte[] bytes = Bytes(name, out int[] shape);
            return RgbImage.Wrap(bytes, shape[1], shape[0]);
        }

        private (string Descr, byte[] Bytes, int Offset, int[] Shape) Read(string name)
        {
            ZipArchiveEntry entry = _zip.GetEntry(name + ".npy") ?? throw new KeyNotFoundException(name);
            using Stream stream = entry.Open();
            var bytes = new byte[entry.Length];
            stream.ReadExactly(bytes);
            int major = bytes[6];
            int headerLength = major == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8);
            int start = major == 1 ? 10 : 12;
            string header = Encoding.ASCII.GetString(bytes, start, headerLength);
            string descr = DescrPattern().Match(header).Groups[1].Value;
            int[] shape = ShapePattern().Match(header).Groups[1].Value
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
            return (descr, bytes, start + headerLength, shape);
        }

        public void Dispose() => _zip.Dispose();

        [GeneratedRegex(@"'descr':\s*'([^']+)'")]
        private static partial Regex DescrPattern();

        [GeneratedRegex(@"'shape':\s*\(([^)]*)\)")]
        private static partial Regex ShapePattern();
    }
}
