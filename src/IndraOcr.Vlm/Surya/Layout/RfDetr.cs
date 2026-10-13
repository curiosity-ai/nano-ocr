using System.Text.Json;
using IndraOcr.Vlm.Formats;
using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>One raw detection: <c>RfDetrTorch.detect</c>'s <c>{label, label_id, score, bbox}</c>.</summary>
/// <param name="Label">The category name (<c>config.json</c>'s, e.g. <c>Section-Header</c>).</param>
/// <param name="LabelId">The 0-based class index the head predicted.</param>
/// <param name="Score">Sigmoid confidence.</param>
/// <param name="X0">Left edge, pixels.</param>
/// <param name="Y0">Top edge, pixels.</param>
/// <param name="X1">Right edge, pixels.</param>
/// <param name="Y1">Bottom edge, pixels.</param>
public readonly record struct RfDetrDetection(string Label, int LabelId, float Score, float X0, float Y0, float X1, float Y1);

/// <summary>
/// RF-DETR (the "large" layout configuration Surya's fast layout ships) in float32:
/// <c>RFDetrDetector</c> of <c>surya/common/rfdetr/predictor.py</c> — preprocessing, <c>LWDETR.forward</c>
/// of <c>models/lwdetr.py</c> at inference, and <c>PostProcess</c>.
/// </summary>
public sealed class RfDetr
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly WindowedDinoV2 _backbone;
    private readonly MultiScaleProjector _projector;
    private readonly RfDetrTransformer _transformer;
    private readonly Dense _classEmbed;
    private readonly Mlp _bboxEmbed;
    private readonly string[] _labels;

    private RfDetr(StateDict state, int resolution, string[] labels)
    {
        Resolution = resolution;
        _labels = labels;
        _backbone = new WindowedDinoV2(state, windows: 2);
        _projector = new MultiScaleProjector(state);
        _transformer = new RfDetrTransformer(state, queries: 300);
        _classEmbed = state.Dense("class_embed");
        _bboxEmbed = Mlp.Load(state, "bbox_embed", 3);
    }

    /// <summary>The square input side (<c>config.json</c>'s <c>resolution</c>; 448 for the layout model).</summary>
    public int Resolution { get; }

    /// <summary>How many top-scoring (query, class) pairs <c>PostProcess</c> keeps (<c>num_select</c>).</summary>
    public int NumSelect { get; } = 100;

    /// <summary>See <see cref="RfDetrTransformer.ProposalOverride"/>.</summary>
    internal Func<int[], int[]>? ProposalOverride
    {
        get => _transformer.ProposalOverride;
        set => _transformer.ProposalOverride = value;
    }

    /// <summary>Stage timings of the last <see cref="Forward"/>, in milliseconds.</summary>
    public IReadOnlyDictionary<string, double> LastTimings { get; private set; } = new Dictionary<string, double>();

    /// <summary>Number of logits per query.</summary>
    public int Classes => _classEmbed.Out;

    /// <summary>Loads <c>config.json</c> and the <c>.pth</c> weights it names from <paramref name="directory"/>.</summary>
    public static RfDetr Load(string directory)
    {
        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        JsonElement root = config.RootElement;
        int resolution = root.TryGetProperty("resolution", out JsonElement r) ? r.GetInt32() : 704;

        // rf-detr's 0-indexed class ids line up with the categories sorted by id (RfDetrTorch.__init__).
        string[] labels = root.GetProperty("categories").EnumerateArray()
            .Select(c => (Id: c.GetProperty("id").GetInt32(), Name: c.GetProperty("name").GetString()!))
            .OrderBy(c => c.Id)
            .Select(c => c.Name)
            .ToArray();

        string? weights = root.TryGetProperty("weights", out JsonElement w) ? Path.Combine(directory, w.GetString()!) : null;
        if (weights is null || !File.Exists(weights))
        {
            weights = Directory.GetFiles(directory, "*.pth").Order(StringComparer.Ordinal).FirstOrDefault()
                ?? throw new FileNotFoundException($"no rf-detr .pth weights found in {directory}");
        }

        var state = new StateDict(TorchCheckpoint.Load(weights).StateDict("model"));
        return new RfDetr(state, resolution, labels);
    }

    /// <summary>
    /// <c>to_tensor</c> → <c>normalize(ImageNet)</c> → <c>resize((R, R))</c> (bilinear, antialiased) —
    /// the preprocessing of <c>RFDetrDetector.predict</c>; returns <c>[3, R, R]</c>.
    /// </summary>
    public float[] Preprocess(RgbImage image)
    {
        int w = image.Width, h = image.Height;
        float[] planes = new float[3 * h * w];
        ReadOnlySpan<byte> px = image.Pixels;
        for (int c = 0; c < 3; c++)
        {
            Span<float> plane = planes.AsSpan(c * h * w, h * w);
            float mean = Mean[c], std = Std[c];
            for (int i = 0; i < plane.Length; i++)
            {
                float v = px[(i * 3) + c] / 255f;
                plane[i] = (v - mean) / std;
            }
        }

        return TorchResize.Resize(planes, 3, h, w, Resolution, Resolution, TorchResize.Mode.Bilinear);
    }

    /// <summary>The raw model outputs of one page.</summary>
    /// <param name="Logits">Per query class logits, <c>[Q, classes]</c>.</param>
    /// <param name="Boxes">Per query <c>(cx, cy, w, h)</c>, normalised to the image, <c>[Q, 4]</c>.</param>
    /// <param name="Features">The projector's map, channels-last <c>[F·F, 256]</c> — the reading-order head's input.</param>
    /// <param name="FeatureSize">The map's side <c>F</c>.</param>
    public sealed record Output(float[] Logits, float[] Boxes, float[] Features, int FeatureSize);

    /// <summary>Runs <c>LWDETR.forward</c> on a preprocessed <c>[3, R, R]</c> tensor.</summary>
    public Output Forward(float[] pixels, LayoutTrace? trace = null)
    {
        long mark = System.Diagnostics.Stopwatch.GetTimestamp();
        float[][] maps = _backbone.Forward(pixels, Resolution, Resolution, trace);
        double backbone = System.Diagnostics.Stopwatch.GetElapsedTime(mark).TotalMilliseconds;
        mark = System.Diagnostics.Stopwatch.GetTimestamp();
        int side = Resolution / 16;
        float[] features = _projector.Forward(maps, side, side);
        if (trace is not null)
        {
            trace("proj", LayoutMath.Transpose(features, side * side, _projector.Channels));
        }

        double projector = System.Diagnostics.Stopwatch.GetElapsedTime(mark).TotalMilliseconds;
        mark = System.Diagnostics.Stopwatch.GetTimestamp();
        (float[] hs, float[] refs) = _transformer.Forward(features, side, side, trace);
        LastTimings = new Dictionary<string, double>
        {
            ["backbone"] = backbone,
            ["projector"] = projector,
            ["decoder"] = System.Diagnostics.Stopwatch.GetElapsedTime(mark).TotalMilliseconds,
        };
        int q = _transformer.Queries;
        float[] logits = _classEmbed.Forward(hs, q);
        float[] delta = _bboxEmbed.Forward(hs, q);
        float[] boxes = new float[q * 4];
        for (int i = 0; i < q; i++)
        {
            RfDetrTransformer.Reparam(delta.AsSpan(i * 4, 4), refs.AsSpan(i * 4, 4), boxes.AsSpan(i * 4, 4));
        }

        trace?.Invoke("pred_logits", logits);
        trace?.Invoke("pred_boxes", boxes);
        return new Output(logits, boxes, features, side);
    }

    /// <summary>
    /// <c>PostProcess.forward</c> followed by <c>predict</c>'s <c>scores &gt; threshold</c> filter and
    /// <c>detect</c>'s label lookup: sigmoid, top <see cref="NumSelect"/> over all (query, class)
    /// pairs, <c>cxcywh → xyxy</c> scaled to the original size.
    /// </summary>
    public List<RfDetrDetection> PostProcess(Output output, int width, int height, float threshold, LayoutTrace? trace = null)
    {
        int classes = Classes;
        int n = output.Logits.Length;
        float[] prob = new float[n];
        for (int i = 0; i < n; i++)
        {
            prob[i] = 1f / (1f + MathF.Exp(-output.Logits[i]));
        }

        int[] top = RfDetrTransformer.TopK(prob, NumSelect);
        var detections = new List<RfDetrDetection>();
        float[]? traced = trace is null ? null : new float[NumSelect * 4];
        for (int k = 0; k < top.Length; k++)
        {
            int query = top[k] / classes, label = top[k] % classes;
            ReadOnlySpan<float> b = output.Boxes.AsSpan(query * 4, 4);
            float halfW = 0.5f * Math.Max(b[2], 0f), halfH = 0.5f * Math.Max(b[3], 0f);
            float x0 = (b[0] - halfW) * width, y0 = (b[1] - halfH) * height;
            float x1 = (b[0] + halfW) * width, y1 = (b[1] + halfH) * height;
            if (traced is not null)
            {
                traced[(k * 4) + 0] = x0;
                traced[(k * 4) + 1] = y0;
                traced[(k * 4) + 2] = x1;
                traced[(k * 4) + 3] = y1;
            }

            if (prob[top[k]] > threshold)
            {
                string name = label < _labels.Length ? _labels[label] : label.ToString(System.Globalization.CultureInfo.InvariantCulture);
                detections.Add(new RfDetrDetection(name, label, prob[top[k]], x0, y0, x1, y1));
            }
        }

        if (trace is not null)
        {
            trace("post.scores", top.Select(i => prob[i]).ToArray());
            trace("post.labels", top.Select(i => (float)(i % classes)).ToArray());
            trace("post.boxes", traced);
        }

        return detections;
    }
}
