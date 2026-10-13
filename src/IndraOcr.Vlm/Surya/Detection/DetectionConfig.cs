using System.Text.Json;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>
/// The detector checkpoint's <c>config.json</c> (<c>EfficientViTConfig</c>,
/// <c>surya/detection/model/config.py</c>) and <c>preprocessor_config.json</c>
/// (<c>SegformerImageProcessor</c>, <c>surya/detection/processor.py</c>).
/// </summary>
public sealed class DetectionConfig
{
    /// <summary>Input channels.</summary>
    public int NumChannels { get; init; } = 3;

    /// <summary>Stem width followed by the four stage widths.</summary>
    public int[] Widths { get; init; } = [32, 64, 128, 256, 512];

    /// <summary>Stem depth followed by the four stage depths.</summary>
    public int[] Depths { get; init; } = [1, 1, 1, 6, 6];

    /// <summary>Stem stride followed by the four stage strides.</summary>
    public int[] Strides { get; init; } = [2, 2, 2, 2, 2];

    /// <summary>LiteMLA head width.</summary>
    public int HeadDim { get; init; } = 32;

    /// <summary>The backbone's batch-norm epsilon (<c>layer_norm_eps</c>).</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Width each stage is projected to by the decode head's <c>linear_c</c>.</summary>
    public int DecoderLayerHiddenSize { get; init; } = 128;

    /// <summary>Width of the decode head's fused map.</summary>
    public int DecoderHiddenSize { get; init; } = 512;

    /// <summary>Stages the decode head fuses.</summary>
    public int NumStages { get; init; } = 4;

    /// <summary>
    /// Output maps — <c>PretrainedConfig.num_labels</c>, which the checkpoint leaves at its default
    /// of 2 (the text heatmap and the affinity map).
    /// </summary>
    public int NumLabels { get; init; } = 2;

    /// <summary>The processor's output height (and the chunk height <c>split_image</c> uses).</summary>
    public int ImageHeight { get; init; } = 1200;

    /// <summary>The processor's output width.</summary>
    public int ImageWidth { get; init; } = 1200;

    /// <summary>Whether pixels are multiplied by <see cref="RescaleFactor"/>.</summary>
    public bool DoRescale { get; init; } = true;

    /// <summary><c>rescale_factor</c>, applied in float64.</summary>
    public double RescaleFactor { get; init; } = 1 / 255.0;

    /// <summary>Whether the mean is subtracted and the result divided by the deviation.</summary>
    public bool DoNormalize { get; init; } = true;

    /// <summary>Per-channel mean.</summary>
    public double[] ImageMean { get; init; } = [0.485, 0.456, 0.406];

    /// <summary>Per-channel standard deviation.</summary>
    public double[] ImageStd { get; init; } = [0.229, 0.224, 0.225];

    /// <summary>Reads both files from a checkpoint directory.</summary>
    public static DetectionConfig Load(string directory)
    {
        using JsonDocument config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "config.json")));
        JsonElement c = config.RootElement;
        string processorPath = Path.Combine(directory, "preprocessor_config.json");
        using JsonDocument processor = JsonDocument.Parse(File.Exists(processorPath) ? File.ReadAllBytes(processorPath) : "{}"u8.ToArray());
        JsonElement p = processor.RootElement;

        var defaults = new DetectionConfig();
        int numLabels = defaults.NumLabels;
        if (c.TryGetProperty("id2label", out JsonElement labels) && labels.ValueKind == JsonValueKind.Object)
        {
            numLabels = labels.EnumerateObject().Count();
        }
        else if (c.TryGetProperty("num_labels", out JsonElement n))
        {
            numLabels = n.GetInt32();
        }

        JsonElement size = p.TryGetProperty("size", out JsonElement s) ? s : default;
        return new DetectionConfig
        {
            NumChannels = Int(c, "num_channels", defaults.NumChannels),
            Widths = Ints(c, "widths", defaults.Widths),
            Depths = Ints(c, "depths", defaults.Depths),
            Strides = Ints(c, "strides", defaults.Strides),
            HeadDim = Int(c, "head_dim", defaults.HeadDim),
            NormEps = c.TryGetProperty("layer_norm_eps", out JsonElement eps) ? (float)eps.GetDouble() : defaults.NormEps,
            DecoderLayerHiddenSize = Int(c, "decoder_layer_hidden_size", defaults.DecoderLayerHiddenSize),
            DecoderHiddenSize = Int(c, "decoder_hidden_size", defaults.DecoderHiddenSize),
            NumStages = Int(c, "num_stages", defaults.NumStages),
            NumLabels = numLabels,
            ImageHeight = size.ValueKind == JsonValueKind.Object ? Int(size, "height", 512) : 512,
            ImageWidth = size.ValueKind == JsonValueKind.Object ? Int(size, "width", 512) : 512,
            DoRescale = !p.TryGetProperty("do_rescale", out JsonElement r) || r.GetBoolean(),
            RescaleFactor = p.TryGetProperty("rescale_factor", out JsonElement f) ? f.GetDouble() : defaults.RescaleFactor,
            DoNormalize = !p.TryGetProperty("do_normalize", out JsonElement dn) || dn.GetBoolean(),
            ImageMean = Doubles(p, "image_mean", defaults.ImageMean),
            ImageStd = Doubles(p, "image_std", defaults.ImageStd),
        };
    }

    private static int Int(JsonElement e, string name, int fallback) =>
        e.TryGetProperty(name, out JsonElement v) ? v.GetInt32() : fallback;

    private static int[] Ints(JsonElement e, string name, int[] fallback) =>
        e.TryGetProperty(name, out JsonElement v) ? v.EnumerateArray().Select(x => x.GetInt32()).ToArray() : fallback;

    private static double[] Doubles(JsonElement e, string name, double[] fallback) =>
        e.TryGetProperty(name, out JsonElement v) ? v.EnumerateArray().Select(x => x.GetDouble()).ToArray() : fallback;
}
