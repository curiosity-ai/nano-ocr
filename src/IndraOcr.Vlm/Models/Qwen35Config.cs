using System.Text.Json;

namespace IndraOcr.Vlm.Models;

/// <summary>
/// The parts of a <c>Qwen3_5ForConditionalGeneration</c> checkpoint's <c>config.json</c> and
/// <c>preprocessor_config.json</c> the forward pass reads (<c>configuration_qwen3_5.py</c>).
/// </summary>
public sealed class Qwen35Config
{
    // ---- language model (text_config) ----

    /// <summary>Decoder width.</summary>
    public int HiddenSize { get; init; }

    /// <summary>Decoder layers.</summary>
    public int NumLayers { get; init; }

    /// <summary><c>"linear_attention"</c> or <c>"full_attention"</c> per layer.</summary>
    public required string[] LayerTypes { get; init; }

    /// <summary>Query heads of the full-attention layers.</summary>
    public int NumHeads { get; init; }

    /// <summary>Key/value heads of the full-attention layers.</summary>
    public int NumKeyValueHeads { get; init; }

    /// <summary>Head width of the full-attention layers.</summary>
    public int HeadDim { get; init; }

    /// <summary>SwiGLU intermediate width.</summary>
    public int IntermediateSize { get; init; }

    /// <summary>Vocabulary (rows of the tied embedding).</summary>
    public int VocabSize { get; init; }

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; }

    /// <summary>Rotary base of the full-attention layers.</summary>
    public double RopeTheta { get; init; }

    /// <summary>Fraction of the head the rotary embedding covers.</summary>
    public double PartialRotaryFactor { get; init; }

    /// <summary>Interleaved M-RoPE sections <c>[t, h, w]</c>, in frequency pairs.</summary>
    public required int[] MropeSection { get; init; }

    /// <summary>Gated delta net: key heads.</summary>
    public int LinearKeyHeads { get; init; }

    /// <summary>Gated delta net: value heads.</summary>
    public int LinearValueHeads { get; init; }

    /// <summary>Gated delta net: key head width.</summary>
    public int LinearKeyHeadDim { get; init; }

    /// <summary>Gated delta net: value head width.</summary>
    public int LinearValueHeadDim { get; init; }

    /// <summary>Gated delta net: causal convolution kernel length.</summary>
    public int LinearConvKernel { get; init; }

    /// <summary>Whether the output head is the embedding matrix.</summary>
    public bool TieWordEmbeddings { get; init; }

    // ---- vision tower (vision_config) ----

    /// <summary>Vision blocks.</summary>
    public int VisionDepth { get; init; }

    /// <summary>Vision width.</summary>
    public int VisionHiddenSize { get; init; }

    /// <summary>Vision heads.</summary>
    public int VisionHeads { get; init; }

    /// <summary>Vision MLP width.</summary>
    public int VisionIntermediateSize { get; init; }

    /// <summary>Entries in the learned position table (a square grid).</summary>
    public int VisionPositionEmbeddings { get; init; }

    /// <summary>Patch side in pixels.</summary>
    public int PatchSize { get; init; }

    /// <summary>Frames per temporal patch (an image is repeated to fill it).</summary>
    public int TemporalPatchSize { get; init; }

    /// <summary>Side of the square of patches merged into one decoder token.</summary>
    public int SpatialMergeSize { get; init; }

    /// <summary>Width the merger projects to (the decoder's).</summary>
    public int VisionOutHiddenSize { get; init; }

    /// <summary>Rotary base of the vision tower.</summary>
    public double VisionRopeTheta { get; init; }

    // ---- special tokens ----

    /// <summary><c>&lt;|image_pad|&gt;</c>.</summary>
    public int ImageTokenId { get; init; }

    /// <summary><c>&lt;|vision_start|&gt;</c>.</summary>
    public int VisionStartTokenId { get; init; }

    // ---- image processor (preprocessor_config.json) ----

    /// <summary><c>size.shortest_edge</c>: the minimum pixel count of <c>smart_resize</c>.</summary>
    public int MinPixels { get; init; }

    /// <summary><c>size.longest_edge</c>: the maximum pixel count of <c>smart_resize</c>.</summary>
    public int MaxPixels { get; init; }

    /// <summary>Per-channel normalisation mean.</summary>
    public required float[] ImageMean { get; init; }

    /// <summary>Per-channel normalisation standard deviation.</summary>
    public required float[] ImageStd { get; init; }

    /// <summary>Head width of the vision tower.</summary>
    public int VisionHeadDim => VisionHiddenSize / VisionHeads;

    /// <summary>Rotary width of the full-attention layers.</summary>
    public int RotaryDim => (int)(HeadDim * PartialRotaryFactor);

    /// <summary>Key width of the gated delta net (all key heads).</summary>
    public int LinearKeyDim => LinearKeyHeads * LinearKeyHeadDim;

    /// <summary>Value width of the gated delta net (all value heads).</summary>
    public int LinearValueDim => LinearValueHeads * LinearValueHeadDim;

    /// <summary>Loads <c>config.json</c> and <c>preprocessor_config.json</c> from a checkpoint directory.</summary>
    public static Qwen35Config Load(string directory)
    {
        using JsonDocument config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "config.json")));
        string preprocessorPath = Path.Combine(directory, "preprocessor_config.json");
        using JsonDocument preprocessor = JsonDocument.Parse(
            File.Exists(preprocessorPath) ? File.ReadAllBytes(preprocessorPath) : "{}"u8.ToArray());
        return Parse(config.RootElement, preprocessor.RootElement);
    }

    internal static Qwen35Config Parse(JsonElement root, JsonElement preprocessor)
    {
        JsonElement text = root.GetProperty("text_config");
        JsonElement vision = root.GetProperty("vision_config");
        JsonElement rope = text.TryGetProperty("rope_parameters", out JsonElement r) ? r : text;

        int Int(JsonElement e, string name, int fallback) =>
            e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
        double Double(JsonElement e, string name, double fallback) =>
            e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;

        int hidden = Int(text, "hidden_size", 0);
        int heads = Int(text, "num_attention_heads", 0);
        int layers = Int(text, "num_hidden_layers", 0);
        string[] types = text.TryGetProperty("layer_types", out JsonElement lt)
            ? lt.EnumerateArray().Select(x => x.GetString()!).ToArray()
            : Enumerable.Range(0, layers)
                .Select(i => (i + 1) % Int(text, "full_attention_interval", 4) == 0 ? "full_attention" : "linear_attention")
                .ToArray();

        double visionTheta = 10000.0;
        if (vision.TryGetProperty("rope_parameters", out JsonElement vr) && vr.ValueKind == JsonValueKind.Object)
        {
            visionTheta = Double(vr, "rope_theta", visionTheta);
        }

        JsonElement size = preprocessor.TryGetProperty("size", out JsonElement s) ? s : default;
        float[] Floats(string name, float[] fallback) => preprocessor.TryGetProperty(name, out JsonElement v)
            ? v.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray()
            : fallback;

        return new Qwen35Config
        {
            HiddenSize = hidden,
            NumLayers = layers,
            LayerTypes = types,
            NumHeads = heads,
            NumKeyValueHeads = Int(text, "num_key_value_heads", heads),
            HeadDim = Int(text, "head_dim", hidden / Math.Max(1, heads)),
            IntermediateSize = Int(text, "intermediate_size", 0),
            VocabSize = Int(text, "vocab_size", 0),
            RmsNormEps = (float)Double(text, "rms_norm_eps", 1e-6),
            RopeTheta = Double(rope, "rope_theta", 10000.0),
            PartialRotaryFactor = Double(rope, "partial_rotary_factor", Double(text, "partial_rotary_factor", 1.0)),
            MropeSection = rope.TryGetProperty("mrope_section", out JsonElement ms)
                ? ms.EnumerateArray().Select(x => x.GetInt32()).ToArray()
                : [11, 11, 10],
            LinearKeyHeads = Int(text, "linear_num_key_heads", 16),
            LinearValueHeads = Int(text, "linear_num_value_heads", 32),
            LinearKeyHeadDim = Int(text, "linear_key_head_dim", 128),
            LinearValueHeadDim = Int(text, "linear_value_head_dim", 128),
            LinearConvKernel = Int(text, "linear_conv_kernel_dim", 4),
            TieWordEmbeddings = root.TryGetProperty("tie_word_embeddings", out JsonElement tie)
                ? tie.ValueKind == JsonValueKind.True
                : text.TryGetProperty("tie_word_embeddings", out JsonElement tie2) && tie2.ValueKind == JsonValueKind.True,
            VisionDepth = Int(vision, "depth", 0),
            VisionHiddenSize = Int(vision, "hidden_size", 0),
            VisionHeads = Int(vision, "num_heads", 0),
            VisionIntermediateSize = Int(vision, "intermediate_size", 0),
            VisionPositionEmbeddings = Int(vision, "num_position_embeddings", 2304),
            PatchSize = Int(vision, "patch_size", 16),
            TemporalPatchSize = Int(vision, "temporal_patch_size", 2),
            SpatialMergeSize = Int(vision, "spatial_merge_size", 2),
            VisionOutHiddenSize = Int(vision, "out_hidden_size", hidden),
            VisionRopeTheta = visionTheta,
            ImageTokenId = Int(root, "image_token_id", 248056),
            VisionStartTokenId = Int(root, "vision_start_token_id", 248053),
            MinPixels = size.ValueKind == JsonValueKind.Object ? Int(size, "shortest_edge", 65536) : 65536,
            MaxPixels = size.ValueKind == JsonValueKind.Object ? Int(size, "longest_edge", 16777216) : 16777216,
            ImageMean = Floats("image_mean", [0.5f, 0.5f, 0.5f]),
            ImageStd = Floats("image_std", [0.5f, 0.5f, 0.5f]),
        };
    }
}
