using IndraOcr.Vlm.Common;
using IndraOcr.Vlm.Core;
using IndraOcr.Vlm.Formats;
using IndraOcr.Vlm.Models;
using IndraOcr.Vlm.Text;

namespace IndraOcr.Vlm.Surya;

/// <summary><c>surya.ocr_error.schema.OCRErrorDetectionResult</c>.</summary>
/// <param name="Texts">The inputs.</param>
/// <param name="Labels"><c>"good"</c> or <c>"bad"</c> per text.</param>
/// <param name="Scores">The probability of <c>"bad"</c> per text.</param>
public sealed record OcrErrorResult(IReadOnlyList<string> Texts, IReadOnlyList<string> Labels, IReadOnlyList<double> Scores);

/// <summary>
/// <c>surya/ocr_error</c>: a multilingual DistilBERT sequence classifier that flags garbled OCR
/// text (<c>DistilBertForSequenceClassification</c>: 6 layers, 768 wide, 12 heads, GELU, post-norm,
/// <c>[CLS]</c> → <c>pre_classifier</c> → ReLU → <c>classifier</c>). The checkpoint's float16
/// weights are widened to float32 at load, which is exact.
/// </summary>
public sealed class OcrErrorDetector : IDisposable
{
    /// <summary>Where Surya's downloader fetches the checkpoint from.</summary>
    public const string RemotePath = "ocr_error_detection/2025_02_18";

    /// <summary>The checkpoint's files.</summary>
    public static readonly string[] Files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors"];

    private const int Width = 768;
    private const int Heads = 12;
    private const int HeadDim = Width / Heads;
    private const int Hidden = 3072;
    private const float LayerNormEps = 1e-12f;

    private readonly WeightStore _weights;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly WeightTensor _wordEmbeddings;
    private readonly float[] _positions;
    private readonly float[] _embedNormWeight;
    private readonly float[] _embedNormBias;
    private readonly Layer[] _layers;
    private readonly WeightMatrix _preClassifier;
    private readonly float[] _preClassifierBias;
    private readonly WeightMatrix _classifier;
    private readonly float[] _classifierBias;

    private OcrErrorDetector(string directory)
    {
        _weights = WeightStore.Open(directory);
        _tokenizer = WordPieceTokenizer.FromFile(Path.Combine(directory, "tokenizer.json"));
        const string p = "distilbert.";
        _wordEmbeddings = _weights.Tensor(p + "embeddings.word_embeddings.weight");
        _positions = _weights.Vector(p + "embeddings.position_embeddings.weight");
        _embedNormWeight = _weights.Vector(p + "embeddings.LayerNorm.weight");
        _embedNormBias = _weights.Vector(p + "embeddings.LayerNorm.bias");
        _layers = new Layer[6];
        for (int i = 0; i < _layers.Length; i++)
        {
            string l = $"{p}transformer.layer.{i}.";
            _layers[i] = new Layer(
                Matrix(l + "attention.q_lin.weight"), _weights.Vector(l + "attention.q_lin.bias"),
                Matrix(l + "attention.k_lin.weight"), _weights.Vector(l + "attention.k_lin.bias"),
                Matrix(l + "attention.v_lin.weight"), _weights.Vector(l + "attention.v_lin.bias"),
                Matrix(l + "attention.out_lin.weight"), _weights.Vector(l + "attention.out_lin.bias"),
                _weights.Vector(l + "sa_layer_norm.weight"), _weights.Vector(l + "sa_layer_norm.bias"),
                Matrix(l + "ffn.lin1.weight"), _weights.Vector(l + "ffn.lin1.bias"),
                Matrix(l + "ffn.lin2.weight"), _weights.Vector(l + "ffn.lin2.bias"),
                _weights.Vector(l + "output_layer_norm.weight"), _weights.Vector(l + "output_layer_norm.bias"));
        }

        _preClassifier = Matrix("pre_classifier.weight");
        _preClassifierBias = _weights.Vector("pre_classifier.bias");
        _classifier = Matrix("classifier.weight");
        _classifierBias = _weights.Vector("classifier.bias");
    }

    /// <summary>
    /// Loads the checkpoint from <paramref name="directory"/>, <c>SURYA_OCR_ERROR_MODEL_DIR</c>,
    /// or Datalab's model store (cached).
    /// </summary>
    public static OcrErrorDetector Load(string? directory = null)
    {
        directory ??= Environment.GetEnvironmentVariable("SURYA_OCR_ERROR_MODEL_DIR");
        directory ??= ModelDownloader.EnsureDatalab(RemotePath, Files);
        return new OcrErrorDetector(directory);
    }

    /// <summary><c>OCRErrorPredictor.__call__</c>.</summary>
    public OcrErrorResult Predict(IReadOnlyList<string> texts)
    {
        var labels = new string[texts.Count];
        var scores = new double[texts.Count];
        for (int i = 0; i < texts.Count; i++)
        {
            float[] logits = Logits(texts[i]);
            float max = Math.Max(logits[0], logits[1]);
            double e0 = Math.Exp(logits[0] - max), e1 = Math.Exp(logits[1] - max);
            scores[i] = (float)(e1 / (e0 + e1));
            labels[i] = logits[1] > logits[0] ? "bad" : "good";
        }

        return new OcrErrorResult(texts, labels, scores);
    }

    /// <summary>The classifier's two logits (good, bad) for one text.</summary>
    public float[] Logits(string text)
    {
        List<int> ids = _tokenizer.Encode(text, 512);
        int n = ids.Count;
        var hidden = new float[n * Width];
        var row = new float[Width];
        int bytesPerRow = Width * 2;
        for (int t = 0; t < n; t++)
        {
            FloatConversion.ToFloat(_wordEmbeddings.Bytes.Span.Slice(ids[t] * bytesPerRow, bytesPerRow), _wordEmbeddings.Dtype, row);
            Span<float> target = hidden.AsSpan(t * Width, Width);
            ReadOnlySpan<float> position = _positions.AsSpan(t * Width, Width);
            for (int i = 0; i < Width; i++)
            {
                target[i] = row[i] + position[i];
            }
        }

        Norms.LayerNorm(hidden, Width, _embedNormWeight, _embedNormBias, LayerNormEps);

        var q = new float[n * Width];
        var k = new float[n * Width];
        var v = new float[n * Width];
        var qh = new float[n * Width];
        var kh = new float[n * Width];
        var vh = new float[n * Width];
        var context = new float[n * Width];
        var merged = new float[n * Width];
        var projected = new float[n * Width];
        var inner = new float[n * Hidden];
        float scale = 1f / MathF.Sqrt(HeadDim);
        foreach (Layer layer in _layers)
        {
            Gemm.Linear(hidden, n, Width, layer.Q, layer.QBias, q, Width);
            Gemm.Linear(hidden, n, Width, layer.K, layer.KBias, k, Width);
            Gemm.Linear(hidden, n, Width, layer.V, layer.VBias, v, Width);
            for (int h = 0; h < Heads; h++)
            {
                for (int t = 0; t < n; t++)
                {
                    q.AsSpan((t * Width) + (h * HeadDim), HeadDim).CopyTo(qh.AsSpan(((h * n) + t) * HeadDim, HeadDim));
                    k.AsSpan((t * Width) + (h * HeadDim), HeadDim).CopyTo(kh.AsSpan(((h * n) + t) * HeadDim, HeadDim));
                    v.AsSpan((t * Width) + (h * HeadDim), HeadDim).CopyTo(vh.AsSpan(((h * n) + t) * HeadDim, HeadDim));
                }
            }

            Attention.Bidirectional(qh, kh, vh, context, Heads, n, HeadDim, scale);
            for (int h = 0; h < Heads; h++)
            {
                for (int t = 0; t < n; t++)
                {
                    context.AsSpan(((h * n) + t) * HeadDim, HeadDim).CopyTo(merged.AsSpan((t * Width) + (h * HeadDim), HeadDim));
                }
            }

            Gemm.Linear(merged, n, Width, layer.Out, layer.OutBias, projected, Width);
            Kernels.AddInPlace(projected, hidden);
            Norms.LayerNorm(projected, Width, layer.SaNormWeight, layer.SaNormBias, LayerNormEps);
            projected.AsSpan().CopyTo(hidden);

            Gemm.Linear(hidden, n, Width, layer.Lin1, layer.Lin1Bias, inner, Hidden);
            Kernels.GeluErf(inner);
            Gemm.Linear(inner, n, Hidden, layer.Lin2, layer.Lin2Bias, projected, Width);
            Kernels.AddInPlace(projected, hidden);
            Norms.LayerNorm(projected, Width, layer.OutNormWeight, layer.OutNormBias, LayerNormEps);
            projected.AsSpan().CopyTo(hidden);
        }

        var pooled = new float[Width];
        Gemm.Linear(hidden.AsMemory(0, Width), 1, Width, _preClassifier, _preClassifierBias, pooled, Width);
        for (int i = 0; i < Width; i++)
        {
            pooled[i] = Math.Max(0f, pooled[i]);
        }

        var logits = new float[2];
        Gemm.Linear(pooled, 1, Width, _classifier, _classifierBias, logits, 2);
        return logits;
    }

    private WeightMatrix Matrix(string name)
    {
        WeightTensor tensor = _weights.Tensor(name);
        return WeightMatrix.FromFloats(tensor.ToFloats(), tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc/>
    public void Dispose() => _weights.Dispose();

    private sealed record Layer(
        WeightMatrix Q, float[] QBias, WeightMatrix K, float[] KBias, WeightMatrix V, float[] VBias,
        WeightMatrix Out, float[] OutBias, float[] SaNormWeight, float[] SaNormBias,
        WeightMatrix Lin1, float[] Lin1Bias, WeightMatrix Lin2, float[] Lin2Bias,
        float[] OutNormWeight, float[] OutNormBias);
}
