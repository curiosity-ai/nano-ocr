using IndraOcr.Vlm.Common;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Models;

namespace IndraOcr.Vlm.Surya;

/// <summary>One request (<c>surya.inference.schema.BatchInputItem</c>).</summary>
public sealed record SuryaRequest
{
    /// <summary>The image; <c>scale_to_fit</c> is applied before the model sees it.</summary>
    public required RgbImage Image { get; init; }

    /// <summary>The prompt text.</summary>
    public required string Prompt { get; init; }

    /// <summary>Token budget.</summary>
    public int? MaxTokens { get; init; }

    /// <summary>Temperature override.</summary>
    public float? Temperature { get; init; }

    /// <summary>Top-p override.</summary>
    public float? TopP { get; init; }

    /// <summary>Guided decoding constraint factory (the schema the server enforces).</summary>
    public Func<IOutputConstraint>? Constraint { get; init; }
}

/// <summary>One response (<c>surya.inference.schema.BatchOutputItem</c>).</summary>
public sealed record SuryaResponse(
    string Raw,
    int TokenCount,
    bool Error,
    double? MeanTokenProb,
    IReadOnlyList<double>? Logprobs,
    IReadOnlyDictionary<string, double> Timings);

/// <summary>Options of the inference layer (<c>surya/settings.py</c> and the backend defaults).</summary>
public sealed record SuryaInferenceOptions
{
    /// <summary>
    /// <c>min_pixels</c> the vLLM server is started with (<c>--mm-processor-kwargs</c>), which
    /// overrides the checkpoint's processor config.
    /// </summary>
    public int MinPixels { get; init; } = 3136;

    /// <summary><c>max_pixels</c> the vLLM server is started with.</summary>
    public int MaxPixels { get; init; } = 6291456;

    /// <summary>Budget when a request names none (<c>chat_completions_batch</c>'s 2048).</summary>
    public int MaxTokensDefault { get; init; } = 2048;

    /// <summary>Default temperature (greedy).</summary>
    public float Temperature { get; init; }

    /// <summary>Default top-p.</summary>
    public float TopP { get; init; } = 0.1f;

    /// <summary>Retries on an error or a runaway repetition, at rising temperature.</summary>
    public int MaxRetries { get; init; } = 3;

    /// <summary><c>SURYA_INFERENCE_LOGPROBS</c>: record token probabilities (the blocks' confidence).</summary>
    public bool Logprobs { get; init; } = true;
}

/// <summary>
/// The OpenAI-compatible client of <c>surya/inference/backends/openai_client.py</c> with the
/// server replaced by <see cref="Qwen35Vlm"/>: <c>scale_to_fit</c>, the chat template, greedy
/// decoding with the server's guided JSON where asked, token probabilities, and up to three
/// retries on error or repetition at temperature <c>0.2·(n+1)</c> (capped at 0.8) and top-p 0.95.
/// </summary>
public sealed class SuryaInference : IDisposable
{
    /// <summary>The Hub repository of the checkpoint.</summary>
    public const string Repository = "datalab-to/surya-ocr-2";

    /// <summary>The checkpoint's files.</summary>
    public static readonly string[] Files =
    [
        "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja", "model.safetensors",
    ];

    private SuryaInference(Qwen35Vlm model, SuryaInferenceOptions options)
    {
        Model = model;
        Options = options;
    }

    /// <summary>The model.</summary>
    public Qwen35Vlm Model { get; }

    /// <summary>The options.</summary>
    public SuryaInferenceOptions Options { get; }

    /// <summary>
    /// Loads the checkpoint from <paramref name="modelDirectory"/>, <c>SURYA_MODEL_DIR</c>, or the
    /// download cache.
    /// </summary>
    public static SuryaInference Load(string? modelDirectory = null, SuryaInferenceOptions? options = null, Action<string, long, long>? progress = null)
    {
        options ??= new SuryaInferenceOptions();
        modelDirectory ??= Environment.GetEnvironmentVariable("SURYA_MODEL_DIR");
        modelDirectory ??= ModelDownloader.Ensure(Repository, Files, progress);
        return new SuryaInference(Qwen35Vlm.Load(modelDirectory, options.MinPixels, options.MaxPixels), options);
    }

    /// <summary>Runs requests in order (<c>chat_completions_batch</c>).</summary>
    public List<SuryaResponse> Generate(IEnumerable<SuryaRequest> batch) => batch.Select(Generate).ToList();

    /// <summary>Runs one request with the retry policy.</summary>
    public SuryaResponse Generate(SuryaRequest request)
    {
        using RgbImage image = ScaleToFit.Apply(request.Image);
        float temperature = request.Temperature ?? Options.Temperature;
        float topP = request.TopP ?? Options.TopP;
        SuryaResponse result = GenerateOnce(image, request, temperature, topP, seed: 0);
        for (int retries = 0; retries < Options.MaxRetries && (result.Error || RepeatDetector.HasRepeat(result.Raw)); retries++)
        {
            float retryTemperature = Math.Min(Options.Temperature + (0.2f * (retries + 1)), 0.8f);
            float retryTopP = result.Error ? Options.TopP : 0.95f;
            result = GenerateOnce(image, request, retryTemperature, retryTopP, seed: retries + 1);
        }

        return result;
    }

    private SuryaResponse GenerateOnce(RgbImage image, SuryaRequest request, float temperature, float topP, int seed)
    {
        try
        {
            GenerationOutput output = Model.Generate(image, request.Prompt, new GenerationOptions
            {
                MaxNewTokens = request.MaxTokens ?? Options.MaxTokensDefault,
                Temperature = temperature,
                TopP = topP,
                Seed = seed,
                Logprobs = Options.Logprobs,
                Constraint = request.Constraint,
            });
            int tokens = output.TokenIds.Count + (output.HitTokenLimit ? 0 : 1);
            return new SuryaResponse(output.Text, tokens, false, output.MeanTokenProbability, output.TokenLogprobs, output.Timings);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return new SuryaResponse(string.Empty, 0, true, null, null, new Dictionary<string, double>());
        }
    }

    /// <inheritdoc/>
    public void Dispose() => Model.Dispose();
}
