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

    /// <summary>
    /// Requests decoded together (vLLM batches concurrent requests too). Each holds its caches
    /// for the whole batch; 1 decodes one request at a time.
    /// </summary>
    public int DecodeBatch { get; init; } = 8;
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

    /// <summary>
    /// <c>chat_completions_batch</c>: runs the requests (decoded <see cref="SuryaInferenceOptions.DecodeBatch"/>
    /// at a time), then retries each failed or looping one on its own.
    /// </summary>
    public List<SuryaResponse> Generate(IReadOnlyList<SuryaRequest> batch)
    {
        var results = new SuryaResponse[batch.Count];
        int size = Math.Max(1, Options.DecodeBatch);
        for (int start = 0; start < batch.Count; start += size)
        {
            IReadOnlyList<SuryaRequest> group = batch.Skip(start).Take(size).ToList();
            var images = group.Select(r => ScaleToFit.Apply(r.Image)).ToList();
            try
            {
                List<SuryaResponse> first = GenerateOnce(images, group, Enumerable.Repeat(0, group.Count).ToList(), retry: -1);
                for (int i = 0; i < group.Count; i++)
                {
                    results[start + i] = Retry(images[i], group[i], first[i]);
                }
            }
            finally
            {
                images.ForEach(i => i.Dispose());
            }
        }

        return [.. results];
    }

    /// <summary>Runs one request with the retry policy.</summary>
    public SuryaResponse Generate(SuryaRequest request) => Generate([request])[0];

    private SuryaResponse Retry(RgbImage image, SuryaRequest request, SuryaResponse result)
    {
        for (int retries = 0; retries < Options.MaxRetries && (result.Error || RepeatDetector.HasRepeat(result.Raw)); retries++)
        {
            result = GenerateOnce([image], [request], [retries + 1], retries, result.Error)[0];
        }

        return result;
    }

    /// <summary>
    /// One decode of each request; <paramref name="retry"/> is -1 for the first attempt, else the
    /// retry index, whose temperature is <c>0.2·(n+1)</c> (capped at 0.8) and top-p 0.95 — or the
    /// default top-p after an error.
    /// </summary>
    private List<SuryaResponse> GenerateOnce(
        IReadOnlyList<RgbImage> images,
        IReadOnlyList<SuryaRequest> requests,
        IReadOnlyList<int> seeds,
        int retry,
        bool afterError = false)
    {
        var batch = new List<GenerationRequest>();
        for (int i = 0; i < requests.Count; i++)
        {
            SuryaRequest request = requests[i];
            float temperature = retry < 0 ? request.Temperature ?? Options.Temperature : Math.Min(Options.Temperature + (0.2f * (retry + 1)), 0.8f);
            float topP = retry < 0 ? request.TopP ?? Options.TopP : afterError ? Options.TopP : 0.95f;
            batch.Add(new GenerationRequest(images[i], request.Prompt, new GenerationOptions
            {
                MaxNewTokens = request.MaxTokens ?? Options.MaxTokensDefault,
                Temperature = temperature,
                TopP = topP,
                Seed = seeds[i],
                Logprobs = Options.Logprobs,
                Constraint = request.Constraint,
            }));
        }

        try
        {
            return Model.GenerateBatch(batch).Select(output => new SuryaResponse(
                output.Text,
                output.TokenIds.Count + (output.HitTokenLimit ? 0 : 1),
                false,
                output.MeanTokenProbability,
                output.TokenLogprobs,
                output.Timings)).ToList();
        }
        catch (Exception e) when (e is not OutOfMemoryException && requests.Count == 1)
        {
            return [new SuryaResponse(string.Empty, 0, true, null, null, new Dictionary<string, double>())];
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            // One bad request must not fail its batch: run them one by one.
            return requests.Select((r, i) => GenerateOnce([images[i]], [r], [seeds[i]], retry, afterError)[0]).ToList();
        }
    }

    /// <inheritdoc/>
    public void Dispose() => Model.Dispose();
}
