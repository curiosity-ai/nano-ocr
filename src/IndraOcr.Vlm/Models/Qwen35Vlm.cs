using System.Diagnostics;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Text;

namespace IndraOcr.Vlm.Models;

/// <summary>How a sequence is decoded.</summary>
public sealed record GenerationOptions
{
    /// <summary>Tokens generated at most.</summary>
    public int MaxNewTokens { get; init; } = 1024;

    /// <summary>0 decodes greedily; above it, samples from <c>softmax(logits / T)</c>.</summary>
    public float Temperature { get; init; }

    /// <summary>Nucleus mass kept when sampling.</summary>
    public float TopP { get; init; } = 1f;

    /// <summary>Sampling seed.</summary>
    public int Seed { get; init; }

    /// <summary>Whether to record each token's probability under the model.</summary>
    public bool Logprobs { get; init; }

    /// <summary>
    /// Called with the text so far after each token; returning true stops the decode (a cheap way
    /// to cut a runaway repetition before the budget is spent).
    /// </summary>
    public Func<string, bool>? StopWhen { get; init; }

    /// <summary>How often <see cref="StopWhen"/> is consulted, in tokens.</summary>
    public int StopCheckInterval { get; init; } = 32;

    /// <summary>
    /// Creates a fresh output constraint for this decode (guided decoding): only tokens it accepts
    /// are sampled, and the stop token only once it is complete.
    /// </summary>
    public Func<IOutputConstraint>? Constraint { get; init; }
}

/// <summary>A character-level grammar the decoded text must follow (vLLM's guided decoding).</summary>
public interface IOutputConstraint
{
    /// <summary>Whether the text so far is a complete sentence of the grammar.</summary>
    bool IsComplete { get; }

    /// <summary>Whether appending <paramref name="text"/> keeps the output a valid prefix.</summary>
    bool Accepts(string text);

    /// <summary>Appends <paramref name="text"/>, which <see cref="Accepts"/> approved.</summary>
    void Advance(string text);
}

/// <summary>One decode's output.</summary>
/// <param name="Text">The generated text, special tokens removed.</param>
/// <param name="TokenIds">The generated ids, the stop token excluded.</param>
/// <param name="MeanTokenProbability">The mean probability of the generated tokens, when requested.</param>
/// <param name="HitTokenLimit">Whether the decode stopped on <see cref="GenerationOptions.MaxNewTokens"/>.</param>
/// <param name="Timings">Milliseconds per stage: vision, prefill, decode.</param>
/// <param name="TokenLogprobs">Each generated token's log-probability (stop token included), when requested.</param>
public sealed record GenerationOutput(
    string Text,
    IReadOnlyList<int> TokenIds,
    double? MeanTokenProbability,
    bool HitTokenLimit,
    IReadOnlyDictionary<string, double> Timings,
    IReadOnlyList<double>? TokenLogprobs = null);

/// <summary>
/// A <c>Qwen3_5ForConditionalGeneration</c> checkpoint — Chandra 2 and Surya 2 are both this
/// architecture — loaded from a Hugging Face directory and run in pure C#.
/// </summary>
public sealed class Qwen35Vlm : IDisposable
{
    /// <summary>Tokens per prefill pass; bounds the activations a long prompt holds at once.</summary>
    private const int PrefillChunk = 1024;

    private readonly WeightStore _weights;
    private readonly int[] _stopTokens;
    private readonly string _generationPrefix;

    private Qwen35Vlm(string directory, int? minPixels, int? maxPixels)
    {
        Directory = directory;
        Config = Qwen35Config.Load(directory);
        _weights = WeightStore.Open(directory);
        Tokenizer = HfTokenizer.FromFile(Path.Combine(directory, "tokenizer.json"));
        Processor = new ImageProcessor(Config, minPixels, maxPixels);
        Vision = new VisionEncoder(_weights, Config);
        Decoder = new TextDecoder(_weights, Config);

        // Stop on <|im_end|> and <|endoftext|> (chandra/model/hf.py adds the first to the
        // generation config's second; vLLM stops on the chat template's end-of-turn too).
        _stopTokens = new[] { "<|im_end|>", "<|endoftext|>" }.Select(Tokenizer.TokenToId).Where(id => id >= 0).ToArray();

        // Chandra's template opens the assistant turn with an empty thinking block.
        string templatePath = Path.Combine(directory, "chat_template.jinja");
        string template = File.Exists(templatePath) ? File.ReadAllText(templatePath) : string.Empty;
        _generationPrefix = template.Contains(@"'<think>\n\n</think>\n\n'", StringComparison.Ordinal)
            ? "<think>\n\n</think>\n\n"
            : string.Empty;
    }

    /// <summary>The checkpoint directory.</summary>
    public string Directory { get; }

    /// <summary>The checkpoint's configuration.</summary>
    public Qwen35Config Config { get; }

    /// <summary>The checkpoint's tokenizer.</summary>
    public HfTokenizer Tokenizer { get; }

    /// <summary>The image processor.</summary>
    public ImageProcessor Processor { get; }

    /// <summary>The vision tower.</summary>
    public VisionEncoder Vision { get; }

    /// <summary>The language model.</summary>
    public TextDecoder Decoder { get; }

    /// <summary>Loads a checkpoint directory.</summary>
    /// <param name="directory">Holds <c>config.json</c>, <c>tokenizer.json</c> and the safetensors.</param>
    /// <param name="minPixels">Overrides the processor's minimum pixel count.</param>
    /// <param name="maxPixels">Overrides the processor's maximum pixel count.</param>
    public static Qwen35Vlm Load(string directory, int? minPixels = null, int? maxPixels = null) =>
        new(directory, minPixels, maxPixels);

    /// <summary>
    /// The chat-template rendering of one user turn holding an image and a prompt, with the
    /// image's placeholder expanded to its token count.
    /// </summary>
    public string BuildPrompt(string prompt, int imageTokens)
    {
        // render_content(...)|trim: the content is the vision markers then the prompt.
        string content = ("<|vision_start|>" + string.Concat(Enumerable.Repeat("<|image_pad|>", imageTokens))
            + "<|vision_end|>" + prompt).Trim();
        return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n" + _generationPrefix;
    }

    /// <summary>
    /// <c>get_rope_index</c> for a single image: text positions count up; the image's merged
    /// grid takes <c>(start, start + row, start + col)</c>; the text after it resumes at
    /// <c>start + max(rows, cols)</c>. Returns <c>[3, n]</c> and the decode offset.
    /// </summary>
    public (int[] Positions, int RopeDelta) RopeIndex(IReadOnlyList<int> ids, int mergedRows, int mergedCols)
    {
        int n = ids.Count;
        var positions = new int[3 * n];
        int current = 0;
        int i = 0;
        int max = -1;
        while (i < n)
        {
            if (ids[i] != Config.ImageTokenId)
            {
                positions[i] = positions[n + i] = positions[(2 * n) + i] = current;
                max = Math.Max(max, current);
                current++;
                i++;
                continue;
            }

            int start = current;
            for (int r = 0; r < mergedRows; r++)
            {
                for (int c = 0; c < mergedCols; c++, i++)
                {
                    positions[i] = start;
                    positions[n + i] = start + r;
                    positions[(2 * n) + i] = start + c;
                    max = Math.Max(max, Math.Max(start + r, start + c));
                }
            }

            current = start + Math.Max(mergedRows, mergedCols);
        }

        return (positions, max + 1 - n);
    }

    /// <summary>Runs the model on an image and a prompt.</summary>
    /// <param name="image">The image, already scaled by the caller's own policy.</param>
    /// <param name="prompt">The user prompt.</param>
    /// <param name="options">Decoding options.</param>
    /// <param name="trace">Receives intermediate tensors (parity tests).</param>
    /// <param name="profile">Receives stage timings.</param>
    public GenerationOutput Generate(
        RgbImage image,
        string prompt,
        GenerationOptions? options = null,
        TraceSink? trace = null,
        StageProfile? profile = null)
    {
        options ??= new GenerationOptions();
        var timings = new Dictionary<string, double>();
        long mark = Stopwatch.GetTimestamp();

        PreprocessedImage pixels = Processor.Process(image);
        trace?.Invoke("pixel_values", pixels.Pixels, pixels.Pixels.Length / pixels.PatchCount);
        float[] features = Vision.Encode(pixels, trace, profile);
        int merge = Config.SpatialMergeSize;
        int imageTokens = Vision.MergedTokens(pixels);
        timings["vision_ms"] = Stopwatch.GetElapsedTime(mark).TotalMilliseconds;

        List<int> ids = Tokenizer.Encode(BuildPrompt(prompt, imageTokens));
        (int[] positions, int delta) = RopeIndex(ids, pixels.GridH / merge, pixels.GridW / merge);
        return Decode(ids, positions, delta, features, options, timings, trace, profile);
    }

    /// <summary>Prefills <paramref name="ids"/> (image features scattered in) and decodes.</summary>
    internal GenerationOutput Decode(
        List<int> ids,
        int[] positions,
        int delta,
        float[] imageFeatures,
        GenerationOptions options,
        Dictionary<string, double> timings,
        TraceSink? trace,
        StageProfile? profile)
    {
        int width = Config.HiddenSize;
        int n = ids.Count;
        long mark = Stopwatch.GetTimestamp();

        var embeddings = new float[n * width];
        Decoder.Embed(ids.ToArray(), embeddings);
        int feature = 0;
        for (int i = 0; i < n; i++)
        {
            if (ids[i] == Config.ImageTokenId)
            {
                imageFeatures.AsSpan(feature * width, width).CopyTo(embeddings.AsSpan(i * width, width));
                feature++;
            }
        }

        trace?.Invoke("txt.embeds", embeddings, width);

        using DecoderState state = Decoder.CreateState(n + Math.Min(options.MaxNewTokens, 4096));
        state.RopeDelta = delta;

        // Prefill in chunks; the trace sees the whole prompt only when it fits in one.
        float[] last = new float[width];
        for (int start = 0; start < n; start += PrefillChunk)
        {
            int count = Math.Min(PrefillChunk, n - start);
            Memory<float> chunk = embeddings.AsMemory(start * width, count * width);
            int[] chunkPositions = new int[3 * count];
            for (int axis = 0; axis < 3; axis++)
            {
                Array.Copy(positions, (axis * n) + start, chunkPositions, axis * count, count);
            }

            Decoder.Forward(chunk, count, chunkPositions, state, n <= PrefillChunk ? trace : null, profile);
            if (start + count == n)
            {
                chunk.Span.Slice((count - 1) * width, width).CopyTo(last);
            }
        }

        timings["prefill_ms"] = Stopwatch.GetElapsedTime(mark).TotalMilliseconds;
        mark = Stopwatch.GetTimestamp();

        var logits = new float[Config.VocabSize];
        var generated = new List<int>();
        var random = new Random(options.Seed);
        double probabilitySum = 0;
        var logprobs = options.Logprobs ? new List<double>() : null;
        IOutputConstraint? constraint = options.Constraint?.Invoke();
        bool hitLimit = true;
        int[] step = new int[3];

        for (int t = 0; t < options.MaxNewTokens; t++)
        {
            Decoder.Logits(last, 1, logits);
            if (t < 4)
            {
                trace?.Invoke($"gen.logits{t}", logits, logits.Length);
            }

            int token = constraint is not null
                ? ConstrainedChoice(logits, constraint, options, random)
                : options.Temperature <= 0f ? ArgMax(logits) : Sample(logits, options.Temperature, options.TopP, random);
            if (options.Logprobs)
            {
                double p = Probability(logits, token);
                probabilitySum += p;
                logprobs!.Add(Math.Log(p));
            }

            if (constraint is not null && Array.IndexOf(_stopTokens, token) < 0)
            {
                constraint.Advance(TokenText(token));
            }

            if (Array.IndexOf(_stopTokens, token) >= 0)
            {
                hitLimit = false;
                break;
            }

            generated.Add(token);
            if (options.StopWhen is not null && generated.Count % options.StopCheckInterval == 0
                && options.StopWhen(Tokenizer.Decode(generated)))
            {
                hitLimit = false;
                break;
            }

            if (t + 1 == options.MaxNewTokens)
            {
                break;
            }

            Decoder.Embed([token], last);
            int position = state.Length + state.RopeDelta;
            step[0] = step[1] = step[2] = position;
            Decoder.Forward(last, 1, step, state, profile: profile);
        }

        timings["decode_ms"] = Stopwatch.GetElapsedTime(mark).TotalMilliseconds;
        int scored = generated.Count + (hitLimit ? 0 : 1);
        return new GenerationOutput(
            Tokenizer.Decode(generated),
            generated,
            options.Logprobs && scored > 0 ? probabilitySum / scored : null,
            hitLimit,
            timings,
            logprobs);
    }

    private readonly Dictionary<int, string> _tokenText = new();

    /// <summary>The text one token contributes to the decoded output.</summary>
    private string TokenText(int token)
    {
        lock (_tokenText)
        {
            if (!_tokenText.TryGetValue(token, out string? text))
            {
                text = Tokenizer.Decode([token], skipSpecialTokens: false);
                _tokenText[token] = text;
            }

            return text;
        }
    }

    /// <summary>
    /// Guided decoding: the most likely token the grammar accepts (greedy), or a sample from the
    /// accepted tokens' renormalised nucleus. The stop token is accepted only once the grammar is
    /// complete. Candidates are examined in logit order, so a greedy step usually tests one.
    /// </summary>
    private int ConstrainedChoice(float[] logits, IOutputConstraint constraint, GenerationOptions options, Random random)
    {
        bool Allowed(int token) => Array.IndexOf(_stopTokens, token) >= 0
            ? constraint.IsComplete
            : !Tokenizer.IsSpecial(token) && constraint.Accepts(TokenText(token));

        if (options.Temperature <= 0f)
        {
            int best = ArgMax(logits);
            if (Allowed(best))
            {
                return best;
            }
        }

        int[] order = Enumerable.Range(0, logits.Length).ToArray();
        float[] keys = new float[logits.Length];
        for (int i = 0; i < keys.Length; i++)
        {
            keys[i] = -logits[i];
        }

        Array.Sort(keys, order);

        if (options.Temperature <= 0f)
        {
            foreach (int token in order)
            {
                if (Allowed(token))
                {
                    return token;
                }
            }

            return _stopTokens[0];
        }

        // Sampling: softmax(logits / T) over the accepted tokens, then top-p. Tokens far below the
        // nucleus are never examined.
        float max = logits[order[0]] / options.Temperature;
        var accepted = new List<(int Token, double Weight)>();
        double total = 0;
        foreach (int token in order)
        {
            double weight = Math.Exp((logits[token] / options.Temperature) - max);
            if (weight < 1e-12 && accepted.Count > 0)
            {
                break;
            }

            if (Allowed(token))
            {
                accepted.Add((token, weight));
                total += weight;
            }
        }

        if (accepted.Count == 0)
        {
            return _stopTokens[0];
        }

        double cumulative = 0;
        int keep = 0;
        while (keep < accepted.Count)
        {
            cumulative += accepted[keep].Weight / total;
            keep++;
            if (cumulative >= options.TopP)
            {
                break;
            }
        }

        double kept = accepted.Take(keep).Sum(a => a.Weight);
        double draw = random.NextDouble() * kept;
        for (int i = 0; i < keep; i++)
        {
            draw -= accepted[i].Weight;
            if (draw <= 0)
            {
                return accepted[i].Token;
            }
        }

        return accepted[keep - 1].Token;
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static double Probability(ReadOnlySpan<float> logits, int token)
    {
        float max = float.NegativeInfinity;
        foreach (float l in logits)
        {
            max = Math.Max(max, l);
        }

        double sum = 0;
        foreach (float l in logits)
        {
            sum += Math.Exp(l - max);
        }

        return Math.Exp(logits[token] - max) / sum;
    }

    /// <summary>Temperature, then nucleus (top-p) sampling, as vLLM's sampler applies them.</summary>
    private static int Sample(ReadOnlySpan<float> logits, float temperature, float topP, Random random)
    {
        int n = logits.Length;
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            max = Math.Max(max, logits[i] / temperature);
        }

        var probabilities = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            probabilities[i] = Math.Exp((logits[i] / temperature) - max);
            sum += probabilities[i];
        }

        int[] order = Enumerable.Range(0, n).ToArray();
        Array.Sort(order, (a, b) => probabilities[b].CompareTo(probabilities[a]));
        double cumulative = 0;
        int keep = 0;
        while (keep < n)
        {
            cumulative += probabilities[order[keep]] / sum;
            keep++;
            if (cumulative >= topP)
            {
                break;
            }
        }

        double kept = 0;
        for (int i = 0; i < keep; i++)
        {
            kept += probabilities[order[i]];
        }

        double draw = random.NextDouble() * kept;
        for (int i = 0; i < keep; i++)
        {
            draw -= probabilities[order[i]];
            if (draw <= 0)
            {
                return order[i];
            }
        }

        return order[keep - 1];
    }

    /// <inheritdoc/>
    public void Dispose() => _weights.Dispose();
}
