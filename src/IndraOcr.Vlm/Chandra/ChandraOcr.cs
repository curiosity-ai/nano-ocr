using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using IndraOcr.Pdf;
using IndraOcr.Vlm.Common;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Models;

namespace IndraOcr.Vlm.Chandra;

/// <summary>Options of a Chandra run (<c>chandra/settings.py</c> and the CLI's flags).</summary>
public sealed record ChandraOptions
{
    /// <summary><c>"ocr_layout"</c> (the CLI's) or <c>"ocr"</c>.</summary>
    public string PromptType { get; init; } = "ocr_layout";

    /// <summary>A custom prompt, replacing <see cref="PromptType"/>.</summary>
    public string? Prompt { get; init; }

    /// <summary><c>MAX_OUTPUT_TOKENS</c>.</summary>
    public int MaxOutputTokens { get; init; } = 12384;

    /// <summary><c>BBOX_SCALE</c>.</summary>
    public int BboxScale { get; init; } = 1000;

    /// <summary>Keep Page-Header and Page-Footer blocks.</summary>
    public bool IncludeHeadersFooters { get; init; }

    /// <summary>Keep Image and Figure blocks (and extract their crops).</summary>
    public bool IncludeImages { get; init; } = true;

    /// <summary>
    /// Re-decode with sampling when the output ends in a runaway repetition, as Chandra's vLLM
    /// path does (temperature <c>0.2·(n+1)</c> capped at 0.8, top-p 0.95). Its local transformers
    /// path does not retry; 0 matches that.
    /// </summary>
    public int MaxRetries { get; init; }

    /// <summary><c>IMAGE_DPI</c>.</summary>
    public int ImageDpi { get; init; } = 192;

    /// <summary><c>MIN_PDF_IMAGE_DIM</c>.</summary>
    public int MinPdfImageDim { get; init; } = 1024;

    /// <summary><c>MIN_IMAGE_DIM</c>.</summary>
    public int MinImageDim { get; init; } = 1536;
}

/// <summary>One page's result (<c>chandra.model.schema.BatchOutputItem</c>).</summary>
public sealed class ChandraPage
{
    /// <summary>Markdown of the page.</summary>
    public required string Markdown { get; init; }

    /// <summary>HTML of the page (layout divs unwrapped).</summary>
    public required string Html { get; init; }

    /// <summary>Layout blocks with pixel boxes.</summary>
    public required IReadOnlyList<LayoutBlock> Chunks { get; init; }

    /// <summary>The model's raw output.</summary>
    public required string Raw { get; init; }

    /// <summary><c>[0, 0, width, height]</c> of the page image.</summary>
    public required int[] PageBox { get; init; }

    /// <summary>Tokens generated.</summary>
    public required int TokenCount { get; init; }

    /// <summary>Crops of the figures, by the file name the HTML references.</summary>
    [JsonIgnore]
    public required IReadOnlyDictionary<string, RgbImage> Images { get; init; }

    /// <summary>Whether generation failed.</summary>
    public bool Error { get; init; }

    /// <summary>Milliseconds per stage.</summary>
    public IReadOnlyDictionary<string, double> Timings { get; init; } = new Dictionary<string, double>();
}

/// <summary>
/// Chandra 2 (<c>datalab-to/chandra-ocr-2</c>) in pure C#: page images or PDFs to layout-aware
/// HTML and markdown, with figure crops — the <c>chandra</c> package's <c>InferenceManager</c>
/// and CLI.
/// </summary>
public sealed class ChandraOcr : IDisposable
{
    /// <summary>The Hub repository of the checkpoint.</summary>
    public const string Repository = "datalab-to/chandra-ocr-2";

    /// <summary>The checkpoint's files.</summary>
    public static readonly string[] Files =
    [
        "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja", "model.safetensors",
    ];

    private readonly Qwen35Vlm _model;

    private ChandraOcr(Qwen35Vlm model, ChandraOptions options)
    {
        _model = model;
        Options = options;
    }

    /// <summary>The options this instance runs with.</summary>
    public ChandraOptions Options { get; }

    /// <summary>The underlying model.</summary>
    public Qwen35Vlm Model => _model;

    /// <summary>
    /// Loads the checkpoint from <paramref name="modelDirectory"/>, or downloads it to the cache
    /// (<see cref="ModelDownloader"/>) when none is given. <c>CHANDRA_MODEL_DIR</c> also names a
    /// local copy.
    /// </summary>
    public static ChandraOcr Load(string? modelDirectory = null, ChandraOptions? options = null, Action<string, long, long>? progress = null)
    {
        modelDirectory ??= Environment.GetEnvironmentVariable("CHANDRA_MODEL_DIR");
        modelDirectory ??= ModelDownloader.Ensure(Repository, Files, progress);
        return new ChandraOcr(Qwen35Vlm.Load(modelDirectory), options ?? new ChandraOptions());
    }

    /// <summary><c>load_image</c>: upscales so the shorter side is at least <c>MIN_IMAGE_DIM</c> (Lanczos).</summary>
    public RgbImage PrepareImage(RgbImage image)
    {
        if (image.Width < Options.MinImageDim || image.Height < Options.MinImageDim)
        {
            double scale = Options.MinImageDim / (double)Math.Min(image.Width, image.Height);
            return PilResize.ResizeLanczos(image, (int)(image.Width * scale), (int)(image.Height * scale));
        }

        return image.Clone();
    }

    /// <summary>
    /// <c>load_pdf_images</c>: renders each selected page at <c>max(IMAGE_DPI, MIN_PDF_IMAGE_DIM /
    /// min side · 72)</c> dpi after flattening its annotations and form fields.
    /// </summary>
    public List<RgbImage> RenderPdf(string path, IReadOnlyCollection<int>? pages = null, string? password = null)
    {
        using PdfDocument document = PdfDocument.Open(path, password);
        var images = new List<RgbImage>();
        for (int index = 0; index < document.PageCount; index++)
        {
            if (pages is { Count: > 0 } && !pages.Contains(index))
            {
                continue;
            }

            double minSide;
            using (PdfPage probe = document.GetPage(index))
            {
                minSide = Math.Min(probe.Width, probe.Height);
                probe.Flatten();
            }

            double dpi = Math.Max(Options.MinPdfImageDim / minSide * 72, Options.ImageDpi);
            using PdfPage page = document.GetPage(index);
            (byte[] rgb, int width, int height) = page.RenderRgb(dpi / 72);
            images.Add(RgbImage.Wrap(rgb, width, height));
        }

        return images;
    }

    /// <summary><c>load_file</c>: a PDF's pages, or one image.</summary>
    public List<RgbImage> LoadFile(string path, IReadOnlyCollection<int>? pages = null)
    {
        if (PdfDocument.IsPdf(path))
        {
            return RenderPdf(path, pages);
        }

        using RgbImage image = RgbImageIO.Read(path);
        return [PrepareImage(image)];
    }

    /// <summary>Runs one page image through the model and parses the result.</summary>
    public ChandraPage Predict(RgbImage page)
    {
        string prompt = Options.Prompt ?? ChandraPrompts.ForType(Options.PromptType);
        using RgbImage scaled = ScaleToFit.Apply(page);

        GenerationOutput output = _model.Generate(scaled, prompt, new GenerationOptions { MaxNewTokens = Options.MaxOutputTokens });
        for (int retry = 0; retry < Options.MaxRetries && RepeatDetector.HasRepeat(output.Text); retry++)
        {
            output = _model.Generate(scaled, prompt, new GenerationOptions
            {
                MaxNewTokens = Options.MaxOutputTokens,
                Temperature = Math.Min(0.2f * (retry + 1), 0.8f),
                TopP = 0.95f,
                Seed = retry + 1,
            });
        }

        string raw = output.Text;
        List<LayoutBlock> chunks = ChandraOutput.ParseLayout(raw, page.Width, page.Height, Options.BboxScale);
        return new ChandraPage
        {
            Markdown = ChandraOutput.ParseMarkdown(raw, Options.IncludeHeadersFooters, Options.IncludeImages),
            Html = ChandraOutput.ParseHtml(raw, Options.IncludeHeadersFooters, Options.IncludeImages),
            Chunks = chunks,
            Raw = raw,
            PageBox = [0, 0, page.Width, page.Height],
            TokenCount = output.TokenIds.Count + (output.HitTokenLimit ? 0 : 1),
            Images = ChandraOutput.ExtractImages(raw, chunks, page),
            Timings = output.Timings,
        };
    }

    /// <summary>Runs every page of a file.</summary>
    public List<ChandraPage> PredictFile(string path, IReadOnlyCollection<int>? pages = null, Action<int, int>? onPage = null)
    {
        List<RgbImage> images = LoadFile(path, pages);
        var results = new List<ChandraPage>();
        for (int i = 0; i < images.Count; i++)
        {
            onPage?.Invoke(i, images.Count);
            try
            {
                results.Add(Predict(images[i]));
            }
            finally
            {
                images[i].Dispose();
            }
        }

        return results;
    }

    /// <summary><c>parse_range_str</c>: <c>"1-5,7,9-12"</c> to sorted, distinct page indices.</summary>
    public static List<int> ParseRange(string range)
    {
        var pages = new SortedSet<int>();
        foreach (string part in range.Split(','))
        {
            if (part.Contains('-'))
            {
                string[] ends = part.Split('-');
                for (int p = int.Parse(ends[0]); p <= int.Parse(ends[1]); p++)
                {
                    pages.Add(p);
                }
            }
            else
            {
                pages.Add(int.Parse(part));
            }
        }

        return [.. pages];
    }

    /// <summary>
    /// <c>save_merged_output</c>: <c>&lt;out&gt;/&lt;stem&gt;/&lt;stem&gt;.md</c>, <c>.html</c>,
    /// <c>_metadata.json</c> and the figure crops.
    /// </summary>
    public static string SaveMergedOutput(
        string outputDirectory,
        string fileName,
        IReadOnlyList<ChandraPage> results,
        bool saveImages = true,
        bool saveHtml = true,
        bool paginate = false)
    {
        string stem = Path.GetFileNameWithoutExtension(fileName);
        string directory = Path.Combine(outputDirectory, stem);
        Directory.CreateDirectory(directory);

        var markdown = new StringBuilder();
        var html = new StringBuilder();
        var pages = new List<object>();
        int totalTokens = 0, totalChunks = 0, totalImages = 0;
        for (int n = 0; n < results.Count; n++)
        {
            ChandraPage result = results[n];
            if (n > 0 && paginate)
            {
                markdown.Append($"\n\n{n}").Append('-', 48).Append("\n\n");
                html.Append($"\n\n<!-- Page {n + 1} -->\n\n");
            }

            markdown.Append(result.Markdown);
            html.Append(result.Html);
            if (!paginate && n < results.Count - 1)
            {
                markdown.Append("\n\n");
                html.Append("\n\n");
            }

            totalTokens += result.TokenCount;
            totalChunks += result.Chunks.Count;
            totalImages += result.Images.Count;
            pages.Add(new PageMetadata(n, result.PageBox, result.TokenCount, result.Chunks.Count, result.Images.Count));

            if (saveImages)
            {
                foreach ((string name, RgbImage image) in result.Images)
                {
                    File.WriteAllBytes(Path.Combine(directory, name), RgbImageIO.EncodeWebp(image));
                }
            }
        }

        string markdownPath = Path.Combine(directory, stem + ".md");
        File.WriteAllText(markdownPath, markdown.ToString());
        if (saveHtml)
        {
            File.WriteAllText(Path.Combine(directory, stem + ".html"), html.ToString());
        }

        var metadata = new DocumentMetadata(fileName, results.Count, totalTokens, totalChunks, totalImages, pages.Cast<PageMetadata>().ToList());
        File.WriteAllText(
            Path.Combine(directory, stem + "_metadata.json"),
            JsonSerializer.Serialize(metadata, ChandraJsonContext.Default.DocumentMetadata));
        return markdownPath;
    }

    /// <inheritdoc/>
    public void Dispose() => _model.Dispose();
}

/// <summary>Per-page entry of <c>_metadata.json</c>.</summary>
public sealed record PageMetadata(
    [property: JsonPropertyName("page_num")] int PageNum,
    [property: JsonPropertyName("page_box")] int[] PageBox,
    [property: JsonPropertyName("token_count")] int TokenCount,
    [property: JsonPropertyName("num_chunks")] int NumChunks,
    [property: JsonPropertyName("num_images")] int NumImages);

/// <summary><c>_metadata.json</c>.</summary>
public sealed record DocumentMetadata(
    [property: JsonPropertyName("file_name")] string FileName,
    [property: JsonPropertyName("num_pages")] int NumPages,
    [property: JsonPropertyName("total_token_count")] int TotalTokenCount,
    [property: JsonPropertyName("total_chunks")] int TotalChunks,
    [property: JsonPropertyName("total_images")] int TotalImages,
    [property: JsonPropertyName("pages")] List<PageMetadata> Pages);

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(DocumentMetadata))]
internal sealed partial class ChandraJsonContext : JsonSerializerContext;
