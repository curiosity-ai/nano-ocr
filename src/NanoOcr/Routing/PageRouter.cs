using System.Runtime.CompilerServices;
using NanoOcr.Imaging;
using SkiaSharp;

namespace NanoOcr.Routing;

/// <summary>A page on its way to a recognizer: where it came from, NanoOCR's reading, and its category.</summary>
/// <param name="Ocr">
/// NanoOCR's result for the page — the answer itself for a page NanoOCR reads well, and the
/// page image (<see cref="OcrResult.Image"/>) when <see cref="NanoOcrOptions.KeepImages"/> is on.
/// </param>
/// <param name="Classification">The page's category and the signals behind it.</param>
/// <param name="Encoded">
/// The encoded input (image file or PDF bytes) when the router was given bytes; otherwise
/// <see langword="null"/>, and <see cref="InputPath"/> names the file.
/// </param>
public sealed record RoutedPage(OcrResult Ocr, PageClassification Classification, byte[]? Encoded = null)
{
    /// <summary>The input file (empty for in-memory input).</summary>
    public string InputPath => Ocr.InputPath;

    /// <summary>Zero-based page (or frame) index in the input.</summary>
    public int PageIndex => Ocr.PageIndex;

    /// <summary>The page's category.</summary>
    public PageCategory Category => Classification.Category;
}

/// <summary>A routed page and what its recognizer returned.</summary>
/// <typeparam name="TOutput">The recognizers' common output type.</typeparam>
/// <param name="Page">The page, its reading and its category.</param>
/// <param name="Route">The route taken: the index of the matching <c>When</c>, in the order they were added, or -1 for the fallback.</param>
/// <param name="Output">The recognizer's output.</param>
public sealed record RoutedResult<TOutput>(RoutedPage Page, int Route, TOutput Output)
{
    /// <summary>Whether no route matched and the fallback ran.</summary>
    public bool UsedFallback => Route < 0;
}

/// <summary>
/// Reads each page with NanoOCR, classifies it (<see cref="PageClassifier"/> by default), and
/// hands it to the recognizer registered for its category — NanoOCR's own reading for printed
/// text, say, and a heavier model for handwriting and tables — falling back to a default for
/// anything else.
/// </summary>
/// <remarks>
/// <para>
/// Generic on the output, so recognizers from any library can share a router through a
/// conversion of their results:
/// </para>
/// <code>
/// using var nano = new NanoOCR(new NanoOcrOptions { KeepImages = true });
/// var router = new PageRouter&lt;string&gt;(nano)
///     .When(PageCategory.PrintedText, page => page.Ocr.Markdown)
///     .When(PageCategory.TableOrComplex, page => paddle.Parse(Load(page)).ToMarkdown())
///     .When(PageCategory.Handwritten, page => paddle.Parse(Load(page)).ToMarkdown())
///     .Otherwise(page => paddle.Parse(Load(page)).ToMarkdown());
/// foreach (RoutedResult&lt;string&gt; r in router.Route("scan.pdf")) { … }
/// </code>
/// <para>
/// Routes are tried in the order they were added and the first match runs, so a predicate route
/// (<see cref="When(Func{PageClassification, bool}, Func{RoutedPage, TOutput})"/>) can refine a
/// category added after it. Without <see cref="Otherwise(Func{RoutedPage, TOutput})"/>, a page no
/// route takes throws. The router adds no threads of its own: pages are read as
/// <see cref="NanoOCR.Predict(string, int, string?)"/> yields them and recognized one at a time,
/// and an instance is safe to use from several threads once configured.
/// </para>
/// </remarks>
/// <typeparam name="TOutput">The recognizers' common output type.</typeparam>
public sealed class PageRouter<TOutput>
{
    private readonly NanoOCR _ocr;
    private readonly Func<OcrResult, PageClassification> _classify;
    private readonly List<(Func<PageClassification, bool> Match, Func<RoutedPage, CancellationToken, ValueTask<TOutput>> Run)> _routes = [];
    private Func<RoutedPage, CancellationToken, ValueTask<TOutput>>? _fallback;

    /// <summary>A router classifying with <see cref="PageClassifier"/> and the given thresholds.</summary>
    /// <param name="ocr">Reads each page (its options decide the markdown, the threads, whether images are kept).</param>
    /// <param name="options">Classifier thresholds; the calibrated defaults when omitted.</param>
    public PageRouter(NanoOCR ocr, PageClassifierOptions? options = null)
        : this(ocr, r => PageClassifier.Classify(r, options))
    {
    }

    /// <summary>A router with a classifier of your own (a trained model, say).</summary>
    /// <param name="ocr">Reads each page.</param>
    /// <param name="classify">Classifies a page from its reading.</param>
    public PageRouter(NanoOCR ocr, Func<OcrResult, PageClassification> classify)
    {
        _ocr = ocr;
        _classify = classify;
    }

    /// <summary>Sends pages of <paramref name="category"/> to <paramref name="recognizer"/>.</summary>
    public PageRouter<TOutput> When(PageCategory category, Func<RoutedPage, TOutput> recognizer) =>
        When(c => c.Category == category, recognizer);

    /// <summary>Sends pages of <paramref name="category"/> to an asynchronous <paramref name="recognizer"/>.</summary>
    public PageRouter<TOutput> When(PageCategory category, Func<RoutedPage, CancellationToken, ValueTask<TOutput>> recognizer) =>
        When(c => c.Category == category, recognizer);

    /// <summary>Sends pages whose classification matches <paramref name="match"/> to <paramref name="recognizer"/>.</summary>
    public PageRouter<TOutput> When(Func<PageClassification, bool> match, Func<RoutedPage, TOutput> recognizer) =>
        When(match, Sync(recognizer));

    /// <summary>Sends pages whose classification matches <paramref name="match"/> to an asynchronous <paramref name="recognizer"/>.</summary>
    public PageRouter<TOutput> When(Func<PageClassification, bool> match, Func<RoutedPage, CancellationToken, ValueTask<TOutput>> recognizer)
    {
        ArgumentNullException.ThrowIfNull(match);
        ArgumentNullException.ThrowIfNull(recognizer);
        _routes.Add((match, recognizer));
        return this;
    }

    /// <summary>The recognizer for pages no route takes.</summary>
    public PageRouter<TOutput> Otherwise(Func<RoutedPage, TOutput> fallback) => Otherwise(Sync(fallback));

    /// <summary>The asynchronous recognizer for pages no route takes.</summary>
    public PageRouter<TOutput> Otherwise(Func<RoutedPage, CancellationToken, ValueTask<TOutput>> fallback)
    {
        ArgumentNullException.ThrowIfNull(fallback);
        _fallback = fallback;
        return this;
    }

    /// <summary>Classifies a page without recognizing it.</summary>
    public PageClassification Classify(OcrResult result) => _classify(result);

    /// <summary>Routes every page of a file (PDF or image), or of every file in a directory.</summary>
    public IEnumerable<RoutedResult<TOutput>> Route(string input, int maxPages = 0, string? password = null)
    {
        foreach (OcrResult r in _ocr.Predict(input, maxPages, password))
        {
            yield return Wait(Run(r, null, CancellationToken.None));
        }
    }

    /// <summary>Routes every page of encoded image bytes or a PDF.</summary>
    public IEnumerable<RoutedResult<TOutput>> Route(byte[] encoded, string? password = null)
    {
        foreach (OcrResult r in _ocr.Predict(encoded, password))
        {
            yield return Wait(Run(r, encoded, CancellationToken.None));
        }
    }

    /// <summary>Routes an in-memory grayscale page.</summary>
    public RoutedResult<TOutput> Route(GrayImage image) => Wait(Run(_ocr.Predict(image), null, CancellationToken.None));

    /// <summary>Routes a SkiaSharp bitmap.</summary>
    public RoutedResult<TOutput> Route(SKBitmap bitmap) => Wait(Run(_ocr.Predict(bitmap), null, CancellationToken.None));

    /// <summary>
    /// Routes every page of a file or directory, awaiting asynchronous recognizers. NanoOCR's own
    /// reading runs on the calling thread between them.
    /// </summary>
    public async IAsyncEnumerable<RoutedResult<TOutput>> RouteAsync(
        string input,
        int maxPages = 0,
        string? password = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (OcrResult r in _ocr.Predict(input, maxPages, password))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return await Run(r, null, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Routes every page of encoded image bytes or a PDF, awaiting asynchronous recognizers.</summary>
    public async IAsyncEnumerable<RoutedResult<TOutput>> RouteAsync(
        byte[] encoded,
        string? password = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (OcrResult r in _ocr.Predict(encoded, password))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return await Run(r, encoded, cancellationToken).ConfigureAwait(false);
        }
    }

    private async ValueTask<RoutedResult<TOutput>> Run(OcrResult result, byte[]? encoded, CancellationToken ct)
    {
        var page = new RoutedPage(result, _classify(result), encoded);
        for (int i = 0; i < _routes.Count; i++)
        {
            if (_routes[i].Match(page.Classification))
            {
                return new(page, i, await _routes[i].Run(page, ct).ConfigureAwait(false));
            }
        }

        if (_fallback is null)
        {
            throw new InvalidOperationException(
                $"No route for {page.InputPath} page {page.PageIndex + 1} ({page.Category}: {page.Classification.Reason}) and no fallback; add Otherwise(…).");
        }

        return new(page, -1, await _fallback(page, ct).ConfigureAwait(false));
    }

    private static Func<RoutedPage, CancellationToken, ValueTask<TOutput>> Sync(Func<RoutedPage, TOutput> f)
    {
        ArgumentNullException.ThrowIfNull(f);
        return (p, _) => ValueTask.FromResult(f(p));
    }

    private static RoutedResult<TOutput> Wait(ValueTask<RoutedResult<TOutput>> task) =>
        task.IsCompletedSuccessfully ? task.Result : task.AsTask().GetAwaiter().GetResult();
}
