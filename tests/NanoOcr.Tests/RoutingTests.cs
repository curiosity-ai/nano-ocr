using NanoOcr.Pipeline;
using NanoOcr.Routing;
using Xunit;

namespace NanoOcr.Tests;

/// <summary><see cref="PageClassifier"/> on synthetic pages, <see cref="PageRouter{TOutput}"/> on the test documents.</summary>
public class RoutingTests
{
    private static readonly Lazy<NanoOCR> Ocr = new(() => new NanoOCR(new NanoOcrOptions { KeepImages = true }));

    private static string Doc(string name) => Path.Combine(TestData.Documents, name);

    /// <summary>A 1000×1300 page of <paramref name="n"/> lines of the given height, width and confidence.</summary>
    private static Page Synthetic(int n, double conf, double height, double width = 200, string text = "word word", int tableLines = 0)
    {
        var page = new Page { Width = 1000, Height = 1300 };
        for (int i = 0; i < n; i++)
        {
            page.Lines.Add(new Line([100, 50 + (i * 40), 100 + width, 50 + (i * 40) + height], text, conf));
        }

        page.Blocks.Add(new Block { Label = "Text", Text = string.Join("\n", Enumerable.Repeat(text, n - tableLines)) });
        if (tableLines > 0)
        {
            page.Blocks.Add(new Block { Label = "Table", Text = string.Join("\n", Enumerable.Repeat(text, tableLines)) });
        }

        return page;
    }

    [Fact]
    public void PagesAreClassifiedByTheirReading()
    {
        Assert.Equal(PageCategory.PrintedText, PageClassifier.Classify(Synthetic(20, 0.97, 12)).Category);
        Assert.Equal(PageCategory.TableOrComplex, PageClassifier.Classify(Synthetic(20, 0.97, 12, tableLines: 8)).Category);

        // Unread, in large strokes: handwriting. Unread in small type: a fax or noise.
        PageClassification hand = PageClassifier.Classify(Synthetic(20, 0.3, 30, text: ""));
        Assert.Equal(PageCategory.Handwritten, hand.Category);
        Assert.Equal(1.0, hand.Signals.LowConfidenceShare);
        Assert.Equal(PageCategory.Handwritten, PageClassifier.Classify(Synthetic(20, 0.3, 8, width: 400, text: "")).Category);
        Assert.Equal(PageCategory.Mixed, PageClassifier.Classify(Synthetic(20, 0.3, 8, text: "")).Category);

        // Read, but not confidently; too little text; a failed page.
        Assert.Equal(PageCategory.Mixed, PageClassifier.Classify(Synthetic(20, 0.8, 12)).Category);
        Assert.Equal(PageCategory.Mixed, PageClassifier.Classify(Synthetic(2, 0.99, 12)).Category);
        Assert.Equal(PageCategory.Mixed, PageClassifier.Classify(new Page { Source = PageSources.Error, Note = "x" }).Category);

        // A text layer is never handwriting, and the thresholds are options.
        Page digital = Synthetic(20, 0.3, 30, text: "");
        digital.Source = PageSources.Text;
        Assert.NotEqual(PageCategory.Handwritten, PageClassifier.Classify(digital).Category);
        Assert.Equal(
            PageCategory.PrintedText,
            PageClassifier.Classify(Synthetic(20, 0.8, 12), PageClassifierOptions.Default with { PrintedConfidence = 0.7 }).Category);
    }

    [Fact]
    public void PagesGoToTheRecognizerOfTheirCategory()
    {
        var router = new PageRouter<string>(Ocr.Value)
            .When(PageCategory.PrintedText, p => "nano:" + p.Ocr.Markdown)
            .When(PageCategory.TableOrComplex, p => $"table:{p.PageIndex}:{p.Ocr.Image!.Width}")
            .Otherwise(p => "fallback:" + p.Category);

        RoutedResult<string> math = router.Route(Doc("page_math.png")).Single();
        Assert.Equal(PageCategory.PrintedText, math.Page.Category);
        Assert.Equal(0, math.Route);
        Assert.StartsWith("nano:For every x", math.Output);

        RoutedResult<string> paper = router.Route(Doc("page_two_column.png")).Single();
        Assert.Equal(PageCategory.TableOrComplex, paper.Page.Category);
        Assert.Equal(1, paper.Page.Classification.Signals.Tables);
        Assert.Equal("table:0:1700", paper.Output);

        // A blank third frame: no text, so Mixed, so the fallback.
        RoutedResult<string> blank = router.Route(Doc(Path.Combine("tiff", "multipage.tif"))).Last();
        Assert.True(blank.UsedFallback);
        Assert.Equal("fallback:Mixed", blank.Output);
    }

    [Fact]
    public void RoutesAreTriedInOrderAndAnUnroutedPageThrows()
    {
        var router = new PageRouter<int>(Ocr.Value)
            .When(c => c.Signals.Tables > 0, _ => 1)
            .When(PageCategory.TableOrComplex, _ => 2);
        Assert.Equal(1, router.Route(Doc("page_two_column.png")).Single().Output);
        Assert.Throws<InvalidOperationException>(() => router.Route(Doc("page_math.png")).ToList());

        // A classifier of one's own.
        var mine = new PageRouter<string>(Ocr.Value, r => new PageClassification(PageCategory.Handwritten, PageClassifier.Signals(r.Page), "test"))
            .When(PageCategory.Handwritten, p => p.Classification.Reason);
        Assert.Equal("test", mine.Route(Doc("page_math.png")).Single().Output);
    }

    [Fact]
    public async Task AsynchronousRecognizersAreAwaited()
    {
        byte[] bytes = File.ReadAllBytes(Doc("page_math.png"));
        var router = new PageRouter<string>(Ocr.Value)
            .When(PageCategory.PrintedText, async (p, ct) =>
            {
                await Task.Yield();
                return $"{p.Encoded!.Length}";
            })
            .Otherwise(_ => "fallback");
        var results = new List<RoutedResult<string>>();
        await foreach (RoutedResult<string> r in router.RouteAsync(bytes))
        {
            results.Add(r);
        }

        Assert.Equal(bytes.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), Assert.Single(results).Output);
        Assert.Equal(results[0].Output, router.Route(bytes).Single().Output);
    }
}
