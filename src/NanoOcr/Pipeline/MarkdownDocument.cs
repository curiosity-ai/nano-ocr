namespace NanoOcr.Pipeline;

/// <summary>
/// The document-level markdown passes — the document title, heading levels
/// (<see cref="TitleLevels"/>) and tables split by page breaks (<see cref="TableMerger"/>) — and
/// the re-rendering of each page's <see cref="Page.Markdown"/> after them.
/// </summary>
/// <remarks>
/// <see cref="NanoOCR"/> runs this over each file's pages. A caller driving
/// <see cref="NanoOcrEngine"/> or <see cref="DocumentConverter"/> directly gets each page's
/// markdown as one page on its own would have it, and calls <see cref="Finish"/> with a document's
/// pages to get the rest.
/// </remarks>
public static class MarkdownDocument
{
    /// <summary>
    /// A heading is the document's title when its text is at least this much taller than every
    /// other heading's.
    /// </summary>
    public const float TitleSize = 1.25f;

    /// <summary>
    /// Applies the document-level passes <paramref name="options"/> asks for to a document's pages
    /// and re-renders their markdown (the passes are PaddleOCR-style only; <see cref="MarkdownStyle.Reference"/>
    /// pages are re-rendered as they are). Calling it
    /// again (with other options, say) starts from the pages as recognized.
    /// </summary>
    /// <param name="pages">The document's pages, in order (or a run of them, see <paramref name="first"/>).</param>
    /// <param name="options">Markdown settings.</param>
    /// <param name="keepFurniture">Keep page headers and footers.</param>
    /// <param name="first">Whether <paramref name="pages"/> starts with the document's first page, the one a title can open.</param>
    public static void Finish(IReadOnlyList<Page> pages, MarkdownOptions options, bool keepFurniture = false, bool first = true)
    {
        var elements = pages.Select(p => (IReadOnlyList<PageElement>)(p.Elements ?? [])).ToList();
        foreach (PageElement e in elements.SelectMany(x => x))
        {
            e.DocTitle = false;
            e.TitleLevel = null;
            e.MergedIntoPrevious = false;
            e.Table?.Unmerge();
        }

        bool paddle = options.Style == MarkdownStyle.PaddleOcr;
        if (paddle && options.DocTitle && first && elements.Count > 0)
        {
            MarkTitle(elements);
        }

        if (paddle && options.MergeTables)
        {
            TableMerger.Apply(elements);
        }

        if (paddle && options.TitleLevels)
        {
            TitleLevels.Apply(elements);
        }

        for (int i = 0; i < pages.Count; i++)
        {
            if (pages[i].Elements is { } els)
            {
                pages[i].Markdown = MarkdownWriter.Write(els, options, keepFurniture);
            }
        }
    }

    /// <summary>
    /// The title test: the first element of the first page (furniture aside) is a heading, there is
    /// another heading among <paramref name="pages"/>, and its text is <see cref="TitleSize"/>
    /// taller than each of theirs.
    /// </summary>
    private static void MarkTitle(IReadOnlyList<IReadOnlyList<PageElement>> pages)
    {
        PageElement? opening = pages[0].FirstOrDefault(e => !e.Furniture);
        if (opening is not { Kind: ElementKind.Heading })
        {
            return;
        }

        var others = pages.SelectMany(p => p).Where(e => e.Kind == ElementKind.Heading && !e.Furniture && e != opening).ToList();
        if (others.Count > 0 && others.All(e => opening.LineHeight >= TitleSize * e.LineHeight))
        {
            opening.DocTitle = true;
        }
    }
}
