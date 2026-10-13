using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using IndraOcr.Vlm.Html;
using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Chandra;

/// <summary>A layout block of Chandra's output (<c>chandra.output.LayoutBlock</c>).</summary>
/// <param name="Bbox">Pixel box <c>[x0, y0, x1, y1]</c> in the page image.</param>
/// <param name="Label">The block's <c>data-label</c>, or <c>"block"</c>.</param>
/// <param name="Content">The block's inner HTML, nested <c>data-bbox</c> attributes removed.</param>
public sealed record LayoutBlock(int[] Bbox, string Label, string Content);

/// <summary>
/// <c>chandra/output.py</c>: turning the model's layout HTML into page HTML, markdown, chunks and
/// figure crops. Built on <see cref="Soup"/> and <see cref="ChandraMarkdownConverter"/>, ports of
/// the BeautifulSoup and markdownify behaviour it depends on.
/// </summary>
public static class ChandraOutput
{
    private static readonly Regex AnyTag = new("<.+>", RegexOptions.CultureInvariant);

    /// <summary><c>get_image_name</c>: <c>md5(html)_{index}_img.webp</c>.</summary>
    public static string ImageName(string html, int divIndex) =>
        $"{System.Convert.ToHexStringLower(MD5.HashData(Encoding.UTF8.GetBytes(html)))}_{divIndex}_img.webp";

    /// <summary><c>parse_html</c>.</summary>
    public static string ParseHtml(string html, bool includeHeadersFooters = false, bool includeImages = true)
    {
        HtmlElement soup = Soup.Parse(html);
        var output = new StringBuilder();
        int divIndex = 0;
        foreach (HtmlElement div in soup.FindAll("div", recursive: false))
        {
            divIndex++;
            string? label = div.Get("data-label");
            if (label == "Blank-Page")
            {
                continue;
            }

            if (!string.IsNullOrEmpty(label) && !includeHeadersFooters && label is "Page-Header" or "Page-Footer")
            {
                continue;
            }

            if (!string.IsNullOrEmpty(label) && !includeImages && label is "Image" or "Figure")
            {
                continue;
            }

            if (label is "Image" or "Figure")
            {
                HtmlElement? img = div.Find("img");
                string source = ImageName(html, divIndex);
                if (img is not null)
                {
                    img.Set("src", source);
                }
                else
                {
                    div.Append(Soup.Parse($"<img src='{source}'/>"));
                }
            }

            if (label is not ("Image" or "Figure"))
            {
                foreach (HtmlElement img in div.FindAll("img"))
                {
                    if (string.IsNullOrEmpty(img.Get("src")))
                    {
                        img.Extract();
                    }
                }
            }

            if (label == "Text" && !AnyTag.IsMatch(PyString.Strip(div.DecodeContents())))
            {
                string text = $"<p>{PyString.Strip(div.DecodeContents())}</p>";
                div.Clear();
                div.Append(Soup.Parse(text));
            }

            output.Append(div.DecodeContents());
        }

        return output.ToString();
    }

    /// <summary><c>parse_markdown</c>.</summary>
    public static string ParseMarkdown(string html, bool includeHeadersFooters = false, bool includeImages = true)
    {
        string page = ParseHtml(html, includeHeadersFooters, includeImages);
        string markdown;
        try
        {
            markdown = new ChandraMarkdownConverter().Convert(page);
        }
        catch (Exception)
        {
            markdown = string.Empty;
        }

        return PyString.Strip(markdown);
    }

    /// <summary><c>parse_layout</c>: the top-level divs, boxes scaled from 0-<paramref name="bboxScale"/> to pixels.</summary>
    public static List<LayoutBlock> ParseLayout(string html, int width, int height, int bboxScale = 1000)
    {
        HtmlElement soup = Soup.Parse(html);
        double widthScaler = width / (double)bboxScale;
        double heightScaler = height / (double)bboxScale;
        var blocks = new List<LayoutBlock>();
        foreach (HtmlElement div in soup.FindAll("div", recursive: false))
        {
            string? label = div.Get("data-label");
            if (label == "Blank-Page")
            {
                continue;
            }

            int[] box = ParseBox(div.Get("data-bbox")) ?? [0, 0, 1, 1];
            int[] bbox =
            [
                Math.Max(0, (int)(box[0] * widthScaler)),
                Math.Max(0, (int)(box[1] * heightScaler)),
                Math.Min((int)(box[2] * widthScaler), width),
                Math.Min((int)(box[3] * heightScaler), height),
            ];

            HtmlElement content = Soup.Parse(div.DecodeContents());
            foreach (HtmlElement tag in content.FindAllWithAttribute("data-bbox"))
            {
                tag.RemoveAttribute("data-bbox");
            }

            blocks.Add(new LayoutBlock(bbox, string.IsNullOrEmpty(label) ? "block" : label, content.Decode()));
        }

        return blocks;
    }

    /// <summary><c>bbox.split(" ")</c> mapped through <c>int</c>, exactly four values, or null.</summary>
    internal static int[]? ParseBox(string? text)
    {
        if (text is null)
        {
            return null;
        }

        string[] parts = text.Split(' ');
        if (parts.Length != 4)
        {
            return null;
        }

        var values = new int[4];
        for (int i = 0; i < 4; i++)
        {
            if (!PyInt.TryParse(parts[i], out values[i]))
            {
                return null;
            }
        }

        return values;
    }

    /// <summary>
    /// <c>extract_images</c>: crops of the Image and Figure blocks that hold an <c>img</c>, keyed
    /// by <see cref="ImageName"/>. The index counts chunks, as upstream does.
    /// </summary>
    public static Dictionary<string, RgbImage> ExtractImages(string html, IReadOnlyList<LayoutBlock> chunks, RgbImage image)
    {
        var images = new Dictionary<string, RgbImage>(StringComparer.Ordinal);
        int divIndex = 0;
        foreach (LayoutBlock chunk in chunks)
        {
            divIndex++;
            if (chunk.Label is not ("Image" or "Figure"))
            {
                continue;
            }

            if (Soup.Parse(chunk.Content).Find("img") is null)
            {
                continue;
            }

            int[] b = chunk.Bbox;
            if (b[2] < b[0] || b[3] < b[1])
            {
                // PIL's crop raises ValueError for an inverted box.
                continue;
            }

            images[ImageName(html, divIndex)] = Crop(image, b[0], b[1], b[2], b[3]);
        }

        return images;
    }

    /// <summary>PIL's <c>crop</c>: the box may extend past the image, which is filled with black.</summary>
    internal static RgbImage Crop(RgbImage image, int x0, int y0, int x1, int y1)
    {
        int width = x1 - x0;
        int height = y1 - y0;
        var pixels = new byte[Math.Max(width, 0) * Math.Max(height, 0) * 3];
        for (int y = 0; y < height; y++)
        {
            int sy = y0 + y;
            if (sy < 0 || sy >= image.Height)
            {
                continue;
            }

            for (int x = 0; x < width; x++)
            {
                int sx = x0 + x;
                if (sx < 0 || sx >= image.Width)
                {
                    continue;
                }

                int s = ((sy * image.Width) + sx) * 3;
                int d = ((y * width) + x) * 3;
                pixels[d] = image.Pixels[s];
                pixels[d + 1] = image.Pixels[s + 1];
                pixels[d + 2] = image.Pixels[s + 2];
            }
        }

        return RgbImage.Wrap(pixels, Math.Max(width, 0), Math.Max(height, 0));
    }
}

/// <summary>Python's <c>int(str)</c> for base-10 text.</summary>
internal static class PyInt
{
    /// <summary>
    /// <c>int(s)</c>: surrounding whitespace allowed, an optional sign, digits with single
    /// underscores between them.
    /// </summary>
    public static bool TryParse(string s, out int value)
    {
        value = 0;
        string t = PyString.Strip(s);
        if (t.Length == 0)
        {
            return false;
        }

        bool negative = false;
        int i = 0;
        if (t[0] is '+' or '-')
        {
            negative = t[0] == '-';
            i = 1;
        }

        if (i >= t.Length || !char.IsDigit(t[i]) || !char.IsDigit(t[^1]))
        {
            return false;
        }

        long result = 0;
        for (; i < t.Length; i++)
        {
            char c = t[i];
            if (c == '_' && char.IsDigit(t[i - 1]) && i + 1 < t.Length && char.IsDigit(t[i + 1]))
            {
                continue;
            }

            int digit = (int)char.GetNumericValue(c);
            if (!char.IsDigit(c) || digit < 0)
            {
                return false;
            }

            result = (result * 10) + digit;
            if (result > int.MaxValue)
            {
                return false;
            }
        }

        value = (int)(negative ? -result : result);
        return true;
    }
}

/// <summary>
/// Chandra's <c>Markdownify</c> subclass of markdownify 1.1.0 with the options
/// <c>parse_markdown</c> passes: ATX headings, <c>-</c> bullets, asterisks, underscores and
/// dollars escaped, <c>&lt;sub&gt;</c>/<c>&lt;sup&gt;</c> kept as HTML, math as <c>$…$</c> /
/// <c>$$…$$</c>, and tables kept as HTML.
/// </summary>
public sealed class ChandraMarkdownConverter() : MarkdownConverter(new MarkdownOptions
{
    HeadingStyle = "atx",
    Bullets = "-",
    EscapeMisc = false,
    EscapeUnderscores = true,
    EscapeAsterisks = true,
    SubSymbol = "<sub>",
    SupSymbol = "<sup>",
})
{
    private static readonly Regex TabsAndSpaces = new("[\t ]+", RegexOptions.CultureInvariant);
    private static readonly Regex Brackets = new(@"([\[\]()])", RegexOptions.CultureInvariant);

    /// <inheritdoc/>
    protected override string ConvertTag(HtmlElement el, string text, HashSet<string> parentTags)
    {
        switch (el.Name.ToLowerInvariant())
        {
            case "math":
                bool block = el.Get("display") == "block";
                return block
                    ? "\n$$" + PyString.Strip(text) + "$$\n"
                    : " $" + PyString.Strip(text) + "$ ";
            case "table":
                return "\n\n" + el.Decode() + "\n\n";
            default:
                return base.ConvertTag(el, text, parentTags);
        }
    }

    /// <inheritdoc/>
    protected override string ConvertA(HtmlElement el, string text, HashSet<string> parentTags)
    {
        text = Escape(text, null);
        text = Brackets.Replace(text, @"\$1");
        return base.ConvertA(el, text, parentTags);
    }

    /// <inheritdoc/>
    protected override string Escape(string text, HashSet<string>? parentTags)
    {
        text = base.Escape(text, parentTags);
        return text.Replace("$", @"\$", StringComparison.Ordinal);
    }

    /// <inheritdoc/>
    protected override string ProcessText(HtmlText el, HashSet<string> parentTags)
    {
        string text = el.Value;
        if (el.FindParent("pre") is null)
        {
            text = TabsAndSpaces.Replace(text, " ");
        }

        if (el.FindParent("pre", "code", "kbd", "samp", "math") is null)
        {
            text = Escape(text, null);
        }

        if (el.Parent?.Name == "li" && (el.NextSibling is null || el.NextSibling is HtmlElement { Name: "ul" or "ol" }))
        {
            text = PyString.RStrip(text);
        }

        return text;
    }
}
