using System.Text;
using System.Text.RegularExpressions;

namespace IndraOcr.Vlm.Html;

/// <summary>Options of markdownify 1.1.0's <c>MarkdownConverter</c>.</summary>
public sealed record MarkdownOptions
{
    /// <summary><c>autolinks</c>.</summary>
    public bool Autolinks { get; init; } = true;

    /// <summary><c>bullets</c>.</summary>
    public string Bullets { get; init; } = "*+-";

    /// <summary><c>code_language</c>.</summary>
    public string CodeLanguage { get; init; } = string.Empty;

    /// <summary><c>default_title</c>.</summary>
    public bool DefaultTitle { get; init; }

    /// <summary><c>escape_asterisks</c>.</summary>
    public bool EscapeAsterisks { get; init; } = true;

    /// <summary><c>escape_underscores</c>.</summary>
    public bool EscapeUnderscores { get; init; } = true;

    /// <summary><c>escape_misc</c>.</summary>
    public bool EscapeMisc { get; init; }

    /// <summary><c>heading_style</c>: "underlined", "atx" or "atx_closed".</summary>
    public string HeadingStyle { get; init; } = "underlined";

    /// <summary><c>keep_inline_images_in</c>.</summary>
    public string[] KeepInlineImagesIn { get; init; } = [];

    /// <summary><c>newline_style</c>: "spaces" or "backslash".</summary>
    public string NewlineStyle { get; init; } = "spaces";

    /// <summary><c>strip_document</c>: "strip", "lstrip", "rstrip" or null.</summary>
    public string? StripDocument { get; init; } = "strip";

    /// <summary><c>strong_em_symbol</c>.</summary>
    public string StrongEmSymbol { get; init; } = "*";

    /// <summary><c>sub_symbol</c>.</summary>
    public string SubSymbol { get; init; } = string.Empty;

    /// <summary><c>sup_symbol</c>.</summary>
    public string SupSymbol { get; init; } = string.Empty;

    /// <summary><c>table_infer_header</c>.</summary>
    public bool TableInferHeader { get; init; }

    /// <summary><c>wrap</c> (only <c>false</c> is implemented).</summary>
    public bool Wrap { get; init; }
}

/// <summary>
/// A port of markdownify 1.1.0's <c>MarkdownConverter</c> over the <see cref="Soup"/> DOM.
/// Subclasses override the same hooks the Python subclasses do.
/// </summary>
public class MarkdownConverter(MarkdownOptions options)
{
    private static readonly Regex LineWithContent = new("^(.*)", RegexOptions.Multiline | RegexOptions.CultureInvariant);
    private static readonly Regex Whitespace = new("[\t ]+", RegexOptions.CultureInvariant);
    private static readonly Regex AllWhitespace = new("[\t \r\n]+", RegexOptions.CultureInvariant);
    private static readonly Regex NewlineWhitespace = new("[\t \r\n]*[\r\n][\t \r\n]*", RegexOptions.CultureInvariant);
    private static readonly Regex HtmlHeading = new(@"^h(\d+)", RegexOptions.CultureInvariant);
    private static readonly Regex ExtractNewlines = new(@"^(\n*)((?:.*[^\n])?)(\n*)$", RegexOptions.Singleline | RegexOptions.CultureInvariant);
    private static readonly Regex EscapeMiscChars = new(@"([]\\&<`[>~=+|])", RegexOptions.CultureInvariant);
    private static readonly Regex EscapeMiscDashes = new(@"(\s|^)(-+(?:\s|$))", RegexOptions.CultureInvariant);
    private static readonly Regex EscapeMiscHashes = new(@"(\s|^)(#{1,6}(?:\s|$))", RegexOptions.CultureInvariant);
    private static readonly Regex EscapeMiscListItems = new(@"((?:\s|^)[0-9]{1,9})([.)](?:\s|$))", RegexOptions.CultureInvariant);

    /// <summary>The converter's options.</summary>
    protected MarkdownOptions Options { get; } = options;

    /// <summary><c>convert(html)</c>.</summary>
    public string Convert(string html) => ProcessTag(Soup.Parse(html), []);

    /// <summary>Converts an already-parsed document or element.</summary>
    public string ConvertSoup(HtmlElement soup) => ProcessTag(soup, []);

    private static bool ShouldRemoveWhitespaceInside(HtmlNode? node)
    {
        if (node is not HtmlElement el)
        {
            return false;
        }

        if (HtmlHeading.IsMatch(el.Name))
        {
            return true;
        }

        return el.Name is "p" or "blockquote" or "article" or "div" or "section" or "ol" or "ul" or "li" or "dl" or "dt"
            or "dd" or "table" or "thead" or "tbody" or "tfoot" or "tr" or "td" or "th";
    }

    /// <summary><c>should_remove_whitespace_outside</c>.</summary>
    protected static bool ShouldRemoveWhitespaceOutside(HtmlNode? node) =>
        ShouldRemoveWhitespaceInside(node) || node is HtmlElement { Name: "pre" };

    private static bool IsBlockContent(HtmlNode? node) => node switch
    {
        HtmlElement => true,
        HtmlText { Kind: TextKind.Comment or TextKind.Doctype } => false,
        HtmlText t => PyString.Strip(t.Value).Length != 0,
        _ => false,
    };

    private static HtmlNode? NextBlockContentSibling(HtmlNode? node)
    {
        while (node is not null)
        {
            node = node.NextSibling;
            if (IsBlockContent(node))
            {
                return node;
            }
        }

        return null;
    }

    private string ProcessElement(HtmlNode node, HashSet<string> parentTags) =>
        node is HtmlText text ? ProcessText(text, parentTags) : ProcessTag((HtmlElement)node, parentTags);

    /// <summary><c>process_tag</c>.</summary>
    protected string ProcessTag(HtmlElement node, HashSet<string> parentTags)
    {
        bool removeInside = ShouldRemoveWhitespaceInside(node);

        bool CanIgnore(HtmlNode el)
        {
            switch (el)
            {
                case HtmlElement:
                    return false;
                case HtmlText { Kind: TextKind.Comment or TextKind.Doctype }:
                    return true;
                case HtmlText t:
                    if (PyString.Strip(t.Value).Length != 0)
                    {
                        return false;
                    }

                    if (removeInside && (t.PreviousSibling is null || t.NextSibling is null))
                    {
                        return true;
                    }

                    return ShouldRemoveWhitespaceOutside(t.PreviousSibling) || ShouldRemoveWhitespaceOutside(t.NextSibling);
                default:
                    return true;
            }
        }

        List<HtmlNode> children = node.Children.Where(c => !CanIgnore(c)).ToList();

        var forChildren = new HashSet<string>(parentTags) { node.Name };
        if (HtmlHeading.IsMatch(node.Name) || node.Name is "td" or "th")
        {
            forChildren.Add("_inline");
        }

        if (node.Name is "pre" or "code" or "kbd" or "samp")
        {
            forChildren.Add("_noformat");
        }

        List<string> childStrings = children.Select(c => ProcessElement(c, forChildren)).Where(s => s.Length > 0).ToList();

        if (!(node.Name == "pre" || node.FindParent("pre") is not null))
        {
            var updated = new List<string> { string.Empty };
            foreach (string child in childStrings)
            {
                Match m = ExtractNewlines.Match(child);
                string leading = m.Groups[1].Value;
                string content = m.Groups[2].Value;
                string trailing = m.Groups[3].Value;
                if (updated[^1].Length > 0 && leading.Length > 0)
                {
                    string previousTrailing = updated[^1];
                    updated.RemoveAt(updated.Count - 1);
                    int count = Math.Min(2, Math.Max(previousTrailing.Length, leading.Length));
                    leading = new string('\n', count);
                }

                updated.Add(leading);
                updated.Add(content);
                updated.Add(trailing);
            }

            childStrings = updated;
        }

        string text = string.Concat(childStrings);
        return ConvertTag(node, text, parentTags);
    }

    /// <summary>Dispatches to <c>convert_&lt;tag&gt;</c>; tags without one keep their text.</summary>
    protected virtual string ConvertTag(HtmlElement el, string text, HashSet<string> parentTags)
    {
        string name = el.Name.ToLowerInvariant();
        Match heading = HtmlHeading.Match(name);
        if (heading.Success)
        {
            int n = int.TryParse(heading.Groups[1].Value, out int level) ? level : 6;
            return ConvertHn(n, text, parentTags);
        }

        return name switch
        {
            "[document]" => ConvertDocument(text),
            "a" => ConvertA(el, text, parentTags),
            "b" or "strong" => Inline(Options.StrongEmSymbol + Options.StrongEmSymbol, text, parentTags),
            "em" or "i" => Inline(Options.StrongEmSymbol, text, parentTags),
            "del" or "s" => Inline("~~", text, parentTags),
            "sub" => Inline(Options.SubSymbol, text, parentTags),
            "sup" => Inline(Options.SupSymbol, text, parentTags),
            "blockquote" => ConvertBlockquote(text, parentTags),
            "br" => ConvertBr(parentTags),
            "code" or "kbd" or "samp" => parentTags.Contains("pre") ? text : Inline("`", text, parentTags),
            "div" or "article" or "section" or "dl" => ConvertDiv(text, parentTags),
            "dd" => ConvertDd(text, parentTags),
            "dt" => ConvertDt(text, parentTags),
            "hr" => "\n\n---\n\n",
            "img" => ConvertImg(el, parentTags),
            "ul" or "ol" => ConvertList(el, text, parentTags),
            "li" => ConvertLi(el, text),
            "p" => ConvertP(text, parentTags),
            "pre" => text.Length == 0 ? string.Empty : $"\n\n```{Options.CodeLanguage}\n{text}\n```\n\n",
            "script" or "style" => string.Empty,
            "table" => ConvertTable(el, text, parentTags),
            "caption" => PyString.Strip(text) + "\n\n",
            "figcaption" => "\n\n" + PyString.Strip(text) + "\n\n",
            "td" or "th" => ConvertCell(el, text),
            "tr" => ConvertTr(el, text),
            "video" => ConvertVideo(el, text, parentTags),
            _ => text,
        };
    }

    /// <summary><c>abstract_inline_conversion</c>.</summary>
    protected static string Inline(string markupPrefix, string text, HashSet<string> parentTags)
    {
        string markupSuffix = markupPrefix.StartsWith('<') && markupPrefix.EndsWith('>')
            ? "</" + markupPrefix[1..]
            : markupPrefix;
        if (parentTags.Contains("_noformat"))
        {
            return text;
        }

        (string prefix, string suffix, string body) = Chomp(text);
        return body.Length == 0 ? string.Empty : prefix + markupPrefix + body + markupSuffix + suffix;
    }

    /// <summary><c>chomp</c>.</summary>
    protected static (string Prefix, string Suffix, string Text) Chomp(string text)
    {
        string prefix = text.Length > 0 && text[0] == ' ' ? " " : string.Empty;
        string suffix = text.Length > 0 && text[^1] == ' ' ? " " : string.Empty;
        return (prefix, suffix, PyString.Strip(text));
    }

    private string ConvertDocument(string text) => Options.StripDocument switch
    {
        "lstrip" => text.TrimStart('\n'),
        "rstrip" => text.TrimEnd('\n'),
        "strip" => text.Trim('\n'),
        _ => text,
    };

    /// <summary><c>process_text</c>.</summary>
    protected virtual string ProcessText(HtmlText el, HashSet<string> parentTags)
    {
        string text = el.Value;
        if (!parentTags.Contains("pre"))
        {
            text = NewlineWhitespace.Replace(text, "\n");
            text = Whitespace.Replace(text, " ");
        }

        if (!parentTags.Contains("_noformat"))
        {
            text = Escape(text, parentTags);
        }

        if (ShouldRemoveWhitespaceOutside(el.PreviousSibling) || (ShouldRemoveWhitespaceInside(el.Parent) && el.PreviousSibling is null))
        {
            text = text.TrimStart(' ', '\t', '\r', '\n');
        }

        if (ShouldRemoveWhitespaceOutside(el.NextSibling) || (ShouldRemoveWhitespaceInside(el.Parent) && el.NextSibling is null))
        {
            text = PyString.RStrip(text);
        }

        return text;
    }

    /// <summary><c>escape</c>.</summary>
    protected virtual string Escape(string text, HashSet<string>? parentTags)
    {
        if (text.Length == 0)
        {
            return string.Empty;
        }

        if (Options.EscapeMisc)
        {
            text = EscapeMiscChars.Replace(text, @"\$1");
            text = EscapeMiscDashes.Replace(text, @"$1\$2");
            text = EscapeMiscHashes.Replace(text, @"$1\$2");
            text = EscapeMiscListItems.Replace(text, @"$1\$2");
        }

        if (Options.EscapeAsterisks)
        {
            text = text.Replace("*", @"\*", StringComparison.Ordinal);
        }

        if (Options.EscapeUnderscores)
        {
            text = text.Replace("_", @"\_", StringComparison.Ordinal);
        }

        return text;
    }

    /// <summary><c>convert_a</c>.</summary>
    protected virtual string ConvertA(HtmlElement el, string text, HashSet<string> parentTags)
    {
        if (parentTags.Contains("_noformat"))
        {
            return text;
        }

        (string prefix, string suffix, string body) = Chomp(text);
        if (body.Length == 0)
        {
            return string.Empty;
        }

        string? href = el.Get("href");
        string? title = el.Get("title");
        if (Options.Autolinks && body.Replace(@"\_", "_", StringComparison.Ordinal) == href && string.IsNullOrEmpty(title) && !Options.DefaultTitle)
        {
            return $"<{href}>";
        }

        if (Options.DefaultTitle && string.IsNullOrEmpty(title))
        {
            title = href;
        }

        string titlePart = !string.IsNullOrEmpty(title) ? $" \"{title.Replace("\"", "\\\"", StringComparison.Ordinal)}\"" : string.Empty;
        return !string.IsNullOrEmpty(href) ? $"{prefix}[{body}]({href}{titlePart}){suffix}" : body;
    }

    private static string ConvertBlockquote(string text, HashSet<string> parentTags)
    {
        text = text.Trim(' ', '\t', '\r', '\n');
        if (parentTags.Contains("_inline"))
        {
            return " " + text + " ";
        }

        if (text.Length == 0)
        {
            return "\n";
        }

        text = LineWithContent.Replace(text, m => m.Groups[1].Value.Length > 0 ? "> " + m.Groups[1].Value : ">");
        return "\n" + text + "\n\n";
    }

    private string ConvertBr(HashSet<string> parentTags)
    {
        if (parentTags.Contains("_inline"))
        {
            return " ";
        }

        return Options.NewlineStyle.Equals("backslash", StringComparison.OrdinalIgnoreCase) ? "\\\n" : "  \n";
    }

    private static string ConvertDiv(string text, HashSet<string> parentTags)
    {
        if (parentTags.Contains("_inline"))
        {
            return " " + PyString.Strip(text) + " ";
        }

        text = PyString.Strip(text);
        return text.Length > 0 ? $"\n\n{text}\n\n" : string.Empty;
    }

    private static string ConvertDd(string text, HashSet<string> parentTags)
    {
        text = PyString.Strip(text);
        if (parentTags.Contains("_inline"))
        {
            return " " + text + " ";
        }

        if (text.Length == 0)
        {
            return "\n";
        }

        text = LineWithContent.Replace(text, m => m.Groups[1].Value.Length > 0 ? "    " + m.Groups[1].Value : string.Empty);
        text = ":" + text[1..];
        return text + "\n";
    }

    private static string ConvertDt(string text, HashSet<string> parentTags)
    {
        text = PyString.Strip(text);
        text = AllWhitespace.Replace(text, " ");
        if (parentTags.Contains("_inline"))
        {
            return " " + text + " ";
        }

        return text.Length == 0 ? "\n" : $"\n\n{text}\n";
    }

    private string ConvertHn(int n, string text, HashSet<string> parentTags)
    {
        if (parentTags.Contains("_inline"))
        {
            return text;
        }

        n = Math.Max(1, Math.Min(6, n));
        string style = Options.HeadingStyle.ToLowerInvariant();
        text = PyString.Strip(text);
        if (style == "underlined" && n <= 2)
        {
            string line = n == 1 ? "=" : "-";
            text = PyString.RStrip(text);
            return text.Length > 0 ? $"\n\n{text}\n{string.Concat(Enumerable.Repeat(line, text.Length))}\n\n" : string.Empty;
        }

        text = AllWhitespace.Replace(text, " ");
        string hashes = new('#', n);
        return style == "atx_closed" ? $"\n\n{hashes} {text} {hashes}\n\n" : $"\n\n{hashes} {text}\n\n";
    }

    private string ConvertImg(HtmlElement el, HashSet<string> parentTags)
    {
        string alt = el.Get("alt") ?? string.Empty;
        string src = el.Get("src") ?? string.Empty;
        string title = el.Get("title") ?? string.Empty;
        string titlePart = title.Length > 0 ? $" \"{title.Replace("\"", "\\\"", StringComparison.Ordinal)}\"" : string.Empty;
        if (parentTags.Contains("_inline") && Array.IndexOf(Options.KeepInlineImagesIn, el.Parent?.Name) < 0)
        {
            return alt;
        }

        return $"![{alt}]({src}{titlePart})";
    }

    private string ConvertVideo(HtmlElement el, string text, HashSet<string> parentTags)
    {
        if (parentTags.Contains("_inline") && Array.IndexOf(Options.KeepInlineImagesIn, el.Parent?.Name) < 0)
        {
            return text;
        }

        string src = el.Get("src") ?? string.Empty;
        if (src.Length == 0)
        {
            HtmlElement? source = el.FindAll("source").FirstOrDefault(s => s.Has("src"));
            src = source?.Get("src") ?? string.Empty;
        }

        string poster = el.Get("poster") ?? string.Empty;
        if (src.Length > 0 && poster.Length > 0)
        {
            return $"[![{text}]({poster})]({src})";
        }

        if (src.Length > 0)
        {
            return $"[{text}]({src})";
        }

        return poster.Length > 0 ? $"![{text}]({poster})" : text;
    }

    private static string ConvertList(HtmlElement el, string text, HashSet<string> parentTags)
    {
        bool beforeParagraph = false;
        HtmlNode? next = NextBlockContentSibling(el);
        if (next is not null && !(next is HtmlElement { Name: "ul" or "ol" }))
        {
            beforeParagraph = true;
        }

        if (parentTags.Contains("li"))
        {
            return "\n" + PyString.RStrip(text);
        }

        return "\n\n" + text + (beforeParagraph ? "\n" : string.Empty);
    }

    private string ConvertLi(HtmlElement el, string text)
    {
        text = PyString.Strip(text);
        if (text.Length == 0)
        {
            return "\n";
        }

        string bullet;
        HtmlElement? parent = el.Parent;
        if (parent is { Name: "ol" })
        {
            string? startText = parent.Get("start");
            int start = !string.IsNullOrEmpty(startText) && IsNumeric(startText) ? int.Parse(startText) : 1;
            bullet = $"{start + el.FindPreviousSiblings("li").Count}.";
        }
        else
        {
            int depth = -1;
            for (HtmlElement? e = el; e is not null; e = e.Parent)
            {
                if (e.Name == "ul")
                {
                    depth++;
                }
            }

            string bullets = Options.Bullets;
            bullet = bullets[((depth % bullets.Length) + bullets.Length) % bullets.Length].ToString();
        }

        bullet += " ";
        int width = bullet.Length;
        string indent = new(' ', width);
        text = LineWithContent.Replace(text, m => m.Groups[1].Value.Length > 0 ? indent + m.Groups[1].Value : string.Empty);
        text = bullet + text[width..];
        return text + "\n";
    }

    private static bool IsNumeric(string s) => s.Length > 0 && s.All(char.IsDigit);

    private string ConvertP(string text, HashSet<string> parentTags)
    {
        if (parentTags.Contains("_inline"))
        {
            return " " + text.Trim(' ', '\t', '\r', '\n') + " ";
        }

        text = text.Trim(' ', '\t', '\r', '\n');
        return text.Length > 0 ? $"\n\n{text}\n\n" : string.Empty;
    }

    /// <summary><c>convert_table</c>.</summary>
    protected virtual string ConvertTable(HtmlElement el, string text, HashSet<string> parentTags) =>
        "\n\n" + PyString.Strip(text) + "\n\n";

    private static string ConvertCell(HtmlElement el, string text)
    {
        int colspan = 1;
        string? value = el.Get("colspan");
        if (value is not null && PyString.IsDigit(value))
        {
            colspan = int.Parse(value);
        }

        return " " + PyString.Strip(text).Replace("\n", " ", StringComparison.Ordinal)
            + string.Concat(Enumerable.Repeat(" |", colspan));
    }

    private string ConvertTr(HtmlElement el, string text)
    {
        List<HtmlElement> cells = el.FindAll(["td", "th"]);
        bool isFirstRow = el.FindPreviousSibling() is null;
        HtmlElement? parent = el.Parent;
        bool isHeadRow = cells.All(c => c.Name == "th")
            || (parent?.Name == "thead" && parent.FindAll("tr").Count == 1);
        bool isHeadRowMissing = (isFirstRow && parent?.Name != "tbody")
            || (isFirstRow && parent?.Name == "tbody" && (parent.Parent?.FindAll(["thead"]).Count ?? 0) < 1);
        int fullColspan = 0;
        foreach (HtmlElement cell in cells)
        {
            string? value = cell.Get("colspan");
            fullColspan += value is not null && PyString.IsDigit(value) ? int.Parse(value) : 1;
        }

        string overline = string.Empty;
        string underline = string.Empty;
        if ((isHeadRow || (isHeadRowMissing && Options.TableInferHeader)) && isFirstRow)
        {
            underline += "| " + string.Join(" | ", Enumerable.Repeat("---", fullColspan)) + " |\n";
        }
        else if ((isHeadRowMissing && !Options.TableInferHeader)
            || (isFirstRow && (parent?.Name == "table" || (parent?.Name == "tbody" && parent.FindPreviousSibling() is null))))
        {
            overline += "| " + string.Join(" | ", Enumerable.Repeat(string.Empty, fullColspan)) + " |\n";
            overline += "| " + string.Join(" | ", Enumerable.Repeat("---", fullColspan)) + " |\n";
        }

        return overline + "|" + text + "\n" + underline;
    }
}
