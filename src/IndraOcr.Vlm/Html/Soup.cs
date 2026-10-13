using System.Text;
using System.Text.RegularExpressions;

namespace IndraOcr.Vlm.Html;

/// <summary>A node of a parsed document: an element or a string.</summary>
public abstract class HtmlNode
{
    /// <summary>The containing element (null at the root or once removed).</summary>
    public HtmlElement? Parent { get; internal set; }

    /// <summary>bs4's <c>next_sibling</c>.</summary>
    public HtmlNode? NextSibling
    {
        get
        {
            if (Parent is null)
            {
                return null;
            }

            int index = Parent.IndexOf(this);
            return index + 1 < Parent.Children.Count ? Parent.Children[index + 1] : null;
        }
    }

    /// <summary>bs4's <c>previous_sibling</c>.</summary>
    public HtmlNode? PreviousSibling
    {
        get
        {
            if (Parent is null)
            {
                return null;
            }

            int index = Parent.IndexOf(this);
            return index > 0 ? Parent.Children[index - 1] : null;
        }
    }

    /// <summary><c>find_parent(name)</c> for any of <paramref name="names"/>.</summary>
    public HtmlElement? FindParent(params string[] names)
    {
        for (HtmlElement? p = Parent; p is not null; p = p.Parent)
        {
            if (Array.IndexOf(names, p.Name) >= 0)
            {
                return p;
            }
        }

        return null;
    }

    /// <summary>Removes the node from its parent (<c>extract</c> / <c>decompose</c>).</summary>
    public void Extract()
    {
        Parent?.Children.Remove(this);
        Parent = null;
    }

    /// <summary>Renders the node as bs4's <c>str()</c> does (the "minimal" formatter).</summary>
    public abstract string Decode();

    /// <inheritdoc/>
    public override string ToString() => Decode();
}

/// <summary>The kinds of bs4 <c>NavigableString</c> the html.parser builder produces.</summary>
public enum TextKind
{
    /// <summary><c>NavigableString</c>.</summary>
    Text,

    /// <summary><c>Comment</c>.</summary>
    Comment,

    /// <summary><c>CData</c>.</summary>
    CData,

    /// <summary><c>Doctype</c>.</summary>
    Doctype,

    /// <summary><c>Declaration</c>.</summary>
    Declaration,

    /// <summary><c>ProcessingInstruction</c>.</summary>
    ProcessingInstruction,

    /// <summary><c>Script</c> (text inside &lt;script&gt;).</summary>
    Script,

    /// <summary><c>Stylesheet</c> (text inside &lt;style&gt;).</summary>
    Stylesheet,

    /// <summary><c>TemplateString</c>.</summary>
    Template,

    /// <summary><c>RubyTextString</c>.</summary>
    RubyText,

    /// <summary><c>RubyParenthesisString</c>.</summary>
    RubyParenthesis,
}

/// <summary>A string node.</summary>
public sealed class HtmlText(string value, TextKind kind = TextKind.Text) : HtmlNode
{
    /// <summary>The string's text.</summary>
    public string Value { get; } = value;

    /// <summary>Which <c>NavigableString</c> subclass this is.</summary>
    public TextKind Kind { get; } = kind;

    /// <inheritdoc/>
    public override string Decode() => Kind switch
    {
        TextKind.Comment => "<!--" + Value + "-->",
        TextKind.CData => "<![CDATA[" + Value + "]]>",
        TextKind.Doctype => "<!DOCTYPE " + Value + ">\n",
        TextKind.Declaration => "<?" + Value + "?>",
        TextKind.ProcessingInstruction => "<?" + Value + ">",
        _ => Parent?.Name is "script" or "style" ? Value : Soup.SubstituteXml(Value),
    };
}

/// <summary>An element (bs4 <c>Tag</c>), or the document itself.</summary>
public sealed class HtmlElement : HtmlNode
{
    internal HtmlElement(string name, List<HtmlAttribute> attributes, bool canBeEmpty)
    {
        Name = name;
        Attributes = attributes;
        CanBeEmpty = canBeEmpty;
    }

    /// <summary>The tag name; <c>"[document]"</c> for the root.</summary>
    public string Name { get; }

    /// <summary>Attributes in source order.</summary>
    public List<HtmlAttribute> Attributes { get; }

    /// <summary>Child nodes.</summary>
    public List<HtmlNode> Children { get; } = [];

    /// <summary>Whether the tag is a void element (rendered <c>&lt;br/&gt;</c> when empty).</summary>
    public bool CanBeEmpty { get; }

    /// <summary>Whether this is the document (rendered without its own tag).</summary>
    public bool Hidden { get; internal set; }

    /// <summary>bs4's <c>is_empty_element</c>.</summary>
    public bool IsEmptyElement => Children.Count == 0 && CanBeEmpty;

    internal int IndexOf(HtmlNode child)
    {
        for (int i = 0; i < Children.Count; i++)
        {
            if (ReferenceEquals(Children[i], child))
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary><c>tag.get(name)</c>: the value, multi-valued attributes joined by spaces.</summary>
    public string? Get(string name)
    {
        foreach (HtmlAttribute attribute in Attributes)
        {
            if (attribute.Name == name)
            {
                return attribute.Value;
            }
        }

        return null;
    }

    /// <summary><c>name in tag.attrs</c>.</summary>
    public bool Has(string name) => Attributes.Any(a => a.Name == name);

    /// <summary><c>tag[name] = value</c>.</summary>
    public void Set(string name, string value)
    {
        for (int i = 0; i < Attributes.Count; i++)
        {
            if (Attributes[i].Name == name)
            {
                Attributes[i] = new HtmlAttribute(name, value);
                return;
            }
        }

        Attributes.Add(new HtmlAttribute(name, value));
    }

    /// <summary><c>del tag[name]</c>.</summary>
    public void RemoveAttribute(string name) => Attributes.RemoveAll(a => a.Name == name);

    /// <summary>Every descendant, depth first.</summary>
    public IEnumerable<HtmlNode> Descendants()
    {
        foreach (HtmlNode child in Children)
        {
            yield return child;
            if (child is HtmlElement element)
            {
                foreach (HtmlNode d in element.Descendants())
                {
                    yield return d;
                }
            }
        }
    }

    /// <summary><c>find_all(names)</c> (recursive) or <c>find_all(names, recursive=False)</c>.</summary>
    public List<HtmlElement> FindAll(string[] names, bool recursive = true) =>
        (recursive ? Descendants() : Children).OfType<HtmlElement>().Where(e => Array.IndexOf(names, e.Name) >= 0).ToList();

    /// <summary><c>find_all(name)</c>.</summary>
    public List<HtmlElement> FindAll(string name, bool recursive = true) => FindAll([name], recursive);

    /// <summary><c>find_all(attrs={name: True})</c>: descendants carrying an attribute.</summary>
    public List<HtmlElement> FindAllWithAttribute(string attribute) =>
        Descendants().OfType<HtmlElement>().Where(e => e.Has(attribute)).ToList();

    /// <summary><c>find(name)</c>.</summary>
    public HtmlElement? Find(string name) => Descendants().OfType<HtmlElement>().FirstOrDefault(e => e.Name == name);

    /// <summary><c>find_previous_sibling()</c>: the nearest preceding element sibling.</summary>
    public HtmlElement? FindPreviousSibling()
    {
        for (HtmlNode? s = PreviousSibling; s is not null; s = s.PreviousSibling)
        {
            if (s is HtmlElement e)
            {
                return e;
            }
        }

        return null;
    }

    /// <summary><c>find_previous_siblings(name)</c>.</summary>
    public List<HtmlElement> FindPreviousSiblings(string name)
    {
        var result = new List<HtmlElement>();
        for (HtmlNode? s = PreviousSibling; s is not null; s = s.PreviousSibling)
        {
            if (s is HtmlElement e && e.Name == name)
            {
                result.Add(e);
            }
        }

        return result;
    }

    /// <summary><c>tag.append(node)</c>; a parsed document is spliced in by its children.</summary>
    public void Append(HtmlNode node)
    {
        if (node is HtmlElement { Hidden: true } document)
        {
            foreach (HtmlNode child in document.Children.ToList())
            {
                Append(child);
            }

            return;
        }

        node.Extract();
        node.Parent = this;
        Children.Add(node);
    }

    /// <summary><c>tag.clear()</c>.</summary>
    public void Clear()
    {
        foreach (HtmlNode child in Children)
        {
            child.Parent = null;
        }

        Children.Clear();
    }

    /// <summary><c>tag.decode_contents()</c>.</summary>
    public string DecodeContents()
    {
        var builder = new StringBuilder();
        foreach (HtmlNode child in Children)
        {
            builder.Append(child.Decode());
        }

        return builder.ToString();
    }

    /// <inheritdoc/>
    public override string Decode()
    {
        if (Hidden)
        {
            return DecodeContents();
        }

        var builder = new StringBuilder();
        builder.Append('<').Append(Name);

        // Formatter.attributes: sorted by name.
        foreach (HtmlAttribute attribute in Attributes.OrderBy(a => a.Name, StringComparer.Ordinal))
        {
            builder.Append(' ');
            if (attribute.Value is null)
            {
                builder.Append(attribute.Name);
            }
            else
            {
                builder.Append(attribute.Name).Append('=').Append(Soup.QuotedAttributeValue(Soup.SubstituteXml(attribute.Value)));
            }
        }

        if (IsEmptyElement)
        {
            builder.Append("/>");
            return builder.ToString();
        }

        builder.Append('>');
        builder.Append(DecodeContents());
        builder.Append("</").Append(Name).Append('>');
        return builder.ToString();
    }
}

/// <summary>An attribute; multi-valued ones (<c>class</c>, ...) hold their tokens joined by single spaces.</summary>
public readonly record struct HtmlAttribute(string Name, string? Value);

/// <summary>
/// <c>BeautifulSoup(markup, "html.parser")</c>: CPython's tokenizer feeding bs4's tree builder.
/// There is no HTML5 tree construction — an end tag closes back to the most recent open element
/// of its name, or is ignored; void elements close themselves; strings of nothing but ASCII
/// whitespace collapse to one newline or one space (outside <c>pre</c> and <c>textarea</c>).
/// </summary>
public sealed class Soup : PyHtmlParser
{
    /// <summary>bs4's <c>HTMLTreeBuilder.empty_element_tags</c>.</summary>
    private static readonly HashSet<string> VoidElements =
    [
        "area", "base", "basefont", "bgsound", "br", "col", "command", "embed", "frame", "hr", "image", "img", "input",
        "isindex", "keygen", "link", "menuitem", "meta", "nextid", "param", "source", "spacer", "track", "wbr",
    ];

    private static readonly HashSet<string> PreserveWhitespaceTags = ["pre", "textarea"];

    private static readonly Dictionary<string, TextKind> StringContainers = new()
    {
        ["rt"] = TextKind.RubyText, ["rp"] = TextKind.RubyParenthesis, ["style"] = TextKind.Stylesheet,
        ["script"] = TextKind.Script, ["template"] = TextKind.Template,
    };

    private static readonly HashSet<string> UniversalListAttributes = ["accesskey", "class", "dropzone"];

    private static readonly Dictionary<string, string[]> TagListAttributes = new()
    {
        ["a"] = ["rel", "rev"], ["link"] = ["rel", "rev"], ["td"] = ["headers"], ["th"] = ["headers"],
        ["form"] = ["accept-charset"], ["object"] = ["archive"], ["area"] = ["rel"], ["icon"] = ["sizes"],
        ["iframe"] = ["sandbox"], ["output"] = ["for"],
    };

    private static readonly Regex NonWhitespace = new(@"\S+", RegexOptions.CultureInvariant);
    private static readonly Regex AmpersandOrBracket = new("([<>&])", RegexOptions.CultureInvariant);

    private readonly HtmlElement _root;
    private readonly List<HtmlElement> _stack = [];
    private readonly Dictionary<string, int> _openCount = new(StringComparer.Ordinal);
    private readonly List<HtmlElement> _preserveWhitespace = [];
    private readonly List<HtmlElement> _stringContainers = [];
    private readonly List<string> _alreadyClosedEmpty = [];
    private readonly StringBuilder _data = new();
    private bool _hasData;

    private Soup()
    {
        _root = new HtmlElement("[document]", [], canBeEmpty: false) { Hidden = true };
        _stack.Add(_root);
    }

    private HtmlElement Current => _stack[^1];

    /// <summary>Parses markup into a document.</summary>
    public static HtmlElement Parse(string markup)
    {
        var soup = new Soup();
        soup.Run(markup);
        soup.EndData();
        while (soup._stack.Count > 1)
        {
            soup.PopTag();
        }

        return soup._root;
    }

    /// <summary><c>EntitySubstitution.substitute_xml</c>: <c>&amp;</c>, <c>&lt;</c>, <c>&gt;</c>.</summary>
    public static string SubstituteXml(string value) =>
        value.IndexOfAny(['&', '<', '>']) < 0
            ? value
            : AmpersandOrBracket.Replace(value, m => m.Value switch { "&" => "&amp;", "<" => "&lt;", _ => "&gt;" });

    /// <summary><c>EntitySubstitution.quoted_attribute_value</c>.</summary>
    public static string QuotedAttributeValue(string value)
    {
        char quote = '"';
        if (value.Contains('"'))
        {
            if (value.Contains('\''))
            {
                value = value.Replace("\"", "&quot;", StringComparison.Ordinal);
            }
            else
            {
                quote = '\'';
            }
        }

        return quote + value + quote;
    }

    private void PushTag(HtmlElement tag)
    {
        tag.Parent = Current;
        Current.Children.Add(tag);
        _stack.Add(tag);
        _openCount[tag.Name] = _openCount.GetValueOrDefault(tag.Name) + 1;
        if (PreserveWhitespaceTags.Contains(tag.Name))
        {
            _preserveWhitespace.Add(tag);
        }

        if (StringContainers.ContainsKey(tag.Name))
        {
            _stringContainers.Add(tag);
        }
    }

    private void PopTag()
    {
        if (_stack.Count <= 1)
        {
            return;
        }

        HtmlElement tag = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        if (_openCount.TryGetValue(tag.Name, out int count))
        {
            _openCount[tag.Name] = count - 1;
        }

        if (_preserveWhitespace.Count > 0 && ReferenceEquals(_preserveWhitespace[^1], tag))
        {
            _preserveWhitespace.RemoveAt(_preserveWhitespace.Count - 1);
        }

        if (_stringContainers.Count > 0 && ReferenceEquals(_stringContainers[^1], tag))
        {
            _stringContainers.RemoveAt(_stringContainers.Count - 1);
        }
    }

    private void PopToTag(string name)
    {
        for (int i = _stack.Count - 1; i > 0; i--)
        {
            if (_openCount.GetValueOrDefault(name) == 0)
            {
                break;
            }

            HtmlElement tag = _stack[i];
            PopTag();
            if (tag.Name == name)
            {
                break;
            }
        }
    }

    private void EndData(TextKind? kind = null)
    {
        if (!_hasData)
        {
            return;
        }

        string data = _data.ToString();
        _data.Clear();
        _hasData = false;
        if (_preserveWhitespace.Count == 0)
        {
            bool strippable = true;
            foreach (char c in data)
            {
                if (c is not (' ' or '\n' or '\t' or '\f' or '\r'))
                {
                    strippable = false;
                    break;
                }
            }

            if (strippable)
            {
                data = data.Contains('\n') ? "\n" : " ";
            }
        }

        TextKind container = kind ?? TextKind.Text;
        if (kind is null && _stringContainers.Count > 0
            && StringContainers.TryGetValue(_stringContainers[^1].Name, out TextKind special))
        {
            container = special;
        }

        var text = new HtmlText(data, container) { Parent = Current };
        Current.Children.Add(text);
    }

    /// <inheritdoc/>
    protected override void HandleStartTag(string tag, List<(string Name, string? Value)> attrs) =>
        StartTag(tag, attrs, handleEmptyElement: true);

    private void StartTag(string tag, List<(string Name, string? Value)> attrs, bool handleEmptyElement)
    {
        // AttributeDict with on_duplicate_attribute=REPLACE: a repeated name keeps its first slot
        // and takes the last value.
        var attributes = new List<HtmlAttribute>();
        foreach ((string name, string? value) in attrs)
        {
            string v = value ?? string.Empty;
            bool isList = UniversalListAttributes.Contains(name)
                || (TagListAttributes.TryGetValue(tag, out string[]? specific) && Array.IndexOf(specific, name) >= 0);
            if (isList)
            {
                v = string.Join(' ', NonWhitespace.Matches(v).Select(m => m.Value));
            }

            int existing = attributes.FindIndex(a => a.Name == name);
            if (existing >= 0)
            {
                attributes[existing] = new HtmlAttribute(name, v);
            }
            else
            {
                attributes.Add(new HtmlAttribute(name, v));
            }
        }

        EndData();
        var element = new HtmlElement(tag, attributes, VoidElements.Contains(tag));
        PushTag(element);
        if (element.IsEmptyElement && handleEmptyElement)
        {
            EndTag(tag, checkAlreadyClosed: false);
            _alreadyClosedEmpty.Add(tag);
        }
    }

    /// <inheritdoc/>
    protected override void HandleStartEndTag(string tag, List<(string Name, string? Value)> attrs)
    {
        StartTag(tag, attrs, handleEmptyElement: false);
        EndTag(tag, checkAlreadyClosed: false);
    }

    /// <inheritdoc/>
    protected override void HandleEndTag(string tag) => EndTag(tag, checkAlreadyClosed: true);

    private void EndTag(string tag, bool checkAlreadyClosed)
    {
        if (checkAlreadyClosed && _alreadyClosedEmpty.Remove(tag))
        {
            return;
        }

        EndData();
        PopToTag(tag);
    }

    /// <inheritdoc/>
    protected override void HandleData(string data)
    {
        _data.Append(data);
        _hasData = true;
    }

    /// <inheritdoc/>
    protected override void HandleCharRef(string name)
    {
        // BeautifulSoupHTMLParser._dereference_numeric_character_reference.
        bool hex = name.StartsWith('x') || name.StartsWith('X');
        string digits = hex ? name[1..] : name;
        int length = 0;
        while (length < digits.Length && (hex ? Uri.IsHexDigit(digits[length]) : char.IsAsciiDigit(digits[length])))
        {
            length++;
        }

        if (length == 0)
        {
            HandleData(digits);
            return;
        }

        long number;
        try
        {
            number = Convert.ToInt64(digits[..length], hex ? 16 : 10);
        }
        catch (OverflowException)
        {
            number = long.MaxValue;
        }

        HandleData(PyString.NumericReference(number, html: false));
        if (length < digits.Length)
        {
            HandleData(digits[length..]);
        }
    }

    /// <inheritdoc/>
    protected override void HandleEntityRef(string name) =>
        HandleData(EntityToCharacter.TryGetValue(name, out string? character) ? character : "&" + name);

    /// <inheritdoc/>
    protected override void HandleComment(string data)
    {
        EndData();
        HandleData(data);
        EndData(TextKind.Comment);
    }

    /// <inheritdoc/>
    protected override void HandleDecl(string decl)
    {
        EndData();
        HandleData(decl.Length >= "DOCTYPE ".Length ? decl["DOCTYPE ".Length..] : string.Empty);
        EndData(TextKind.Doctype);
    }

    /// <inheritdoc/>
    protected override void UnknownDecl(string data)
    {
        TextKind kind = TextKind.Declaration;
        if (data.StartsWith("CDATA[", StringComparison.OrdinalIgnoreCase))
        {
            kind = TextKind.CData;
            data = data["CDATA[".Length..];
        }

        EndData();
        HandleData(data);
        EndData(kind);
    }

    /// <inheritdoc/>
    protected override void HandlePi(string data)
    {
        EndData();
        HandleData(data);
        EndData(TextKind.ProcessingInstruction);
    }

    /// <summary>bs4's <c>HTML_ENTITY_TO_CHARACTER</c>: html5's names without their semicolons, first wins.</summary>
    private static readonly Dictionary<string, string> EntityToCharacter = BuildEntityTable();

    private static Dictionary<string, string> BuildEntityTable()
    {
        var table = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string key in HtmlEntities.Html5.Keys.Order(StringComparer.Ordinal))
        {
            string name = key.EndsWith(';') ? key[..^1] : key;
            table.TryAdd(name, HtmlEntities.Html5[key]);
        }

        return table;
    }
}
