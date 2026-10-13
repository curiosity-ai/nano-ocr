using System.Text.RegularExpressions;

namespace IndraOcr.Vlm.Html;

/// <summary>
/// A port of CPython 3.13's <c>html.parser.HTMLParser</c> with <c>convert_charrefs=False</c>,
/// which is how BeautifulSoup's <c>"html.parser"</c> builder runs it. The regular expressions are
/// the module's own; what the tokenizer makes of malformed markup (an unterminated tag at the
/// end, <c>&lt;b&lt;/math&gt;</c>, a bare ampersand) is therefore what Python makes of it, which is
/// what decides the DOM both pipelines' output is built from.
/// </summary>
public abstract class PyHtmlParser
{
    private const RegexOptions Options = RegexOptions.CultureInvariant;
    private const RegexOptions Verbose = RegexOptions.CultureInvariant | RegexOptions.IgnorePatternWhitespace;

    private static readonly Regex InterestingNormal = new("[&<]", Options);
    private static readonly Regex Incomplete = new(@"\G&[a-zA-Z#]", Options);
    private static readonly Regex EntityRef = new(@"\G&([a-zA-Z][-.a-zA-Z0-9]*)[^a-zA-Z0-9]", Options);
    private static readonly Regex CharRef = new(@"\G&#(?:[0-9]+|[xX][0-9a-fA-F]+)[^0-9a-fA-F]", Options);
    private static readonly Regex IncompleteCharRef = new(@"\G&#(?:[0-9]|[xX][0-9a-fA-F])", Options);
    private static readonly Regex AttrCharRef = new(@"&(#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*)[;=]?", Options);
    private static readonly Regex StartTagOpen = new(@"\G<[a-zA-Z]", Options);
    private static readonly Regex EndTagOpen = new(@"\G</[a-zA-Z]", Options);
    private static readonly Regex PiClose = new(">", Options);
    private static readonly Regex CommentClose = new(@"--!?>", Options);
    private static readonly Regex CommentAbruptClose = new(@"\G-?>", Options);
    private static readonly Regex TagFindTolerant = new(@"\G([a-zA-Z][^\t\n\r\f />]*)(?:[\t\n\r\f ]|/(?!>))*", Options);

    private static readonly Regex AttrFindTolerant = new(@"\G
  (
    (?<=['""\t\n\r\f /])[^\t\n\r\f />][^\t\n\r\f /=>]*  # attribute name
   )
  ([\t\n\r\f ]*=[\t\n\r\f ]*        # value indicator
    ('[^']*'                        # LITA-enclosed value
    |""[^""]*""                     # LIT-enclosed value
    |(?!['""])[^>\t\n\r\f ]*        # bare value
    )
   )?
  (?:[\t\n\r\f ]|/(?!>))*           # possibly followed by a space
", Verbose);

    private static readonly Regex LocateTagEnd = new(@"\G
  [a-zA-Z][^\t\n\r\f />]*           # tag name
  [\t\n\r\f /]*                     # optional whitespace before attribute name
  (?:(?<=['""\t\n\r\f /])[^\t\n\r\f />][^\t\n\r\f /=>]*  # attribute name
    (?:[\t\n\r\f ]*=[\t\n\r\f ]*    # value indicator
      (?:'[^']*'                    # LITA-enclosed value
        |""[^""]*""                 # LIT-enclosed value
        |(?!['""])[^>\t\n\r\f ]*    # bare value
       )
     )?
    [\t\n\r\f /]*                   # possibly followed by a space
   )*
   >?
", Verbose);

    private static readonly string[] CdataContentElements = ["script", "style", "xmp", "iframe", "noembed", "noframes"];
    private static readonly string[] RcdataContentElements = ["textarea", "title"];

    private string _rawdata = string.Empty;
    private Regex _interesting = InterestingNormal;
    private string? _cdataElement;

    /// <summary>Feeds the whole document and closes it, as BeautifulSoup does.</summary>
    protected void Run(string markup)
    {
        _rawdata = markup;
        GoAhead(false);
        GoAhead(true);
    }

    protected abstract void HandleStartTag(string tag, List<(string Name, string? Value)> attrs);

    protected abstract void HandleStartEndTag(string tag, List<(string Name, string? Value)> attrs);

    protected abstract void HandleEndTag(string tag);

    protected abstract void HandleData(string data);

    protected abstract void HandleCharRef(string name);

    protected abstract void HandleEntityRef(string name);

    protected abstract void HandleComment(string data);

    protected abstract void HandleDecl(string decl);

    protected abstract void UnknownDecl(string data);

    protected abstract void HandlePi(string data);

    private void SetCdataMode(string element, bool escapable)
    {
        _cdataElement = element.ToLowerInvariant();
        string escaped = Regex.Escape(_cdataElement);
        _interesting = escapable
            ? new Regex($@"&|</{escaped}(?=[\t\n\r\f />])", RegexOptions.IgnoreCase | Options)
            : new Regex($@"</{escaped}(?=[\t\n\r\f />])", RegexOptions.IgnoreCase | Options);
        if (_cdataElement == "plaintext")
        {
            _interesting = new Regex(@"\z", Options);
        }
    }

    private void ClearCdataMode()
    {
        _interesting = InterestingNormal;
        _cdataElement = null;
    }

    private void GoAhead(bool end)
    {
        string rawdata = _rawdata;
        int i = 0;
        int n = rawdata.Length;
        while (i < n)
        {
            int j;
            Match interesting = _interesting.Match(rawdata, i);
            if (interesting.Success)
            {
                j = interesting.Index;
            }
            else
            {
                if (_cdataElement is not null)
                {
                    break;
                }

                j = n;
            }

            if (i < j)
            {
                HandleData(rawdata[i..j]);
            }

            i = j;
            if (i == n)
            {
                break;
            }

            if (rawdata[i] == '<')
            {
                int k;
                if (StartTagOpen.IsMatch(rawdata, i))
                {
                    k = ParseStartTag(i);
                }
                else if (StartsWith(rawdata, "</", i))
                {
                    k = ParseEndTag(i);
                }
                else if (StartsWith(rawdata, "<!--", i))
                {
                    k = ParseComment(i);
                }
                else if (StartsWith(rawdata, "<?", i))
                {
                    k = ParsePi(i);
                }
                else if (StartsWith(rawdata, "<!", i))
                {
                    k = ParseHtmlDeclaration(i);
                }
                else if (i + 1 < n || end)
                {
                    HandleData("<");
                    k = i + 1;
                }
                else
                {
                    break;
                }

                if (k < 0)
                {
                    if (!end)
                    {
                        break;
                    }

                    if (StartTagOpen.IsMatch(rawdata, i))
                    {
                        // An unterminated start tag at the end is dropped.
                    }
                    else if (StartsWith(rawdata, "</", i))
                    {
                        if (i + 2 == n)
                        {
                            HandleData("</");
                        }
                        else if (EndTagOpen.IsMatch(rawdata, i))
                        {
                        }
                        else
                        {
                            HandleComment(rawdata[(i + 2)..]);
                        }
                    }
                    else if (StartsWith(rawdata, "<!--", i))
                    {
                        int stop = n;
                        foreach (string suffix in (string[])["--!", "--", "-"])
                        {
                            if (rawdata.Length - suffix.Length >= i + 4 && rawdata.EndsWith(suffix, StringComparison.Ordinal))
                            {
                                stop -= suffix.Length;
                                break;
                            }
                        }

                        HandleComment(rawdata[(i + 4)..stop]);
                    }
                    else if (StartsWith(rawdata, "<![CDATA[", i))
                    {
                        UnknownDecl(rawdata[(i + 3)..]);
                    }
                    else if (rawdata.Length >= i + 9 && rawdata.Substring(i, 9).Equals("<!doctype", StringComparison.OrdinalIgnoreCase))
                    {
                        HandleDecl(rawdata[(i + 2)..]);
                    }
                    else if (StartsWith(rawdata, "<!", i))
                    {
                        HandleComment(rawdata[(i + 2)..]);
                    }
                    else if (StartsWith(rawdata, "<?", i))
                    {
                        HandlePi(rawdata[(i + 2)..]);
                    }

                    k = n;
                }

                i = k;
            }
            else if (StartsWith(rawdata, "&#", i))
            {
                Match match = CharRef.Match(rawdata, i);
                if (match.Success)
                {
                    HandleCharRef(match.Value[2..^1]);
                    int k = match.Index + match.Length;
                    if (rawdata[k - 1] != ';')
                    {
                        k--;
                    }

                    i = k;
                    continue;
                }

                if (IncompleteCharRef.IsMatch(rawdata, i))
                {
                    if (end)
                    {
                        HandleCharRef(rawdata[(i + 2)..]);
                        i = n;
                    }

                    break;
                }
                else if (i + 3 < n)
                {
                    HandleData("&#");
                    i += 2;
                }
                else
                {
                    break;
                }
            }
            else if (rawdata[i] == '&')
            {
                Match match = EntityRef.Match(rawdata, i);
                if (match.Success)
                {
                    HandleEntityRef(match.Groups[1].Value);
                    int k = match.Index + match.Length;
                    if (rawdata[k - 1] != ';')
                    {
                        k--;
                    }

                    i = k;
                    continue;
                }

                if (Incomplete.IsMatch(rawdata, i))
                {
                    if (end)
                    {
                        HandleEntityRef(rawdata[(i + 1)..]);
                        i = n;
                    }

                    break;
                }
                else if (i + 1 < n)
                {
                    HandleData("&");
                    i++;
                }
                else
                {
                    break;
                }
            }
            else
            {
                throw new InvalidOperationException("interesting.search() lied");
            }
        }

        if (end && i < n)
        {
            HandleData(rawdata[i..n]);
            i = n;
        }

        _rawdata = rawdata[i..];
    }

    private static bool StartsWith(string s, string prefix, int at) =>
        string.CompareOrdinal(s, at, prefix, 0, prefix.Length) == 0 && s.Length - at >= prefix.Length;

    private int ParseHtmlDeclaration(int i)
    {
        string rawdata = _rawdata;
        if (StartsWith(rawdata, "<!--", i))
        {
            return ParseComment(i);
        }

        if (StartsWith(rawdata, "<![CDATA[", i))
        {
            int j = rawdata.IndexOf("]]>", i + 9, StringComparison.Ordinal);
            if (j < 0)
            {
                return -1;
            }

            UnknownDecl(rawdata[(i + 3)..j]);
            return j + 3;
        }

        if (rawdata.Length >= i + 9 && rawdata.Substring(i, 9).Equals("<!doctype", StringComparison.OrdinalIgnoreCase))
        {
            int gt = rawdata.IndexOf('>', i + 9);
            if (gt == -1)
            {
                return -1;
            }

            HandleDecl(rawdata[(i + 2)..gt]);
            return gt + 1;
        }

        return ParseBogusComment(i);
    }

    private int ParseComment(int i)
    {
        string rawdata = _rawdata;
        Match match = CommentAbruptClose.Match(rawdata, i + 4);
        if (!match.Success)
        {
            match = CommentClose.Match(rawdata, i + 4);
            if (!match.Success)
            {
                return -1;
            }
        }

        HandleComment(rawdata[(i + 4)..match.Index]);
        return match.Index + match.Length;
    }

    private int ParseBogusComment(int i)
    {
        int pos = _rawdata.IndexOf('>', i + 2);
        if (pos == -1)
        {
            return -1;
        }

        HandleComment(_rawdata[(i + 2)..pos]);
        return pos + 1;
    }

    private int ParsePi(int i)
    {
        Match match = PiClose.Match(_rawdata, i + 2);
        if (!match.Success)
        {
            return -1;
        }

        HandlePi(_rawdata[(i + 2)..match.Index]);
        return match.Index + match.Length;
    }

    private int ParseStartTag(int i)
    {
        string rawdata = _rawdata;
        Match whole = LocateTagEnd.Match(rawdata, i + 1);
        int endpos = whole.Index + whole.Length;
        if (rawdata[endpos - 1] != '>')
        {
            return -1;
        }

        var attrs = new List<(string, string?)>();
        Match tagMatch = TagFindTolerant.Match(rawdata, i + 1);
        int k = tagMatch.Index + tagMatch.Length;
        string tag = tagMatch.Groups[1].Value.ToLowerInvariant();
        while (k < endpos)
        {
            Match m = AttrFindTolerant.Match(rawdata, k);
            if (!m.Success)
            {
                break;
            }

            string name = m.Groups[1].Value;
            string? value = m.Groups[2].Success && m.Groups[2].Length > 0 ? m.Groups[3].Value : null;
            if (value is not null && value.Length >= 1
                && ((value[0] == '\'' && value[^1] == '\'') || (value[0] == '"' && value[^1] == '"')))
            {
                // attrvalue[:1] == "'" == attrvalue[-1:] holds for a lone quote too.
                value = value.Length >= 2 ? value[1..^1] : string.Empty;
            }

            if (!string.IsNullOrEmpty(value))
            {
                value = UnescapeAttributeValue(value);
            }

            attrs.Add((name.ToLowerInvariant(), value));
            k = m.Index + m.Length;
        }

        string tail = PyString.Strip(rawdata[k..endpos]);
        if (tail != ">" && tail != "/>")
        {
            HandleData(rawdata[i..endpos]);
            return endpos;
        }

        if (tail.EndsWith("/>", StringComparison.Ordinal))
        {
            HandleStartEndTag(tag, attrs);
        }
        else
        {
            HandleStartTag(tag, attrs);
            if (Array.IndexOf(CdataContentElements, tag) >= 0 || tag == "plaintext")
            {
                SetCdataMode(tag, escapable: false);
            }
            else if (Array.IndexOf(RcdataContentElements, tag) >= 0)
            {
                SetCdataMode(tag, escapable: true);
            }
        }

        return endpos;
    }

    private int ParseEndTag(int i)
    {
        string rawdata = _rawdata;
        if (rawdata.IndexOf('>', i + 2) < 0)
        {
            return -1;
        }

        if (!EndTagOpen.IsMatch(rawdata, i))
        {
            if (rawdata.Length > i + 2 && rawdata[i + 2] == '>')
            {
                return i + 3;
            }

            return ParseBogusComment(i);
        }

        Match whole = LocateTagEnd.Match(rawdata, i + 2);
        int j = whole.Index + whole.Length;
        if (rawdata[j - 1] != '>')
        {
            return -1;
        }

        Match tagMatch = TagFindTolerant.Match(rawdata, i + 2);
        HandleEndTag(tagMatch.Groups[1].Value.ToLowerInvariant());
        ClearCdataMode();
        return j;
    }

    /// <summary><c>_unescape_attrvalue</c>.</summary>
    private static string UnescapeAttributeValue(string value) =>
        AttrCharRef.Replace(value, match =>
        {
            string reference = match.Value;
            if (reference.StartsWith("&#", StringComparison.Ordinal))
            {
                return PyString.Unescape(reference);
            }

            if (!reference.EndsWith('=') && HtmlEntities.Html5.ContainsKey(reference[1..]))
            {
                return PyString.Unescape(reference);
            }

            return reference;
        });
}
