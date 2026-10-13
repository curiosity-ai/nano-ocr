using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using IndraOcr.Vlm.Html;

namespace IndraOcr.Vlm.Surya;

/// <summary>A layout block as the layout prompt reports it, box normalised to 0-1000.</summary>
/// <param name="Label">Block label.</param>
/// <param name="Bbox"><c>(x0, y0, x1, y1)</c>.</param>
/// <param name="Count">The model's estimate of the block's token count.</param>
public sealed record ParsedLayoutBlock(string Label, double[] Bbox, int Count);

/// <summary>A table row or column, box normalised to 0-1000.</summary>
public sealed record ParsedTableElement(string Label, double[] Bbox);

/// <summary>A top-level block of the full-page HTML prompt.</summary>
public sealed record ParsedFullPageBlock(string Label, double[] Bbox, string Html);

/// <summary><c>surya/inference/parsers.py</c>.</summary>
public static class SuryaParsers
{
    private static readonly Regex JsonArray = new(@"\[.*\]", RegexOptions.Singleline | RegexOptions.CultureInvariant);
    private static readonly Regex FenceOpen = new(@"^```[a-zA-Z]*\n", RegexOptions.CultureInvariant);
    private static readonly Regex FenceClose = new(@"\n```\s*$", RegexOptions.CultureInvariant);

    /// <summary><c>_strip_fences</c>.</summary>
    public static string StripFences(string text)
    {
        string cleaned = PyString.Strip(text);
        if (cleaned.StartsWith("```", StringComparison.Ordinal))
        {
            cleaned = FenceOpen.Replace(cleaned, string.Empty, 1);
            cleaned = FenceClose.Replace(cleaned, string.Empty, 1);
        }

        return cleaned;
    }

    /// <summary><c>clean_block_html</c>.</summary>
    public static string CleanBlockHtml(string html) => PyString.Strip(StripFences(html));

    /// <summary><c>parse_layout</c>: the JSON array of the layout prompt's output.</summary>
    public static List<ParsedLayoutBlock> ParseLayout(string text)
    {
        var output = new List<ParsedLayoutBlock>();
        foreach (JsonElement item in ParseArray(text, "layout"))
        {
            if (item.ValueKind != JsonValueKind.Object || !item.TryGetProperty("bbox", out JsonElement box)
                || CoerceBox(box) is not { } bbox)
            {
                continue;
            }

            string label = item.TryGetProperty("label", out JsonElement l) ? PyStr(l) : "block";
            int count = item.TryGetProperty("count", out JsonElement c) ? CoerceCount(c) : 0;
            output.Add(new ParsedLayoutBlock(label, bbox, count));
        }

        return output;
    }

    /// <summary><c>parse_table_rec</c>: the Row and Col entries of the table prompt's output.</summary>
    public static List<ParsedTableElement> ParseTableRec(string text)
    {
        var output = new List<ParsedTableElement>();
        foreach (JsonElement item in ParseArray(text, "table_rec"))
        {
            if (item.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            string label = PyString.Strip(item.TryGetProperty("label", out JsonElement l) ? PyStr(l) : string.Empty);
            if (label is not ("Row" or "Col") || !item.TryGetProperty("bbox", out JsonElement box) || CoerceBox(box) is not { } bbox)
            {
                continue;
            }

            output.Add(new ParsedTableElement(label, bbox));
        }

        return output;
    }

    /// <summary><c>parse_full_page_html</c>: top-level divs with a label and a four-number box.</summary>
    public static List<ParsedFullPageBlock> ParseFullPageHtml(string text)
    {
        string cleaned = PyString.Strip(StripFences(text));
        var output = new List<ParsedFullPageBlock>();
        if (cleaned.Length == 0)
        {
            return output;
        }

        HtmlElement soup = Soup.Parse(cleaned);
        foreach (HtmlElement div in soup.FindAll("div", recursive: false))
        {
            string? label = div.Get("data-label");
            string? boxText = div.Get("data-bbox");
            if (string.IsNullOrEmpty(label) || string.IsNullOrEmpty(boxText))
            {
                continue;
            }

            string[] parts = SplitWhitespace(boxText);
            var values = new double[parts.Length];
            bool ok = true;
            for (int i = 0; i < parts.Length && ok; i++)
            {
                ok = PyFloat.TryParse(parts[i], out values[i]);
            }

            if (!ok || values.Length != 4)
            {
                continue;
            }

            foreach (HtmlElement tag in div.FindAllWithAttribute("data-bbox"))
            {
                tag.RemoveAttribute("data-bbox");
            }

            foreach (HtmlElement tag in div.FindAllWithAttribute("data-label"))
            {
                tag.RemoveAttribute("data-label");
            }

            output.Add(new ParsedFullPageBlock(label, values, PyString.Strip(div.DecodeContents())));
        }

        return output;
    }

    /// <summary><c>denorm_bbox</c>: a 0-<paramref name="scale"/> box in pixels.</summary>
    public static double[] DenormBbox(double[] bbox, int width, int height, int scale = 1000) =>
        [bbox[0] / scale * width, bbox[1] / scale * height, bbox[2] / scale * width, bbox[3] / scale * height];

    private static List<JsonElement> ParseArray(string text, string what)
    {
        string cleaned = StripFences(text);
        Match match = JsonArray.Match(cleaned);
        if (!match.Success)
        {
            throw new FormatException($"No JSON array found in {what} output: {text[..Math.Min(500, text.Length)]}");
        }

        using JsonDocument document = JsonDocument.Parse(match.Value);
        if (document.RootElement.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException($"{what} output is not a JSON array");
        }

        return document.RootElement.EnumerateArray().Select(e => e.Clone()).ToList();
    }

    /// <summary><c>_coerce_bbox</c>: a string of four numbers (commas allowed) or a list of four.</summary>
    private static double[]? CoerceBox(JsonElement box)
    {
        var values = new List<double>();
        if (box.ValueKind == JsonValueKind.String)
        {
            foreach (string part in SplitWhitespace(box.GetString()!.Replace(',', ' ')))
            {
                if (!PyFloat.TryParse(part, out double v))
                {
                    return null;
                }

                values.Add(v);
            }
        }
        else if (box.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement part in box.EnumerateArray())
            {
                if (part.ValueKind == JsonValueKind.Number)
                {
                    values.Add(part.GetDouble());
                }
                else if (part.ValueKind == JsonValueKind.String && PyFloat.TryParse(part.GetString()!, out double v))
                {
                    values.Add(v);
                }
                else if (part.ValueKind is JsonValueKind.True or JsonValueKind.False)
                {
                    values.Add(part.ValueKind == JsonValueKind.True ? 1 : 0);
                }
                else
                {
                    return null;
                }
            }
        }
        else
        {
            return null;
        }

        return values.Count == 4 ? [.. values] : null;
    }

    /// <summary><c>_coerce_count</c>.</summary>
    private static int CoerceCount(JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Number:
                double d = value.GetDouble();
                return double.IsFinite(d) ? Math.Max(0, (int)Math.Truncate(Math.Clamp(d, int.MinValue, int.MaxValue))) : 0;
            case JsonValueKind.String:
                return int.TryParse(PyString.Strip(value.GetString()!), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int i)
                    ? Math.Max(0, i)
                    : 0;
            case JsonValueKind.True:
                return 1;
            default:
                return 0;
        }
    }

    /// <summary><c>str(value)</c> of a JSON value.</summary>
    private static string PyStr(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.String => value.GetString()!,
        JsonValueKind.Null => "None",
        JsonValueKind.True => "True",
        JsonValueKind.False => "False",
        _ => value.GetRawText(),
    };

    private static string[] SplitWhitespace(string s) =>
        s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
}

/// <summary>Python's <c>float(str)</c>.</summary>
internal static class PyFloat
{
    public static bool TryParse(string s, out double value)
    {
        string t = PyString.Strip(s).Replace("_", string.Empty, StringComparison.Ordinal);
        string lower = t.ToLowerInvariant().TrimStart('+', '-');
        if (lower is "inf" or "infinity" or "nan")
        {
            value = lower == "nan" ? double.NaN : t.StartsWith('-') ? double.NegativeInfinity : double.PositiveInfinity;
            return true;
        }

        return double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }
}
