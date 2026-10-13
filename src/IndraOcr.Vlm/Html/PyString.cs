using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace IndraOcr.Vlm.Html;

/// <summary>Python <c>str</c> semantics the HTML stages depend on.</summary>
public static class PyString
{
    private static readonly Regex CharRef = new(@"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)", RegexOptions.CultureInvariant);

    /// <summary>Windows-1252 for 0x80-0x9F, and the two other remapped references (<c>html._invalid_charrefs</c>).</summary>
    private static readonly Dictionary<int, string> InvalidCharRefs = new()
    {
        [0x00] = "�", [0x0D] = "\r", [0x80] = "€", [0x81] = "\u0081", [0x82] = "‚", [0x83] = "ƒ",
        [0x84] = "„", [0x85] = "…", [0x86] = "†", [0x87] = "‡", [0x88] = "ˆ", [0x89] = "‰",
        [0x8A] = "Š", [0x8B] = "‹", [0x8C] = "Œ", [0x8D] = "\u008D", [0x8E] = "Ž", [0x8F] = "\u008F",
        [0x90] = "\u0090", [0x91] = "‘", [0x92] = "’", [0x93] = "“", [0x94] = "”", [0x95] = "•",
        [0x96] = "–", [0x97] = "—", [0x98] = "˜", [0x99] = "™", [0x9A] = "š", [0x9B] = "›",
        [0x9C] = "œ", [0x9D] = "\u009D", [0x9E] = "ž", [0x9F] = "Ÿ",
    };

    /// <summary>Python's <c>str.isspace</c> for one UTF-16 unit.</summary>
    public static bool IsSpace(char c) =>
        c is ' ' or '\t' or '\n' or '\r' or '\v' or '\f' or '\x1c' or '\x1d' or '\x1e' or '\x1f' or '\x85'
        || (c > 127 && char.IsWhiteSpace(c));

    /// <summary><c>str.strip()</c>.</summary>
    public static string Strip(string s)
    {
        int start = 0;
        int end = s.Length;
        while (start < end && IsSpace(s[start]))
        {
            start++;
        }

        while (end > start && IsSpace(s[end - 1]))
        {
            end--;
        }

        return start == 0 && end == s.Length ? s : s[start..end];
    }

    /// <summary><c>str.lstrip()</c>.</summary>
    public static string LStrip(string s)
    {
        int start = 0;
        while (start < s.Length && IsSpace(s[start]))
        {
            start++;
        }

        return s[start..];
    }

    /// <summary><c>str.rstrip()</c>.</summary>
    public static string RStrip(string s)
    {
        int end = s.Length;
        while (end > 0 && IsSpace(s[end - 1]))
        {
            end--;
        }

        return s[..end];
    }

    /// <summary><c>str.strip(chars)</c>.</summary>
    public static string Strip(string s, string chars) => s.Trim(chars.ToCharArray());

    /// <summary><c>str.isdigit()</c>: non-empty and every character a Unicode digit (Nd, and digit-valued No).</summary>
    public static bool IsDigit(string s)
    {
        if (s.Length == 0)
        {
            return false;
        }

        foreach (Rune r in s.EnumerateRunes())
        {
            UnicodeCategory category = Rune.GetUnicodeCategory(r);
            if (category != UnicodeCategory.DecimalDigitNumber
                && !(category == UnicodeCategory.OtherNumber && CharUnicodeInfo.GetDigitValue(r.ToString(), 0) >= 0))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary><c>html.unescape</c>.</summary>
    public static string Unescape(string s)
    {
        if (!s.Contains('&'))
        {
            return s;
        }

        return CharRef.Replace(s, match =>
        {
            string body = match.Groups[1].Value;
            if (body[0] == '#')
            {
                long number;
                try
                {
                    number = body.Length > 1 && (body[1] == 'x' || body[1] == 'X')
                        ? long.Parse(body[2..].TrimEnd(';'), NumberStyles.HexNumber, CultureInfo.InvariantCulture)
                        : long.Parse(body[1..].TrimEnd(';'), CultureInfo.InvariantCulture);
                }
                catch (OverflowException)
                {
                    return "�";
                }

                return NumericReference(number, html: true);
            }

            if (HtmlEntities.Html5.TryGetValue(body, out string? value))
            {
                return value;
            }

            for (int x = body.Length - 1; x > 1; x--)
            {
                if (HtmlEntities.Html5.TryGetValue(body[..x], out value))
                {
                    return value + body[x..];
                }
            }

            return "&" + body;
        });
    }

    /// <summary>
    /// A numeric character reference's text: <c>html.unescape</c>'s rules when
    /// <paramref name="html"/>, else bs4's <c>UnicodeDammit.numeric_character_reference</c>.
    /// </summary>
    internal static string NumericReference(long number, bool html)
    {
        if (html)
        {
            if (InvalidCharRefs.TryGetValue((int)Math.Min(number, int.MaxValue), out string? mapped))
            {
                return mapped;
            }

            if (number is >= 0xD800 and <= 0xDFFF || number > 0x10FFFF)
            {
                return "�";
            }

            if (IsInvalidCodepoint(number))
            {
                return string.Empty;
            }

            return char.ConvertFromUtf32((int)number);
        }

        if (number == 0 || number > 0x10FFFF || number is >= 0xD800 and <= 0xDFFF)
        {
            return "�";
        }

        if (number is >= 0x80 and <= 0x9F && InvalidCharRefs.TryGetValue((int)number, out string? windows))
        {
            return windows;
        }

        return char.ConvertFromUtf32((int)number);
    }

    private static bool IsInvalidCodepoint(long n) =>
        n is >= 0x1 and <= 0x8 or 0xB or >= 0xE and <= 0x1F or >= 0x7F and <= 0x9F or >= 0xFDD0 and <= 0xFDEF
        || (n & 0xFFFE) == 0xFFFE && n <= 0x10FFFF;
}
