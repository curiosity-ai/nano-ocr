using System.Globalization;
using System.Text;
using System.Text.Json;

namespace IndraOcr.Vlm.Text;

/// <summary>
/// A BERT <c>tokenizer.json</c>: <c>BertNormalizer</c>, <c>BertPreTokenizer</c>, a
/// <c>WordPiece</c> model and the <c>[CLS] A [SEP]</c> template — what Surya's OCR-error
/// classifier (multilingual DistilBERT) ships, following the <c>tokenizers</c> library's rules.
/// </summary>
public sealed class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly string _prefix;
    private readonly int _maxWordChars;
    private readonly int _unknown;
    private readonly bool _cleanText;
    private readonly bool _chineseChars;
    private readonly bool _lowercase;
    private readonly bool _stripAccents;
    private readonly List<(string Content, int Id)> _added;

    private WordPieceTokenizer(
        Dictionary<string, int> vocab,
        string prefix,
        int maxWordChars,
        int unknown,
        bool cleanText,
        bool chineseChars,
        bool lowercase,
        bool stripAccents,
        List<(string, int)> added)
    {
        _vocab = vocab;
        _prefix = prefix;
        _maxWordChars = maxWordChars;
        _unknown = unknown;
        _cleanText = cleanText;
        _chineseChars = chineseChars;
        _lowercase = lowercase;
        _stripAccents = stripAccents;
        _added = added.OrderByDescending(a => a.Item1.Length).ToList();
        ClsId = vocab["[CLS]"];
        SepId = vocab["[SEP]"];
        PadId = vocab.TryGetValue("[PAD]", out int pad) ? pad : 0;
    }

    /// <summary><c>[CLS]</c>.</summary>
    public int ClsId { get; }

    /// <summary><c>[SEP]</c>.</summary>
    public int SepId { get; }

    /// <summary><c>[PAD]</c>.</summary>
    public int PadId { get; }

    /// <summary>Loads <c>tokenizer.json</c>.</summary>
    public static WordPieceTokenizer FromFile(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonElement root = document.RootElement;
        JsonElement model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "WordPiece")
        {
            throw new NotSupportedException("Not a WordPiece tokenizer.");
        }

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (JsonProperty entry in model.GetProperty("vocab").EnumerateObject())
        {
            vocab[entry.Name] = entry.Value.GetInt32();
        }

        JsonElement normalizer = root.GetProperty("normalizer");
        bool lowercase = normalizer.TryGetProperty("lowercase", out JsonElement lc) && lc.ValueKind == JsonValueKind.True;
        bool stripAccents = normalizer.TryGetProperty("strip_accents", out JsonElement sa)
            ? sa.ValueKind == JsonValueKind.True || (sa.ValueKind == JsonValueKind.Null && lowercase)
            : lowercase;
        var added = root.GetProperty("added_tokens").EnumerateArray()
            .Select(a => (a.GetProperty("content").GetString()!, a.GetProperty("id").GetInt32())).ToList();
        return new WordPieceTokenizer(
            vocab,
            model.GetProperty("continuing_subword_prefix").GetString() ?? "##",
            model.TryGetProperty("max_input_chars_per_word", out JsonElement max) ? max.GetInt32() : 100,
            vocab[model.GetProperty("unk_token").GetString()!],
            normalizer.TryGetProperty("clean_text", out JsonElement ct) && ct.ValueKind == JsonValueKind.True,
            normalizer.TryGetProperty("handle_chinese_chars", out JsonElement hc) && hc.ValueKind == JsonValueKind.True,
            lowercase,
            stripAccents,
            added);
    }

    /// <summary>Encodes one text as <c>[CLS] … [SEP]</c>, truncated to <paramref name="maxLength"/>.</summary>
    public List<int> Encode(string text, int maxLength = 512)
    {
        var ids = new List<int>();
        int cursor = 0;
        while (cursor < text.Length)
        {
            int found = -1;
            (string Content, int Id) match = default;
            for (int i = cursor; i < text.Length && found < 0; i++)
            {
                foreach ((string content, int id) in _added)
                {
                    if (string.CompareOrdinal(text, i, content, 0, content.Length) == 0 && i + content.Length <= text.Length)
                    {
                        found = i;
                        match = (content, id);
                        break;
                    }
                }
            }

            if (found < 0)
            {
                EncodeSegment(text[cursor..], ids);
                break;
            }

            EncodeSegment(text[cursor..found], ids);
            ids.Add(match.Id);
            cursor = found + match.Content.Length;
        }

        if (ids.Count > maxLength - 2)
        {
            ids.RemoveRange(maxLength - 2, ids.Count - (maxLength - 2));
        }

        ids.Insert(0, ClsId);
        ids.Add(SepId);
        return ids;
    }

    private void EncodeSegment(string text, List<int> ids)
    {
        string normalized = Normalize(text);
        foreach (string word in PreTokenize(normalized))
        {
            WordPiece(word, ids);
        }
    }

    private string Normalize(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (Rune rune in text.EnumerateRunes())
        {
            int c = rune.Value;
            if (_cleanText)
            {
                if (c == 0 || c == 0xFFFD || IsControl(rune))
                {
                    continue;
                }

                if (IsWhitespace(rune))
                {
                    builder.Append(' ');
                    continue;
                }
            }

            if (_chineseChars && IsChinese(c))
            {
                builder.Append(' ').Append(rune.ToString()).Append(' ');
                continue;
            }

            builder.Append(rune.ToString());
        }

        string result = builder.ToString();
        if (_stripAccents)
        {
            result = new string(result.Normalize(NormalizationForm.FormD)
                .Where(ch => CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark).ToArray());
        }

        return _lowercase ? result.ToLowerInvariant() : result;
    }

    /// <summary><c>BertPreTokenizer</c>: split on whitespace, each punctuation character on its own.</summary>
    private static IEnumerable<string> PreTokenize(string text)
    {
        var current = new StringBuilder();
        foreach (Rune rune in text.EnumerateRunes())
        {
            if (Rune.IsWhiteSpace(rune))
            {
                if (current.Length > 0)
                {
                    yield return current.ToString();
                    current.Clear();
                }

                continue;
            }

            if (IsPunctuation(rune))
            {
                if (current.Length > 0)
                {
                    yield return current.ToString();
                    current.Clear();
                }

                yield return rune.ToString();
                continue;
            }

            current.Append(rune.ToString());
        }

        if (current.Length > 0)
        {
            yield return current.ToString();
        }
    }

    private void WordPiece(string word, List<int> ids)
    {
        int runes = word.EnumerateRunes().Count();
        if (runes > _maxWordChars)
        {
            ids.Add(_unknown);
            return;
        }

        var pieces = new List<int>();
        int start = 0;
        while (start < word.Length)
        {
            int end = word.Length;
            int found = -1;
            while (start < end)
            {
                string piece = start > 0 ? _prefix + word[start..end] : word[start..end];
                if (_vocab.TryGetValue(piece, out int id))
                {
                    found = id;
                    break;
                }

                end--;
                if (end > start && char.IsLowSurrogate(word[end]))
                {
                    end--;
                }
            }

            if (found < 0)
            {
                ids.Add(_unknown);
                return;
            }

            pieces.Add(found);
            start = end;
        }

        ids.AddRange(pieces);
    }

    private static bool IsWhitespace(Rune r) => r.Value is '\t' or '\n' or '\r' || Rune.IsWhiteSpace(r);

    private static bool IsControl(Rune r)
    {
        if (r.Value is '\t' or '\n' or '\r')
        {
            return false;
        }

        UnicodeCategory category = Rune.GetUnicodeCategory(r);
        return category is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate
            or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned;
    }

    private static bool IsPunctuation(Rune r)
    {
        int c = r.Value;
        if (c is >= 33 and <= 47 or >= 58 and <= 64 or >= 91 and <= 96 or >= 123 and <= 126)
        {
            return true;
        }

        UnicodeCategory category = Rune.GetUnicodeCategory(r);
        return category is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    private static bool IsChinese(int c) =>
        c is >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF or >= 0x20000 and <= 0x2A6DF or >= 0x2A700 and <= 0x2B73F
        or >= 0x2B740 and <= 0x2B81F or >= 0x2B820 and <= 0x2CEAF or >= 0xF900 and <= 0xFAFF or >= 0x2F800 and <= 0x2FA1F;
}
