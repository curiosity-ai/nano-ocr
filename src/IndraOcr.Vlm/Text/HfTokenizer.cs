using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace IndraOcr.Vlm.Text;

/// <summary>
/// A Hugging Face <c>tokenizer.json</c> tokenizer, covering the two pipelines the Qwen3.5 OCR
/// checkpoints ship.
/// </summary>
/// <remarks>
/// <list type="bullet">
///   <item><b>Surya 2</b>: no normalizer, <c>Split(Regex("."), Isolated)</c>, a <c>WordLevel</c>
///   model and a <c>Fuse</c> decoder — every code point is a token, apart from the ~1,900 added
///   tokens (numbers, HTML tags, LaTeX commands) matched before the split.</item>
///   <item><b>Chandra 2</b>: <c>NFC</c>, the Qwen regex split followed by <c>ByteLevel</c>, a
///   byte-level <c>BPE</c> model and a <c>ByteLevel</c> decoder.</item>
/// </list>
/// <para>
/// This follows the file, not <c>transformers</c>: transformers 5 loads both checkpoints as its
/// own Qwen3.5 tokenizer class and replaces Surya's pipeline with the byte-level one, which
/// encodes a space as <c>Ġ</c> (id 2224) where the model was trained on <c>' '</c> (id 1979) and
/// visibly derails its first tokens. The <c>tokenizers</c> library — what vLLM serves through —
/// runs the file's pipeline, and so does this.
/// </para>
/// </remarks>
public sealed class HfTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly string[] _idToToken;
    private readonly bool[] _special;
    private readonly AddedTrie _added;
    private readonly ModelKind _model;
    private readonly bool _nfc;
    private readonly Regex? _split;
    private readonly bool _byteLevel;
    private readonly int _unknownId;
    private readonly Dictionary<(string, string), int>? _mergeRanks;
    private readonly Dictionary<string, int[]> _bpeCache = new(StringComparer.Ordinal);

    private enum ModelKind
    {
        WordLevel,
        Bpe,
    }

    private HfTokenizer(
        Dictionary<string, int> vocab,
        string[] idToToken,
        bool[] special,
        AddedTrie added,
        ModelKind model,
        bool nfc,
        Regex? split,
        bool byteLevel,
        int unknownId,
        Dictionary<(string, string), int>? mergeRanks)
    {
        _vocab = vocab;
        _idToToken = idToToken;
        _special = special;
        _added = added;
        _model = model;
        _nfc = nfc;
        _split = split;
        _byteLevel = byteLevel;
        _unknownId = unknownId;
        _mergeRanks = mergeRanks;
    }

    /// <summary>Size of the id space (vocabulary plus added tokens).</summary>
    public int VocabSize => _idToToken.Length;

    /// <summary>Loads <c>tokenizer.json</c>.</summary>
    public static HfTokenizer FromFile(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return FromJson(stream);
    }

    /// <summary>Loads a <c>tokenizer.json</c> stream.</summary>
    public static HfTokenizer FromJson(Stream stream)
    {
        using JsonDocument document = JsonDocument.Parse(stream);
        JsonElement root = document.RootElement;

        bool nfc = false;
        if (root.TryGetProperty("normalizer", out JsonElement normalizer) && normalizer.ValueKind != JsonValueKind.Null)
        {
            string type = normalizer.GetProperty("type").GetString()!;
            nfc = type == "NFC" ? true : throw new NotSupportedException($"Normalizer '{type}' is not supported.");
        }

        Regex? split = null;
        bool byteLevel = false;
        if (root.TryGetProperty("pre_tokenizer", out JsonElement pre) && pre.ValueKind != JsonValueKind.Null)
        {
            IEnumerable<JsonElement> steps = pre.GetProperty("type").GetString() == "Sequence"
                ? pre.GetProperty("pretokenizers").EnumerateArray()
                : [pre];
            foreach (JsonElement step in steps)
            {
                switch (step.GetProperty("type").GetString())
                {
                    case "Split":
                        if (step.GetProperty("behavior").GetString() != "Isolated" || step.GetProperty("invert").GetBoolean())
                        {
                            throw new NotSupportedException("Only an isolated, non-inverted Split is supported.");
                        }

                        split = TranslateRegex(step.GetProperty("pattern").GetProperty("Regex").GetString()!);
                        break;
                    case "ByteLevel":
                        if (step.TryGetProperty("add_prefix_space", out JsonElement prefix) && prefix.GetBoolean())
                        {
                            throw new NotSupportedException("ByteLevel add_prefix_space is not supported.");
                        }

                        byteLevel = true;
                        break;
                    default:
                        throw new NotSupportedException($"Pre-tokenizer '{step.GetProperty("type").GetString()}' is not supported.");
                }
            }
        }

        JsonElement model = root.GetProperty("model");
        string modelType = model.GetProperty("type").GetString()!;
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (JsonProperty entry in model.GetProperty("vocab").EnumerateObject())
        {
            vocab[entry.Name] = entry.Value.GetInt32();
        }

        Dictionary<(string, string), int>? ranks = null;
        ModelKind kind;
        if (modelType == "WordLevel")
        {
            kind = ModelKind.WordLevel;
        }
        else if (modelType == "BPE")
        {
            kind = ModelKind.Bpe;
            ranks = new Dictionary<(string, string), int>();
            int rank = 0;
            foreach (JsonElement merge in model.GetProperty("merges").EnumerateArray())
            {
                (string left, string right) = merge.ValueKind == JsonValueKind.Array
                    ? (merge[0].GetString()!, merge[1].GetString()!)
                    : SplitMerge(merge.GetString()!);
                ranks.TryAdd((left, right), rank++);
            }
        }
        else
        {
            throw new NotSupportedException($"Model '{modelType}' is not supported.");
        }

        var addedTokens = new List<(string Content, int Id, bool Special)>();
        foreach (JsonElement entry in root.GetProperty("added_tokens").EnumerateArray())
        {
            string content = entry.GetProperty("content").GetString()!;
            int id = entry.GetProperty("id").GetInt32();
            if (entry.GetProperty("single_word").GetBoolean() || entry.GetProperty("lstrip").GetBoolean()
                || entry.GetProperty("rstrip").GetBoolean() || entry.GetProperty("normalized").GetBoolean())
            {
                throw new NotSupportedException($"Added token '{content}' uses options this tokenizer does not implement.");
            }

            addedTokens.Add((content, id, entry.GetProperty("special").GetBoolean()));
            vocab.TryAdd(content, id);
        }

        int size = vocab.Values.Max() + 1;
        var idToToken = new string[size];
        foreach ((string token, int id) in vocab)
        {
            idToToken[id] = token;
        }

        var special = new bool[size];
        var trie = new AddedTrie();
        foreach ((string content, int id, bool isSpecial) in addedTokens)
        {
            special[id] = isSpecial;
            idToToken[id] = content;
            trie.Add(content, id);
        }

        int unknownId = model.TryGetProperty("unk_token", out JsonElement unk) && unk.ValueKind == JsonValueKind.String
            && vocab.TryGetValue(unk.GetString()!, out int u) ? u : -1;

        if (root.TryGetProperty("decoder", out JsonElement decoder) && decoder.ValueKind != JsonValueKind.Null)
        {
            string type = decoder.GetProperty("type").GetString()!;
            if ((type == "ByteLevel") != byteLevel || (type != "ByteLevel" && type != "Fuse"))
            {
                throw new NotSupportedException($"Decoder '{type}' does not match the pre-tokenizer.");
            }
        }

        return new HfTokenizer(vocab, idToToken, special, trie, kind, nfc, split, byteLevel, unknownId, ranks);
    }

    /// <summary>The id of a token (vocabulary or added), or -1.</summary>
    public int TokenToId(string token) => _vocab.TryGetValue(token, out int id) ? id : -1;

    /// <summary>The literal string of a token id.</summary>
    public string IdToToken(int id) => (uint)id < (uint)_idToToken.Length ? _idToToken[id] ?? string.Empty : string.Empty;

    /// <summary>Whether <paramref name="id"/> is a special added token.</summary>
    public bool IsSpecial(int id) => (uint)id < (uint)_special.Length && _special[id];

    /// <summary>Encodes text: added tokens first (leftmost-longest), then the pipeline on the rest.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>(text.Length + 8);
        int cursor = 0;
        while (cursor < text.Length)
        {
            (int start, int length, int id) = _added.FindLeftmostLongest(text, cursor);
            if (start < 0)
            {
                EncodeSegment(text[cursor..], ids);
                break;
            }

            if (start > cursor)
            {
                EncodeSegment(text[cursor..start], ids);
            }

            ids.Add(id);
            cursor = start + length;
        }

        return ids;
    }

    /// <summary>Decodes ids to text, dropping special tokens when asked.</summary>
    public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true)
    {
        var builder = new StringBuilder();
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)_idToToken.Length || (skipSpecialTokens && _special[id]))
            {
                continue;
            }

            string token = _idToToken[id] ?? string.Empty;
            if (!_byteLevel)
            {
                builder.Append(token);
                continue;
            }

            // The ByteLevel decoder maps every character of every token back to its byte, added
            // tokens included (their characters are all in the byte alphabet's printable range).
            foreach (char c in token)
            {
                if (ByteLevel.CharToByte.TryGetValue(c, out byte b))
                {
                    bytes.Add(b);
                }
                else
                {
                    bytes.AddRange(Encoding.UTF8.GetBytes(c.ToString()));
                }
            }
        }

        if (_byteLevel)
        {
            builder.Append(Encoding.UTF8.GetString(bytes.ToArray()));
        }

        return builder.ToString();
    }

    private void EncodeSegment(string segment, List<int> ids)
    {
        if (_nfc)
        {
            segment = segment.Normalize(NormalizationForm.FormC);
        }

        foreach (string piece in PreTokenize(segment))
        {
            if (_model == ModelKind.WordLevel)
            {
                ids.Add(_vocab.TryGetValue(piece, out int id) ? id : _unknownId);
            }
            else
            {
                ids.AddRange(Bpe(_byteLevel ? ByteLevel.Encode(piece) : piece));
            }
        }
    }

    /// <summary>The split pre-tokenizer: matches and the gaps between them, both kept.</summary>
    private IEnumerable<string> PreTokenize(string text)
    {
        if (_split is null)
        {
            yield return text;
            yield break;
        }

        int last = 0;
        for (Match match = _split.Match(text); match.Success; match = match.NextMatch())
        {
            if (match.Length == 0)
            {
                continue;
            }

            if (match.Index > last)
            {
                yield return text[last..match.Index];
            }

            yield return match.Value;
            last = match.Index + match.Length;
        }

        if (last < text.Length)
        {
            yield return text[last..];
        }
    }

    private int[] Bpe(string word)
    {
        if (_vocab.TryGetValue(word, out int direct))
        {
            return [direct];
        }

        lock (_bpeCache)
        {
            if (_bpeCache.TryGetValue(word, out int[]? cached))
            {
                return cached;
            }
        }

        var parts = new List<string>(word.Length);
        for (int i = 0; i < word.Length; i++)
        {
            parts.Add(char.IsHighSurrogate(word[i]) && i + 1 < word.Length ? word.Substring(i++, 2) : word[i].ToString());
        }

        while (parts.Count > 1)
        {
            int best = -1;
            int bestRank = int.MaxValue;
            for (int i = 0; i + 1 < parts.Count; i++)
            {
                if (_mergeRanks!.TryGetValue((parts[i], parts[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    best = i;
                }
            }

            if (best < 0)
            {
                break;
            }

            // Merge every occurrence of the best pair, left to right, as the reference does.
            string left = parts[best];
            string right = parts[best + 1];
            var merged = new List<string>(parts.Count);
            for (int i = 0; i < parts.Count; i++)
            {
                if (i + 1 < parts.Count && parts[i] == left && parts[i + 1] == right)
                {
                    merged.Add(left + right);
                    i++;
                }
                else
                {
                    merged.Add(parts[i]);
                }
            }

            parts = merged;
        }

        int[] result = parts.Select(p => _vocab.TryGetValue(p, out int id) ? id : _unknownId).Where(id => id >= 0).ToArray();
        lock (_bpeCache)
        {
            _bpeCache[word] = result;
        }

        return result;
    }

    private static (string, string) SplitMerge(string merge)
    {
        int space = merge.IndexOf(' ');
        return (merge[..space], merge[(space + 1)..]);
    }

    /// <summary>
    /// Translates a tokenizers (Oniguruma) pattern to .NET. The two patterns these checkpoints
    /// use need only one change: Oniguruma's <c>.</c> matches one code point, .NET's one UTF-16
    /// unit, so a surrogate pair is matched as a unit explicitly.
    /// </summary>
    private static Regex TranslateRegex(string pattern)
    {
        string translated = pattern == "." ? @"[\uD800-\uDBFF][\uDC00-\uDFFF]|[^\n]" : pattern;
        return new Regex(translated, RegexOptions.CultureInvariant | RegexOptions.Compiled);
    }

    /// <summary>GPT-2's byte-to-unicode table.</summary>
    private static class ByteLevel
    {
        public static readonly char[] ByteToChar = Build();
        public static readonly Dictionary<char, byte> CharToByte = ByteToChar.Select((c, b) => (c, (byte)b)).ToDictionary(x => x.c, x => x.Item2);

        public static string Encode(string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            var chars = new char[bytes.Length];
            for (int i = 0; i < bytes.Length; i++)
            {
                chars[i] = ByteToChar[bytes[i]];
            }

            return new string(chars);
        }

        private static char[] Build()
        {
            var table = new char[256];
            int next = 0;
            for (int b = 0; b < 256; b++)
            {
                bool printable = b is >= '!' and <= '~' or >= 0xA1 and <= 0xAC or >= 0xAE and <= 0xFF;
                table[b] = printable ? (char)b : (char)(256 + next++);
            }

            return table;
        }
    }

    /// <summary>A trie over the added tokens, for leftmost-longest matching.</summary>
    private sealed class AddedTrie
    {
        private readonly Node _root = new();

        public void Add(string content, int id)
        {
            Node node = _root;
            foreach (char c in content)
            {
                node.Children ??= [];
                if (!node.Children.TryGetValue(c, out Node? child))
                {
                    child = new Node();
                    node.Children[c] = child;
                }

                node = child;
            }

            node.Id = id;
        }

        /// <summary>The earliest start with any match, and the longest match there.</summary>
        public (int Start, int Length, int Id) FindLeftmostLongest(string text, int from)
        {
            for (int start = from; start < text.Length; start++)
            {
                Node node = _root;
                int bestLength = 0;
                int bestId = -1;
                for (int i = start; i < text.Length; i++)
                {
                    if (node.Children is null || !node.Children.TryGetValue(text[i], out Node? child))
                    {
                        break;
                    }

                    node = child;
                    if (node.Id >= 0)
                    {
                        bestLength = i - start + 1;
                        bestId = node.Id;
                    }
                }

                if (bestId >= 0)
                {
                    return (start, bestLength, bestId);
                }
            }

            return (-1, 0, -1);
        }

        private sealed class Node
        {
            public Dictionary<char, Node>? Children;
            public int Id = -1;
        }
    }
}
