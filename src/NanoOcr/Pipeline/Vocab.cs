using System.Text;
using System.Text.Json;

namespace NanoOcr.Pipeline;

/// <summary>
/// Token vocabulary of a CTC recognizer (port of <c>indra_ocr/vocab.py</c>, the parts inference
/// needs). Class 0 is the CTC blank; class <c>i + 1</c> is <see cref="Tokens"/>[i]. A token is one
/// character or one LaTeX command such as <c>\alpha</c>.
/// </summary>
public sealed class Vocab
{
    private readonly Dictionary<string, int> _index;

    /// <summary>Creates a vocabulary from its token list.</summary>
    public Vocab(IReadOnlyList<string> tokens)
    {
        Tokens = [.. tokens];
        _index = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < Tokens.Length; i++)
        {
            _index.TryAdd(Tokens[i], i + 1);
        }

        HasCommands = Tokens.Any(t => Py.CodePoints(t).Length > 1);
    }

    /// <summary>The tokens; class <c>i + 1</c> is <c>Tokens[i]</c>.</summary>
    public string[] Tokens { get; }

    /// <summary>Whether some tokens are multi-character LaTeX commands.</summary>
    public bool HasCommands { get; }

    /// <summary>Number of classes, blank included.</summary>
    public int ClassCount => Tokens.Length + 1;

    /// <summary>The class id of <paramref name="token"/>, or -1.</summary>
    public int IndexOf(string token) => _index.TryGetValue(token, out int i) ? i : -1;

    /// <summary>Whether the vocabulary has <paramref name="token"/>.</summary>
    public bool Contains(string token) => _index.ContainsKey(token);

    /// <summary>
    /// Text of a class-id sequence (blanks and out-of-range ids skipped). A LaTeX command followed
    /// by a letter gets a space, so <c>\alpha</c> + <c>x</c> does not read back as <c>\alphax</c>.
    /// </summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        var toks = new List<string>(ids.Count);
        foreach (int i in ids)
        {
            if (i > 0 && i <= Tokens.Length)
            {
                toks.Add(Tokens[i - 1]);
            }
        }

        if (!HasCommands)
        {
            return string.Concat(toks);
        }

        var sb = new StringBuilder();
        for (int k = 0; k < toks.Count; k++)
        {
            string t = toks[k];
            sb.Append(t);
            string next = k + 1 < toks.Count ? toks[k + 1] : "";
            int[] cp = Py.CodePoints(t);
            if (cp.Length > 2 && cp[0] == '\\' && cp.Skip(1).All(Py.IsAlpha) && next.Length > 0 && Py.IsAlpha(Py.CodePoints(next)[0]))
            {
                sb.Append(' ');
            }
        }

        return sb.ToString();
    }

    /// <summary>Parses the <c>{"blank": 0, "chars": [...]}</c> JSON stored in <c>meta.json</c>.</summary>
    public static Vocab FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        if (doc.RootElement.TryGetProperty("blank", out JsonElement blank) && blank.GetInt32() != 0)
        {
            throw new InvalidDataException("Only blank=0 vocabularies are supported.");
        }

        return new Vocab(doc.RootElement.GetProperty("chars").EnumerateArray().Select(e => e.GetString()!).ToList());
    }
}
