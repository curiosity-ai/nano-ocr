using IndraOcr.Vlm.Models;

namespace IndraOcr.Vlm.Common;

/// <summary>
/// Guided decoding for a JSON array of flat objects — the shape of Surya's
/// <c>LAYOUT_JSON_SCHEMA</c> and <c>TABLE_REC_JSON_SCHEMA</c> — as a character-level automaton.
/// </summary>
/// <remarks>
/// vLLM compiles those schemas with xgrammar, which emits the properties in schema order (all are
/// required and <c>additionalProperties</c> is false) and allows any JSON whitespace between
/// tokens. This automaton accepts the same language: <c>[</c>, up to <c>maxItems</c> objects
/// separated by commas, <c>]</c>; each object's fields in order, string enums, strings of four
/// 1-4 digit numbers separated by single spaces, and non-negative integers within their bounds.
/// </remarks>
public sealed class JsonArrayConstraint : IOutputConstraint
{
    private readonly Field[] _fields;
    private readonly int _maxItems;
    private readonly Segment[] _segments;
    private State _state;

    /// <summary>Kinds of field value the schemas use.</summary>
    public enum FieldKind
    {
        /// <summary>A string from a fixed set.</summary>
        Enum,

        /// <summary>A string matching <c>^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$</c>.</summary>
        Bbox,

        /// <summary>A non-negative integer up to <see cref="Field.Max"/>.</summary>
        Integer,
    }

    /// <summary>One required property.</summary>
    /// <param name="Name">Property name.</param>
    /// <param name="Kind">Value kind.</param>
    /// <param name="Values">The allowed strings for <see cref="FieldKind.Enum"/>.</param>
    /// <param name="Max">The maximum for <see cref="FieldKind.Integer"/>.</param>
    public sealed record Field(string Name, FieldKind Kind, string[]? Values = null, int Max = int.MaxValue);

    /// <summary>Creates the automaton for an array of objects with <paramref name="fields"/>.</summary>
    public JsonArrayConstraint(Field[] fields, int maxItems)
    {
        _fields = fields;
        _maxItems = maxItems;
        var segments = new List<Segment> { Segment.Literal("{"), Segment.Whitespace };
        for (int i = 0; i < fields.Length; i++)
        {
            segments.Add(Segment.Literal($"\"{fields[i].Name}\""));
            segments.Add(Segment.Whitespace);
            segments.Add(Segment.Literal(":"));
            segments.Add(Segment.Whitespace);
            segments.Add(Segment.Value(i));
            segments.Add(Segment.Whitespace);
            if (i < fields.Length - 1)
            {
                segments.Add(Segment.Literal(","));
                segments.Add(Segment.Whitespace);
            }
        }

        segments.Add(Segment.Literal("}"));
        _segments = [.. segments];
    }

    /// <summary>Surya's <c>LAYOUT_JSON_SCHEMA</c>.</summary>
    public static JsonArrayConstraint SuryaLayout() => new(
        [
            new Field("label", FieldKind.Enum, Surya.SuryaPrompts.LayoutLabels),
            new Field("bbox", FieldKind.Bbox),
            new Field("count", FieldKind.Integer, Max: 10000),
        ],
        maxItems: 200);

    /// <summary>Surya's <c>TABLE_REC_JSON_SCHEMA</c>.</summary>
    public static JsonArrayConstraint SuryaTableRec() => new(
        [
            new Field("label", FieldKind.Enum, ["Row", "Col"]),
            new Field("bbox", FieldKind.Bbox),
        ],
        maxItems: 200);

    /// <inheritdoc/>
    public bool IsComplete => _state.Phase == Phase.Done;

    /// <inheritdoc/>
    public bool Accepts(string text)
    {
        State state = _state;
        foreach (char c in text)
        {
            if (!Feed(ref state, c))
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public void Advance(string text)
    {
        foreach (char c in text)
        {
            if (!Feed(ref _state, c))
            {
                throw new InvalidOperationException($"'{text}' violates the grammar.");
            }
        }
    }

    private static bool IsWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';

    private bool Feed(ref State s, char c)
    {
        switch (s.Phase)
        {
            case Phase.Start:
                if (IsWhitespace(c))
                {
                    return true;
                }

                if (c != '[')
                {
                    return false;
                }

                s.Phase = Phase.AfterOpen;
                return true;

            case Phase.AfterOpen:
                if (IsWhitespace(c))
                {
                    return true;
                }

                if (c == ']')
                {
                    s.Phase = Phase.Done;
                    return true;
                }

                return BeginObject(ref s, c);

            case Phase.AfterItem:
                if (IsWhitespace(c))
                {
                    return true;
                }

                if (c == ']')
                {
                    s.Phase = Phase.Done;
                    return true;
                }

                if (c == ',' && s.Items < _maxItems)
                {
                    s.Phase = Phase.AfterComma;
                    return true;
                }

                return false;

            case Phase.AfterComma:
                return IsWhitespace(c) || BeginObject(ref s, c);

            case Phase.InObject:
                return FeedObject(ref s, c);

            default:
                return IsWhitespace(c);
        }
    }

    private bool BeginObject(ref State s, char c)
    {
        if (c != '{')
        {
            return false;
        }

        s.Phase = Phase.InObject;
        s.Items++;
        s.Segment = 0;
        s.Progress = 0;
        return FeedObject(ref s, c);
    }

    private bool FeedObject(ref State s, char c)
    {
        while (true)
        {
            Segment segment = _segments[s.Segment];
            switch (segment.Kind)
            {
                case SegmentKind.Whitespace:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }

                    NextSegment(ref s);
                    continue;

                case SegmentKind.Literal:
                    if (segment.Text[s.Progress] != c)
                    {
                        return false;
                    }

                    s.Progress++;
                    if (s.Progress == segment.Text.Length)
                    {
                        if (s.Segment == _segments.Length - 1)
                        {
                            s.Phase = Phase.AfterItem;
                            return true;
                        }

                        NextSegment(ref s);
                    }

                    return true;

                default:
                    Field field = _fields[segment.FieldIndex];
                    switch (FeedValue(ref s, field, c))
                    {
                        case ValueStep.Consumed:
                            return true;
                        case ValueStep.Finished:
                            NextSegment(ref s);
                            return true;
                        case ValueStep.FinishedBefore:
                            NextSegment(ref s);
                            continue;
                        default:
                            return false;
                    }
            }
        }
    }

    private static void NextSegment(ref State s)
    {
        s.Segment++;
        s.Progress = 0;
        s.Text = string.Empty;
        s.Number = 0;
        s.Digits = 0;
        s.Group = 0;
    }

    private static ValueStep FeedValue(ref State s, Field field, char c)
    {
        switch (field.Kind)
        {
            case FieldKind.Enum:
                if (s.Progress == 0)
                {
                    if (c != '"')
                    {
                        return ValueStep.Rejected;
                    }

                    s.Progress = 1;
                    return ValueStep.Consumed;
                }

                if (c == '"')
                {
                    return Array.IndexOf(field.Values!, s.Text) >= 0 ? ValueStep.Finished : ValueStep.Rejected;
                }

                string candidate = s.Text + c;
                foreach (string value in field.Values!)
                {
                    if (value.StartsWith(candidate, StringComparison.Ordinal))
                    {
                        s.Text = candidate;
                        return ValueStep.Consumed;
                    }
                }

                return ValueStep.Rejected;

            case FieldKind.Bbox:
                if (s.Progress == 0)
                {
                    if (c != '"')
                    {
                        return ValueStep.Rejected;
                    }

                    s.Progress = 1;
                    return ValueStep.Consumed;
                }

                if (char.IsAsciiDigit(c))
                {
                    if (s.Digits >= 4)
                    {
                        return ValueStep.Rejected;
                    }

                    s.Digits++;
                    return ValueStep.Consumed;
                }

                if (c == ' ' && s.Group < 3 && s.Digits > 0)
                {
                    s.Group++;
                    s.Digits = 0;
                    return ValueStep.Consumed;
                }

                return c == '"' && s.Group == 3 && s.Digits > 0 ? ValueStep.Finished : ValueStep.Rejected;

            default:
                if (char.IsAsciiDigit(c))
                {
                    if (s.Digits > 0 && s.Number == 0)
                    {
                        // No leading zeros.
                        return ValueStep.Rejected;
                    }

                    long next = (s.Number * 10) + (c - '0');
                    if (next > field.Max)
                    {
                        return ValueStep.Rejected;
                    }

                    s.Number = next;
                    s.Digits++;
                    return ValueStep.Consumed;
                }

                return s.Digits > 0 ? ValueStep.FinishedBefore : ValueStep.Rejected;
        }
    }

    private enum Phase
    {
        Start,
        AfterOpen,
        InObject,
        AfterItem,
        AfterComma,
        Done,
    }

    private enum SegmentKind
    {
        Literal,
        Whitespace,
        Value,
    }

    private enum ValueStep
    {
        Rejected,
        Consumed,
        Finished,
        FinishedBefore,
    }

    private readonly record struct Segment(SegmentKind Kind, string Text, int FieldIndex)
    {
        public static Segment Whitespace => new(SegmentKind.Whitespace, string.Empty, -1);

        public static Segment Literal(string text) => new(SegmentKind.Literal, text, -1);

        public static Segment Value(int field) => new(SegmentKind.Value, string.Empty, field);
    }

    private struct State
    {
        public Phase Phase;
        public int Items;
        public int Segment;
        public int Progress;
        public string Text;
        public long Number;
        public int Digits;
        public int Group;
    }
}
