namespace IndraOcr.Vlm.Common;

/// <summary>
/// <c>detect_repeat_token</c> (<c>chandra/model/util.py</c>, identical in
/// <c>surya/inference/util.py</c>): whether the tail of a decode is one short sequence repeated
/// more often than its length makes plausible.
/// </summary>
public static class RepeatDetector
{
    /// <summary>Whether <paramref name="text"/> ends in a runaway repetition.</summary>
    public static bool Detect(
        string text,
        int baseMaxRepeats = 4,
        int windowSize = 500,
        int cutFromEnd = 0,
        double scalingFactor = 3.0)
    {
        if (cutFromEnd > 0)
        {
            // predicted_tokens[:-cut_from_end]
            text = cutFromEnd >= text.Length ? string.Empty : text[..^cutFromEnd];
        }

        for (int length = 1; length <= windowSize / 2; length++)
        {
            int maxRepeats = (int)(baseMaxRepeats * (1 + (scalingFactor / length)));
            int position = text.Length - length;
            if (position < 0)
            {
                continue;
            }

            // candidate_seq = predicted_tokens[-seq_len:]
            ReadOnlySpan<char> candidate = text.AsSpan(position, length);
            int repeats = 0;
            while (position >= 0)
            {
                if (text.AsSpan(position, length).SequenceEqual(candidate))
                {
                    repeats++;
                    position -= length;
                }
                else
                {
                    break;
                }
            }

            if (repeats > maxRepeats)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Chandra's and Surya's retry condition: a repetition at the end, or one that ends 50
    /// characters before it.
    /// </summary>
    public static bool HasRepeat(string text) =>
        Detect(text) || (text.Length > 50 && Detect(text, cutFromEnd: 50));
}
