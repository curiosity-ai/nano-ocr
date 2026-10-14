using System.Diagnostics;
using System.Globalization;
using System.Text;
using NanoOcr.Imaging;
using NanoOcr.Pdf;

namespace NanoOcr.Pipeline;

/// <summary>How a PDF page's text is obtained.</summary>
public enum ConvertMode
{
    /// <summary>The text layer where it passes the checks, OCR elsewhere.</summary>
    Auto,

    /// <summary>Always OCR; never read the text layer.</summary>
    Ocr,

    /// <summary>Use the text layer even when the checks fail.</summary>
    Text,
}

/// <summary>A file that cannot be read: not a PDF or an image, corrupt, or locked.</summary>
public sealed class InputException(string message, Exception? inner = null) : Exception(message, inner);

/// <summary>
/// Whole documents (port of <c>indra_ocr/document.py</c>): the PDF text layer where it can be
/// trusted, OCR elsewhere. Characters of the layer go to the detected line containing their
/// centre; detected lines without text are OCR'd; then the OCR path's splitting, ordering and
/// markdown run on the result.
/// </summary>
public sealed class DocumentConverter
{
    /// <summary>Resolution of the render for the text-layer ink checks.</summary>
    public const double CheckDpi = 72.0;

    /// <summary>A page larger than this at the set dpi renders at a lower dpi.</summary>
    public const double MaxPagePixels = 100e6;

    /// <summary>Share of the layer's letters the recognizer must know for it to be "readable".</summary>
    public const double ReadableMin = 0.9;

    /// <summary>Creates a converter over an engine.</summary>
    public DocumentConverter(NanoOcrEngine engine, double dpi = 200.0, ConvertMode mode = ConvertMode.Auto, bool pdfOrder = true, bool imageOcr = true)
    {
        Engine = engine;
        Dpi = dpi;
        Mode = mode;
        PdfOrder = pdfOrder;
        ImageOcr = imageOcr;
    }

    /// <summary>The OCR engine.</summary>
    public NanoOcrEngine Engine { get; }

    /// <summary>Render resolution for PDFs.</summary>
    public double Dpi { get; }

    /// <summary>Text layer or OCR.</summary>
    public ConvertMode Mode { get; }

    /// <summary>Keep the PDF content order when it is coherent.</summary>
    public bool PdfOrder { get; }

    /// <summary>OCR detected lines that hold no text-layer characters.</summary>
    public bool ImageOcr { get; }

    /// <summary><c>readable</c>: whether the recognizer can read the script of these runs.</summary>
    public bool Readable(IEnumerable<TextRun> runs)
    {
        var letters = runs.SelectMany(r => r.Text.EnumerateRunes()).Where(r => Py.IsAlpha(r.Value)).Select(r => r.ToString()).ToList();
        if (letters.Count == 0)
        {
            return true;
        }

        int known = letters.Count(c => Engine.Vocab.Contains(c) || (Engine.VocabCjk?.Contains(c) ?? false));
        return known >= ReadableMin * letters.Count;
    }

    /// <summary><c>right_to_left</c>: most strong characters are right-to-left.</summary>
    public static bool RightToLeft(IEnumerable<Token> tokens)
    {
        int rtl = 0, ltr = 0;
        foreach (Token t in tokens)
        {
            if (t.Text is null || Py.CodePoints(t.Text).Length != 1)
            {
                continue;
            }

            int c = Py.CodePoints(t.Text)[0];
            switch (Bidi.Class(c))
            {
                case Bidi.Strong.Rtl: rtl++; break;
                case Bidi.Strong.Ltr: ltr++; break;
            }
        }

        return rtl > ltr;
    }

    /// <summary>
    /// <c>assign_runs</c>: tokens (char, x, content index) per detected line. Each character goes
    /// to the smallest box containing its centre (a space follows the character before it); inside
    /// a line each run's part keeps the PDF order, and parts are ordered by x (right to left in a
    /// right-to-left line). Returns the tokens per line and the characters outside every box.
    /// </summary>
    public static (List<List<Token>> Tokens, int Lost) AssignRuns(IReadOnlyList<TextRun> runs, IReadOnlyList<LineBox> boxes, double k)
    {
        int lost = 0;
        if (boxes.Count == 0)
        {
            return ([], runs.Sum(r => r.Chars.Count(c => !TextLayer.IsSpace(c.Char))));
        }

        var parts = boxes.Select(_ => new Dictionary<int, List<Token>>()).ToList();
        var partOrder = boxes.Select(_ => new List<int>()).ToList();
        var area = boxes.Select(b => (b.X1 - b.X0) * (b.Y1 - b.Y0)).ToArray();
        for (int idx = 0; idx < runs.Count; idx++)
        {
            int? j = null;
            foreach (var (c, b) in runs[idx].Chars)
            {
                if (!TextLayer.IsSpace(c))
                {
                    double cx = (b[0] + b[2]) / 2 * k, cy = (b[1] + b[3]) / 2 * k;

                    // numpy compares the float32 boxes with the centre narrowed to float32.
                    float fx = (float)cx, fy = (float)cy;
                    int best = -1;
                    for (int i = 0; i < boxes.Count; i++)
                    {
                        LineBox bx = boxes[i];
                        if (bx.X0 <= fx && fx <= bx.X1 && bx.Y0 <= fy && fy <= bx.Y1 && (best < 0 || area[i] < area[best]))
                        {
                            best = i;
                        }
                    }

                    if (best < 0)
                    {
                        lost++;
                        j = null;
                        continue;
                    }

                    j = best;
                    if (!parts[best].TryGetValue(idx, out List<Token>? list))
                    {
                        parts[best][idx] = list = [];
                        partOrder[best].Add(idx);
                    }

                    list.Add(new Token(-1, c, cx, idx));
                }
                else if (j is int jj)
                {
                    List<Token> list = parts[jj][idx];
                    list.Add(new Token(-1, c, list[^1].X, idx));
                }
            }
        }

        var toks = new List<List<Token>>(boxes.Count);
        for (int i = 0; i < boxes.Count; i++)
        {
            Dictionary<int, List<Token>> line = parts[i];
            var tk = new List<Token>();
            bool rtl = RightToLeft(line.Values.SelectMany(p => p));
            IEnumerable<int> order = rtl
                ? partOrder[i].OrderBy(r => -line[r].Max(t => t.X))
                : partOrder[i].OrderBy(r => line[r].Min(t => t.X));
            foreach (int r in order)
            {
                if (tk.Count > 0)
                {
                    tk.Add(new Token(-1, " ", tk[^1].X, r));
                }

                tk.AddRange(line[r]);
            }

            toks.Add(tk);
        }

        return (toks, lost);
    }

    /// <summary><c>page_scale</c>: pixels per point — the set dpi, lowered for very large pages.</summary>
    public double PageScale(PdfPage page)
    {
        double w = page.Width, h = page.Height;
        double k = Dpi / 72.0;
        return Math.Min(k, Math.Pow(MaxPagePixels / Math.Max(w * h, 1.0), 0.5));
    }

    /// <summary><c>pdf_page</c>: one PDF page through the text layer or OCR.</summary>
    public Page PdfPageResult(PdfDocument doc, int index)
    {
        using PdfPage page = doc.GetPage(index);
        var t = new Dictionary<string, double>();
        string note = "mode=ocr";
        bool turn = true;
        if (Mode != ConvertMode.Ocr)
        {
            long t0 = Stopwatch.GetTimestamp();
            double s = CheckDpi / 72.0;
            TextLayerResult tl = TextLayer.Read(page, page.Render(s), s);
            t["text_ms"] = NanoOcrEngine.Ms(t0);
            bool unreadable = tl.Reason != TextLayer.BrokenFontMap && !Readable(tl.Lines);
            if (tl.Lines.Count > 0 && (tl.Trusted || Mode == ConvertMode.Text || unreadable))
            {
                return TextPage(page, tl, t);
            }

            note = tl.Reason;
            Dictionary<string, double> st = tl.Stats;
            turn = !(tl.Reason == TextLayer.BrokenFontMap && st["rotated"] < 0.5 * st["chars"]);
        }

        long tr = Stopwatch.GetTimestamp();
        GrayImage gray = page.Render(PageScale(page));
        t["render_ms"] = NanoOcrEngine.Ms(tr);
        Page result = Engine.Process(gray, turn);
        var merged = new Dictionary<string, double>(t);
        foreach (var (key, v) in result.Timings)
        {
            merged[key] = v;
        }

        merged["total_ms"] = merged.Where(kv => kv.Key != "total_ms").Sum(kv => kv.Value);
        result.Timings = merged;
        result.Note = $"{note}; {result.Note}";
        return result;
    }

    private Page TextPage(PdfPage page, TextLayerResult tl, Dictionary<string, double> t)
    {
        double k = PageScale(page);
        long t0 = Stopwatch.GetTimestamp();
        GrayImage gray = page.Render(k);
        t["render_ms"] = NanoOcrEngine.Ms(t0);
        t0 = Stopwatch.GetTimestamp();
        List<LineBox> boxes = Engine.Detect(gray);
        t["det_ms"] = NanoOcrEngine.Ms(t0);
        (List<List<Token>> toks, int lost) = AssignRuns(tl.Lines, boxes, k);
        var rec = new LineReading?[boxes.Count];
        var empty = new List<int>();
        for (int i = 0; i < toks.Count; i++)
        {
            string text = toks[i].Count > 0 ? Engine.DecodeTokens(toks[i]) : "";
            if (text.Length > 0)
            {
                rec[i] = new LineReading(text, 1.0, toks[i]);
            }
            else
            {
                empty.Add(i);
            }
        }

        string source = PageSources.Text;
        if (empty.Count > 0 && ImageOcr && Readable(tl.Lines))
        {
            t0 = Stopwatch.GetTimestamp();
            LineReading[] ocr = Engine.Recognize(gray, empty.Select(i => boxes[i]).ToList());
            for (int e = 0; e < empty.Count; e++)
            {
                if (Py.Strip(ocr[e].Text).Length > 0)
                {
                    rec[empty[e]] = ocr[e];
                    source = PageSources.Hybrid;
                }
            }

            t["rec_ms"] = NanoOcrEngine.Ms(t0);
        }

        if (Engine.MathReader is not null)
        {
            t0 = Stopwatch.GetTimestamp();
            ReadMath(gray, boxes, rec);
            t["math_ms"] = NanoOcrEngine.Ms(t0);
        }

        var keep = Enumerable.Range(0, rec.Length).Where(i => rec[i] is not null).ToList();
        Page result = Engine.Assemble(gray.Width, gray.Height, keep.Select(i => boxes[i]).ToList(), keep.Select(i => rec[i]!).ToList(), t, contentOrder: PdfOrder);
        result.Source = source;
        if (lost > 0)
        {
            result.Note += $"; {lost} chars outside detected lines";
        }

        return result;
    }

    /// <summary>
    /// <c>_read_math</c>: text-layer lines that hold math are read by the math reader instead (the
    /// layer has the symbols but not the LaTeX), keeping their place in the content order.
    /// </summary>
    private void ReadMath(GrayImage gray, List<LineBox> boxes, LineReading?[] rec)
    {
        var idx = Enumerable.Range(0, rec.Length)
            .Where(i => rec[i] is { } r && r.Tokens.Count > 0 && r.Tokens[0].IsText && r.Tokens[0].HasContentIndex)
            .ToList();
        if (idx.Count == 0)
        {
            return;
        }

        var crops = idx.Select(i => LineCrops.Crop(gray, boxes[i], Engine.Height)).ToList();
        bool[] flags = Engine.MathReader!.Flags(crops);
        var math = idx.Where((_, k) => flags[k]).ToList();
        if (math.Count == 0)
        {
            return;
        }

        var readings = Engine.MathReader.Read(gray, math.Select(i => boxes[i]).ToList());
        for (int m = 0; m < math.Count; m++)
        {
            (string text, double conf) = readings[m];
            if (text.Contains('$', StringComparison.Ordinal) && conf >= MathReader.MathMinConf)
            {
                int i = math[m];
                int first = rec[i]!.Tokens.Min(tk => tk.ContentIndex);
                double xc = (double)(boxes[i].X0 + boxes[i].X1) / 2;
                rec[i] = new LineReading(text, 1.0, [new Token(-1, text, xc, first)]);
            }
        }
    }

    /// <summary>
    /// <c>convert</c>: (page index, page) for a PDF or an image file — the file's bytes decide
    /// which. A page that fails yields a page whose <see cref="Page.Source"/> is <c>error</c>; a
    /// file that cannot be opened throws <see cref="InputException"/>.
    /// </summary>
    public IEnumerable<(int Index, Page Page)> Convert(string path, int maxPages = 0, string? password = null)
    {
        if (!File.Exists(path))
        {
            throw new InputException($"{path}: no such file");
        }

        if (!PdfDocument.IsPdf(path))
        {
            List<GrayImage> frames = ImageIO.ReadPages(path);
            int count = maxPages > 0 ? Math.Min(maxPages, frames.Count) : frames.Count;
            for (int i = 0; i < count; i++)
            {
                GrayImage frame = frames[i];
                yield return (i, Safe(() => Engine.Process(frame)));
            }

            yield break;
        }

        PdfDocument doc;
        try
        {
            doc = PdfDocument.Open(path, password);
        }
        catch (PdfException e)
        {
            throw new InputException(e.NeedsPassword
                ? $"{path}: password-protected PDF (pass a password)"
                : $"{path}: {e.Message}", e);
        }

        using (doc)
        {
            int n = maxPages > 0 ? Math.Min(doc.PageCount, maxPages) : doc.PageCount;
            for (int i = 0; i < n; i++)
            {
                int index = i;
                yield return (i, Safe(() => PdfPageResult(doc, index)));
            }
        }
    }

    /// <summary>One bad page must not end the document: failures become an error page.</summary>
    public static Page Safe(Func<Page> read)
    {
        try
        {
            return read();
        }
        catch (Exception e) when (e is not OperationCanceledException and not OutOfMemoryException)
        {
            return new Page
            {
                Timings = new Dictionary<string, double> { ["total_ms"] = 0.0 },
                Markdown = "",
                Source = PageSources.Error,
                Note = $"{e.GetType().Name}: {e.Message}",
            };
        }
    }
}

/// <summary>Unicode bidirectional classes, enough to tell strong left-to-right from right-to-left.</summary>
internal static class Bidi
{
    internal enum Strong { None, Ltr, Rtl }

    /// <summary>
    /// The strong direction of a code point: Hebrew, Arabic, Syriac, Thaana, NKo, Samaritan,
    /// Mandaic and their presentation forms are right-to-left; letters elsewhere are left-to-right.
    /// </summary>
    public static Strong Class(int c)
    {
        if ((c >= 0x0590 && c <= 0x08FF) || (c >= 0xFB1D && c <= 0xFDFF) || (c >= 0xFE70 && c <= 0xFEFF)
            || (c >= 0x10800 && c <= 0x10FFF) || (c >= 0x1E800 && c <= 0x1EFFF))
        {
            // Arabic-Indic digits and some marks are weak, not strong.
            if ((c >= 0x0660 && c <= 0x0669) || (c >= 0x06F0 && c <= 0x06F9) || IsMark(c))
            {
                return Strong.None;
            }

            return Strong.Rtl;
        }

        if (!Rune.IsValid(c))
        {
            return Strong.None;
        }

        var r = new Rune(c);
        UnicodeCategory cat = Rune.GetUnicodeCategory(r);
        return cat is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter or UnicodeCategory.SpacingCombiningMark
            ? Strong.Ltr
            : Strong.None;
    }

    private static bool IsMark(int c) => Rune.IsValid(c)
        && Rune.GetUnicodeCategory(new Rune(c)) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark;
}
