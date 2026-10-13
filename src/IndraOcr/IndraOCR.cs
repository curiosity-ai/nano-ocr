using System.Collections.Concurrent;
using IndraOcr.Core;
using IndraOcr.Imaging;
using IndraOcr.Models;
using IndraOcr.Pdf;
using IndraOcr.Pipeline;
using SkiaSharp;

namespace IndraOcr;

/// <summary>Options of <see cref="IndraOCR"/> (the arguments of Python's <c>indra_ocr.load</c>).</summary>
public sealed class IndraOcrOptions
{
    /// <summary>A folder with the models; <see langword="null"/> uses the embedded models (or <c>INDRA_OCR_MODEL_DIR</c>).</summary>
    public string? ModelDir { get; set; }

    /// <summary>Threads the kernels use for one page (default: all cores).</summary>
    public int Threads { get; set; } = Environment.ProcessorCount;

    /// <summary>Render resolution for PDFs.</summary>
    public double Dpi { get; set; } = 200.0;

    /// <summary><see cref="ConvertMode.Auto"/>: text layer where sound, OCR elsewhere.</summary>
    public ConvertMode Mode { get; set; } = ConvertMode.Auto;

    /// <summary>Keep the PDF content order when it is coherent (false: the order model orders every page).</summary>
    public bool PdfOrder { get; set; } = true;

    /// <summary>On text-layer pages, OCR text inside images.</summary>
    public bool ImageOcr { get; set; } = true;

    /// <summary>Keep page headers and footers in the markdown.</summary>
    public bool Furniture { get; set; }

    /// <summary>Write math as LaTeX (<c>$…$</c>); slower on pages with math.</summary>
    public bool Math { get; set; }

    /// <summary>Turn pages upright with the orientation model (PaddleOCR's <c>use_doc_orientation_classify</c>).</summary>
    public bool UseDocOrientationClassify { get; set; } = true;

    /// <summary>Build tables from the order model's table heads instead of from geometry.</summary>
    public bool TableHeads { get; set; }

    /// <summary>Keep each page image on its result (needed by <see cref="OcrResult.SaveToImg"/>).</summary>
    public bool KeepImages { get; set; }
}

/// <summary>
/// Indra OCR for .NET, with an API shaped like PaddleOCR's:
/// <code>
/// using var ocr = new IndraOCR();
/// foreach (OcrResult res in ocr.Predict("paper.pdf"))
/// {
///     res.Print();
///     res.SaveToMarkdown("output");
///     res.SaveToJson("output");
/// }
/// </code>
/// PDFs read their text layer where it can be trusted and are OCR'd elsewhere; images are OCR'd.
/// An instance is thread-safe.
/// </summary>
public sealed class IndraOCR : IDisposable
{
    private readonly DocumentConverter _converter;

    /// <summary>Loads the models and builds the pipeline.</summary>
    public IndraOCR(IndraOcrOptions? options = null)
    {
        Options = options ?? new IndraOcrOptions();
        Engine = new IndraEngine(new EngineOptions
        {
            Models = ModelStore.FromDirectory(Options.ModelDir),
            Furniture = Options.Furniture,
            Math = Options.Math,
            TableHeads = Options.TableHeads,
        });
        _converter = new DocumentConverter(Engine, Options.Dpi, Options.Mode, Options.PdfOrder, Options.ImageOcr);
    }

    /// <summary>The options this instance was built with.</summary>
    public IndraOcrOptions Options { get; }

    /// <summary>The OCR engine underneath, for stage-level access.</summary>
    public IndraEngine Engine { get; }

    /// <summary>The document converter underneath.</summary>
    public DocumentConverter Converter => _converter;

    private Parallelism.Scope Threads() => Parallelism.Use(Options.Threads);

    /// <summary>
    /// Runs the pipeline on a file (PDF or image) or every file in a directory, one result per
    /// page, in order. Unreadable files throw <see cref="InputException"/>.
    /// </summary>
    public IEnumerable<OcrResult> Predict(string input, int maxPages = 0, string? password = null)
    {
        IEnumerable<string> files = Directory.Exists(input)
            ? Directory.EnumerateFiles(input).Order(StringComparer.Ordinal)
            : [input];
        foreach (string file in files)
        {
            foreach (OcrResult r in PredictFile(file, maxPages, password))
            {
                yield return r;
            }
        }
    }

    private IEnumerable<OcrResult> PredictFile(string path, int maxPages, string? password)
    {
        if (!File.Exists(path))
        {
            throw new InputException($"{path}: no such file");
        }

        if (!PdfDocument.IsPdf(path))
        {
            List<GrayImage> frames = ImageIO.ReadPages(path);
            int count = maxPages > 0 ? System.Math.Min(maxPages, frames.Count) : frames.Count;
            for (int i = 0; i < count; i++)
            {
                yield return PredictImage(frames[i], path, i);
            }

            yield break;
        }

        PdfDocument doc = OpenPdf(path, password);
        using (doc)
        {
            int n = maxPages > 0 ? System.Math.Min(doc.PageCount, maxPages) : doc.PageCount;
            for (int i = 0; i < n; i++)
            {
                yield return PredictPdfPage(doc, path, i);
            }
        }
    }

    private static PdfDocument OpenPdf(string path, string? password)
    {
        try
        {
            return PdfDocument.Open(path, password);
        }
        catch (PdfException e)
        {
            throw new InputException(e.NeedsPassword ? $"{path}: password-protected PDF (pass a password)" : $"{path}: {e.Message}", e);
        }
    }

    private OcrResult PredictPdfPage(PdfDocument doc, string path, int index)
    {
        using Parallelism.Scope _ = Threads();
        Page page = DocumentConverter.Safe(() => _converter.PdfPageResult(doc, index));
        GrayImage? image = null;
        if (Options.KeepImages && page.Source != PageSources.Error)
        {
            using PdfPage p = doc.GetPage(index);
            image = p.Render(_converter.PageScale(p));
        }

        return new OcrResult(path, index, page, image);
    }

    private OcrResult PredictImage(GrayImage image, string path, int index)
    {
        using Parallelism.Scope _ = Threads();
        Page page = DocumentConverter.Safe(() => Engine.Process(image, Options.UseDocOrientationClassify));
        return new OcrResult(path, index, page, Options.KeepImages ? image : null);
    }

    /// <summary>OCR of an in-memory grayscale page.</summary>
    public OcrResult Predict(GrayImage image) => PredictImage(image, "", 0);

    /// <summary>OCR of a SkiaSharp bitmap.</summary>
    public OcrResult Predict(SKBitmap bitmap) => PredictImage(ImageIO.FromBitmap(bitmap), "", 0);

    /// <summary>OCR of encoded image bytes (PNG, JPEG, …) or a PDF.</summary>
    public IEnumerable<OcrResult> Predict(byte[] encoded, string? password = null)
    {
        if (PdfDocument.IsPdf(encoded))
        {
            using PdfDocument doc = PdfDocument.Open(encoded, password);
            for (int i = 0; i < doc.PageCount; i++)
            {
                yield return PredictPdfPage(doc, "", i);
            }

            yield break;
        }

        List<GrayImage> frames = ImageIO.Decode(encoded);
        for (int i = 0; i < frames.Count; i++)
        {
            yield return PredictImage(frames[i], "", i);
        }
    }

    /// <summary>
    /// PaddleOCR 2.x's <c>ocr.ocr(path)</c>: per page, a list of (four corner points, (text, score)).
    /// </summary>
    public List<List<(float[][] Box, (string Text, double Score) Rec)>> Ocr(string path)
    {
        var pages = new List<List<(float[][], (string, double))>>();
        foreach (OcrResult r in Predict(path))
        {
            pages.Add(r.DtPolys.Zip(r.RecTexts, r.RecScores).Select(z => (z.First, (z.Second, z.Third))).ToList());
        }

        return pages;
    }

    /// <summary>
    /// Many files on many cores: every page of every input is a task, run with one kernel thread
    /// each on <paramref name="cores"/> workers (the counterpart of Python's <c>convert_many</c>
    /// with one worker process per core). Yields (path, results or the error) in input order.
    /// </summary>
    public IEnumerable<(string Path, IReadOnlyList<OcrResult>? Results, InputException? Error)> PredictMany(IEnumerable<string> paths, int cores = 0, int maxPages = 0, string? password = null)
    {
        string[] inputs = paths.ToArray();
        cores = cores > 0 ? cores : Environment.ProcessorCount;
        var plan = new List<(string Path, int Pages, bool Pdf, InputException? Error)>();
        foreach (string path in inputs)
        {
            try
            {
                if (!File.Exists(path))
                {
                    throw new InputException($"{path}: no such file");
                }

                if (PdfDocument.IsPdf(path))
                {
                    using PdfDocument doc = OpenPdf(path, password);
                    plan.Add((path, maxPages > 0 ? System.Math.Min(maxPages, doc.PageCount) : doc.PageCount, true, null));
                }
                else
                {
                    plan.Add((path, 1, false, null));
                }
            }
            catch (InputException e)
            {
                plan.Add((path, 0, false, e));
            }
        }

        var units = plan.Where(p => p.Error is null)
            .SelectMany(p => p.Pdf ? Enumerable.Range(0, p.Pages).Select(i => (p.Path, Index: i, p.Pdf)) : [(p.Path, Index: -1, p.Pdf)])
            .ToList();
        var results = new ConcurrentDictionary<(string, int), object>();
        var docs = new ConcurrentDictionary<string, Lazy<PdfDocument>>();
        var single = new ParallelOptions { MaxDegreeOfParallelism = 1 };
        Parallel.ForEach(units, new ParallelOptions { MaxDegreeOfParallelism = cores }, u =>
        {
            using Parallelism.Scope _ = Parallelism.Use(single);
            try
            {
                if (u.Pdf)
                {
                    PdfDocument doc = docs.GetOrAdd(u.Path, p => new Lazy<PdfDocument>(() => OpenPdf(p, password))).Value;
                    Page page = DocumentConverter.Safe(() => _converter.PdfPageResult(doc, u.Index));
                    results[(u.Path, u.Index)] = new List<OcrResult> { new(u.Path, u.Index, page, null) };
                }
                else
                {
                    List<GrayImage> frames = ImageIO.ReadPages(u.Path);
                    int count = maxPages > 0 ? System.Math.Min(maxPages, frames.Count) : frames.Count;
                    results[(u.Path, -1)] = Enumerable.Range(0, count)
                        .Select(i => new OcrResult(u.Path, i, DocumentConverter.Safe(() => Engine.Process(frames[i], Options.UseDocOrientationClassify)), null))
                        .ToList();
                }
            }
            catch (InputException e)
            {
                results[(u.Path, u.Index)] = e;
            }
        });
        foreach (Lazy<PdfDocument> d in docs.Values)
        {
            if (d.IsValueCreated)
            {
                d.Value.Dispose();
            }
        }

        foreach (var p in plan)
        {
            if (p.Error is not null)
            {
                yield return (p.Path, null, p.Error);
                continue;
            }

            var pages = new List<OcrResult>();
            InputException? error = null;
            foreach (int i in p.Pdf ? Enumerable.Range(0, p.Pages) : [-1])
            {
                object r = results[(p.Path, i)];
                if (r is InputException e)
                {
                    error ??= e;
                }
                else
                {
                    pages.AddRange((List<OcrResult>)r);
                }
            }

            yield return (p.Path, error is null ? pages : null, error);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }
}
