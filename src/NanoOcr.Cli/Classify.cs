using System.Globalization;
using NanoOcr;
using NanoOcr.Routing;

/// <summary>
/// <c>nano-ocr classify files…</c>: each page's <see cref="PageCategory"/> and the signals behind it,
/// one tab-separated line per page — what a <see cref="PageRouter{TOutput}"/> would route on.
/// </summary>
internal static class Classify
{
    private const string Usage = """
        usage: nano-ocr classify [options] inputs...

          prints, per page: file, page, category, confidence, unread share, table share, tables,
          line height (of the page), wide-line share, and the deciding rule

          --cores N        cores to use (default: all)
          --max-pages N    read at most N pages per file
          --password PW    password of encrypted PDFs
        """;

    public static int Run(string[] args)
    {
        var inputs = new List<string>();
        int cores = 0, maxPages = 0;
        string? password = null;
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{a} needs a value");
            switch (a)
            {
                case "-h" or "--help": Console.WriteLine(Usage); return 0;
                case "--cores": cores = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--max-pages": maxPages = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--password": password = Next(); break;
                default:
                    if (a.StartsWith("--", StringComparison.Ordinal))
                    {
                        Console.Error.WriteLine($"nano-ocr classify: unknown option {a}\n{Usage}");
                        return 2;
                    }

                    inputs.Add(a);
                    break;
            }
        }

        if (inputs.Count == 0)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        using var ocr = new NanoOCR(new NanoOcrOptions { Threads = cores > 0 ? cores : Environment.ProcessorCount });
        Console.WriteLine("file\tpage\tcategory\tconfidence\tunread\ttables\ttable_count\tline_height\twide\treason");
        int failed = 0;
        foreach (var (path, pages, error) in ocr.PredictMany(inputs, cores, maxPages, password))
        {
            if (error is not null)
            {
                Console.Error.WriteLine($"nano-ocr: {error.Message}");
                failed++;
                continue;
            }

            foreach (OcrResult r in pages!)
            {
                PageClassification c = PageClassifier.Classify(r);
                PageSignals s = c.Signals;
                Console.WriteLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"{path}\t{r.PageIndex + 1}\t{c.Category}\t{s.Confidence:0.000}\t{s.LowConfidenceShare:0.000}\t{s.TableShare:0.000}\t{s.Tables}\t{s.LineHeight:0.0000}\t{s.WideLineShare:0.000}\t{c.Reason}"));
            }
        }

        return failed > 0 ? 1 : 0;
    }
}
