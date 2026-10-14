using System.Globalization;
using System.Text;
using NanoOcr;
using NanoOcr.Pipeline;

// nano-ocr: PDFs or images in, markdown (or text, or JSON lines) out — the C# counterpart of
// the Python `indra-ocr` command, with the same options.
//
//   nano-ocr paper.pdf                    # markdown to stdout
//   nano-ocr *.pdf --out results/         # one .md per input
//   nano-ocr scan.png --format json       # lines with boxes, confidences, timings
return args.Length > 0 && args[0] == "bench" ? Bench.Run(args[1..]) : args.Length > 0 && args[0] == "bench-gemm" ? GemmBench.Run() : Cli.Run(args);

internal static class Cli
{
    private const string Usage = """
        usage: nano-ocr [options] inputs...

          --out DIR             write one file per input (.md, .txt or .jsonl)
          --format md|text|json output format (default: md)
          --mode auto|ocr|text  auto: text layer where it passes the checks, OCR elsewhere (default);
                                ocr: never read the text layer; text: use the text layer even when
                                the checks fail
          --model-order         order text-layer lines with the order model, not the PDF content order
          --no-image-ocr        on text-layer pages, do not OCR text inside images
          --cores N             cores to use (default: all)
          --dpi DPI             render resolution for PDFs (default: 200)
          --max-pages N         read at most N pages per file
          --password PW         password of encrypted PDFs
          --keep-furniture      keep page headers and footers in the markdown
          --math                write math as LaTeX ($...$); slower on pages with math
          --timings             print per-page source and time to stderr
          --model-dir DIR       folder with the models (default: the embedded models)
          --save-img DIR        also save each page with its blocks and lines drawn over it
          --version             print the version
        """;

    public static int Run(string[] args)
    {
        var inputs = new List<string>();
        string? outDir = null, modelDir = null, password = null, saveImg = null;
        string format = "md";
        var options = new NanoOcrOptions();
        int cores = 0, maxPages = 0;
        bool timings = false;
        try
        {
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{a} needs a value");
                switch (a)
                {
                    case "-h" or "--help":
                        Console.WriteLine(Usage);
                        return 0;
                    case "--version":
                        Console.WriteLine($"nano-ocr {typeof(NanoOCR).Assembly.GetName().Version}");
                        return 0;
                    case "--out": outDir = Next(); break;
                    case "--format": format = Next(); break;
                    case "--mode":
                        options.Mode = Next() switch
                        {
                            "auto" => ConvertMode.Auto,
                            "ocr" => ConvertMode.Ocr,
                            "text" => ConvertMode.Text,
                            var m => throw new ArgumentException($"--mode must be auto, ocr or text, not {m}"),
                        };
                        break;
                    case "--model-order": options.PdfOrder = false; break;
                    case "--no-image-ocr": options.ImageOcr = false; break;
                    case "--cores" or "--threads": cores = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--dpi": options.Dpi = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-pages": maxPages = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--password": password = Next(); break;
                    case "--keep-furniture": options.Furniture = true; break;
                    case "--math": options.Math = true; break;
                    case "--timings": timings = true; break;
                    case "--model-dir": modelDir = Next(); break;
                    case "--save-img": saveImg = Next(); options.KeepImages = true; break;
                    default:
                        if (a.StartsWith("--", StringComparison.Ordinal))
                        {
                            throw new ArgumentException($"unknown option {a}");
                        }

                        inputs.Add(a);
                        break;
                }
            }

            if (inputs.Count == 0)
            {
                throw new ArgumentException("no inputs");
            }

            if (format is not ("md" or "text" or "json"))
            {
                throw new ArgumentException("--format must be md, text or json");
            }
        }
        catch (Exception e) when (e is ArgumentException or FormatException)
        {
            Console.Error.WriteLine($"nano-ocr: {e.Message}");
            Console.Error.WriteLine(Usage);
            return 2;
        }

        options.ModelDir = modelDir;
        if (outDir is not null)
        {
            Directory.CreateDirectory(outDir);
        }

        string ext = format switch { "md" => ".md", "text" => ".txt", _ => ".jsonl" };
        int failed = 0;
        var names = new HashSet<string>(StringComparer.Ordinal);
        var stdout = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true };

        // Pages run one per core (each single-threaded) when there are enough of them, else one at a
        // time with every core on each — the same rule as Python's convert_many.
        options.Threads = cores > 0 ? cores : Environment.ProcessorCount;
        using var ocr = new NanoOCR(options);
        IEnumerable<(string Path, IReadOnlyList<OcrResult>? Results, InputException? Error)> results =
            saveImg is null
                ? ocr.PredictMany(inputs, cores, maxPages, password)
                : inputs.Select(p => Single(ocr, p, maxPages, password));
        foreach (var (path, pages, error) in results)
        {
            if (error is not null)
            {
                Console.Error.WriteLine($"nano-ocr: {error.Message}");
                failed++;
                continue;
            }

            var chunks = new List<string>();
            foreach (OcrResult r in pages!)
            {
                if (r.Page.Source == PageSources.Error)
                {
                    Console.Error.WriteLine($"nano-ocr: {path} page {r.PageIndex + 1}: {r.Page.Note}");
                }

                chunks.Add(format switch
                {
                    "json" => r.ToJson(),
                    "md" => r.Page.Markdown ?? r.Page.Text,
                    _ => r.Page.Text,
                });
                if (timings)
                {
                    Console.Error.WriteLine($"{path} page {r.PageIndex + 1}: {r.Page.Source}, {r.Page.Timings.GetValueOrDefault("total_ms"):0} ms {r.Page.Note}");
                }

                if (saveImg is not null)
                {
                    r.SaveToImg(saveImg);
                }
            }

            string body = string.Join(format == "json" ? "\n" : "\n\n", chunks) + "\n";
            if (outDir is not null)
            {
                string stem = Path.GetFileNameWithoutExtension(path);
                string name = stem;
                for (int k = 2; names.Contains(name); k++)
                {
                    name = $"{stem}_{k}";
                }

                names.Add(name);
                File.WriteAllText(Path.Combine(outDir, name + ext), body, new UTF8Encoding(false));
            }
            else
            {
                stdout.Write(body);
            }
        }

        if (failed > 0)
        {
            Console.Error.WriteLine($"nano-ocr: {failed} of {inputs.Count} inputs failed");
        }

        return failed > 0 ? 1 : 0;
    }

    private static (string, IReadOnlyList<OcrResult>?, InputException?) Single(NanoOCR ocr, string path, int maxPages, string? password)
    {
        try
        {
            return (path, ocr.Predict(path, maxPages, password).ToList(), null);
        }
        catch (InputException e)
        {
            return (path, null, e);
        }
    }
}
