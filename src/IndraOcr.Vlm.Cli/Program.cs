using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using IndraOcr.Vlm.Chandra;
using IndraOcr.Vlm.Imaging;
using IndraOcr.Vlm.Models;
using IndraOcr.Vlm.Surya;

// indra-vlm: Chandra 2 and Surya 2 in pure C#. The commands mirror the Python CLIs:
//
//   indra-vlm chandra paper.pdf out/                 # chandra INPUT OUTPUT
//   indra-vlm surya ocr scan.png --output-dir out/   # surya_ocr
//   indra-vlm surya layout page.pdf                  # surya_layout
//   indra-vlm surya table page.pdf --mode full       # surya_table
//   indra-vlm generate --model DIR --image x.png --prompt "OCR this block image to HTML."
return Cli.Run(args);

internal static class Cli
{
    private const string Usage = """
        usage:
          indra-vlm chandra INPUT OUTPUT [options]
              --model DIR                  checkpoint folder (default: CHANDRA_MODEL_DIR or download to the cache)
              --page-range R               PDF pages, e.g. "1-5,7" (0-based, as upstream)
              --max-output-tokens N        token budget per page (default 12384)
              --prompt-type ocr_layout|ocr (default ocr_layout)
              --max-retries N              re-decode runaway repetitions with sampling (default 0)
              --no-images                  drop figures and do not save their crops
              --include-headers-footers    keep page headers and footers
              --no-html                    do not write the .html file
              --paginate-output            separate pages in the merged output

          indra-vlm surya ocr|layout|table INPUT [options]
              --model DIR                  checkpoint folder (default: SURYA_MODEL_DIR or download to the cache)
              --output-dir DIR             default results/surya
              --page-range R               PDF pages, e.g. "0,5-10"
              --mode simple|full           table recognition mode (default simple)
              --skip-table-detection       the inputs are table crops already

          indra-vlm generate --model DIR --image FILE --prompt TEXT [--max-tokens N]

          common: --threads N   cores to use (default: all)
        """;

    public static int Run(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            Console.WriteLine(Usage);
            return args.Length == 0 ? 1 : 0;
        }

        try
        {
            var options = Options.Parse(args[1..]);
            if (options.Threads > 0)
            {
                IndraOcr.Vlm.Core.Parallelism.Default = new ParallelOptions { MaxDegreeOfParallelism = options.Threads };
            }

            return args[0] switch
            {
                "chandra" => RunChandra(options),
                "surya" => RunSurya(options),
                "generate" => RunGenerate(options),
                _ => throw new ArgumentException($"unknown command '{args[0]}'"),
            };
        }
        catch (ArgumentException e)
        {
            Console.Error.WriteLine($"indra-vlm: {e.Message}");
            Console.Error.WriteLine(Usage);
            return 2;
        }
    }

    private static int RunChandra(Options o)
    {
        if (o.Positional.Count != 2)
        {
            throw new ArgumentException("chandra needs INPUT and OUTPUT");
        }

        string input = o.Positional[0];
        string output = o.Positional[1];
        var settings = new ChandraOptions
        {
            PromptType = o.Get("--prompt-type") ?? "ocr_layout",
            MaxOutputTokens = o.GetInt("--max-output-tokens") ?? 12384,
            MaxRetries = o.GetInt("--max-retries") ?? 0,
            IncludeImages = !o.Has("--no-images"),
            IncludeHeadersFooters = o.Has("--include-headers-footers"),
        };

        string[] extensions = [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"];
        List<string> files = File.Exists(input)
            ? extensions.Contains(Path.GetExtension(input).ToLowerInvariant())
                ? [input]
                : throw new ArgumentException($"Unsupported file type: {Path.GetExtension(input)}")
            : Directory.Exists(input)
                ? Directory.EnumerateFiles(input).Where(f => extensions.Contains(Path.GetExtension(f).ToLowerInvariant())).Order(StringComparer.Ordinal).ToList()
                : throw new ArgumentException($"Path does not exist: {input}");

        Directory.CreateDirectory(output);
        Console.WriteLine("Loading model...");
        using ChandraOcr chandra = ChandraOcr.Load(o.Get("--model"), settings, Progress);
        IReadOnlyCollection<int>? pages = o.Get("--page-range") is { } range ? ChandraOcr.ParseRange(range) : null;
        for (int i = 0; i < files.Count; i++)
        {
            string file = files[i];
            Console.WriteLine($"\n[{i + 1}/{files.Count}] Processing: {Path.GetFileName(file)}");
            try
            {
                var timer = Stopwatch.StartNew();
                List<ChandraPage> results = chandra.PredictFile(file, pages, (page, total) =>
                    Console.WriteLine($"  page {page + 1}/{total}"));
                string markdown = ChandraOcr.SaveMergedOutput(
                    output, Path.GetFileName(file), results, settings.IncludeImages, !o.Has("--no-html"), o.Has("--paginate-output"));
                Console.WriteLine($"  Saved: {markdown} ({results.Count} page(s), {timer.Elapsed.TotalSeconds:F1} s)");
            }
            catch (Exception e) when (e is IOException or InvalidDataException or ArgumentException)
            {
                Console.Error.WriteLine($"  Error processing {Path.GetFileName(file)}: {e.Message}");
            }
        }

        return 0;
    }

    private static int RunSurya(Options o)
    {
        if (o.Positional.Count != 2)
        {
            throw new ArgumentException("surya needs a task (ocr, layout, table) and INPUT");
        }

        string task = o.Positional[0];
        string input = o.Positional[1];
        List<int>? pages = o.Get("--page-range") is { } range ? SuryaInput.ParseRange(range) : null;
        bool folder = Directory.Exists(input);
        (List<RgbImage> images, List<string> names) = folder
            ? SuryaInput.LoadFolder(input, pages)
            : SuryaInput.LoadFile(input, pages);
        List<RgbImage>? highres = task is "ocr" or "table"
            ? (folder ? SuryaInput.LoadFolder(input, pages, SuryaInput.ImageDpiHighRes) : SuryaInput.LoadFile(input, pages, SuryaInput.ImageDpiHighRes)).Images
            : null;

        string resultPath = Path.GetFullPath(Path.Combine(
            o.Get("--output-dir") ?? Path.Combine("results", "surya"),
            folder ? Path.GetFileName(Path.TrimEndingDirectorySeparator(input)) : Path.GetFileName(input).Split('.')[0]));
        Directory.CreateDirectory(resultPath);

        using SuryaInference inference = SuryaInference.Load(o.Get("--model"), progress: Progress);
        var byName = new Dictionary<string, JsonArray>(StringComparer.Ordinal);
        void Add(string name, JsonNode node)
        {
            if (!byName.TryGetValue(name, out JsonArray? list))
            {
                byName[name] = list = [];
            }

            node["page"] = list.Count + 1;
            list.Add(node);
        }

        var timer = Stopwatch.StartNew();
        switch (task)
        {
            case "ocr":
                List<PageOcrResult> ocr = new SuryaRecognitionPredictor(inference).Predict(highres!, fullPage: true);
                for (int i = 0; i < ocr.Count; i++)
                {
                    Add(names[i], JsonSerializer.SerializeToNode(ocr[i], SuryaJson.Default.PageOcrResult)!);
                }

                break;
            case "layout":
                List<LayoutResult> layout = new SuryaLayoutPredictor(inference).Predict(images);
                for (int i = 0; i < layout.Count; i++)
                {
                    Add(names[i], JsonSerializer.SerializeToNode(layout[i], SuryaJson.Default.LayoutResult)!);
                }

                break;
            case "table":
                RunTables(o, inference, images, highres!, names, byName);
                break;
            default:
                throw new ArgumentException($"unknown surya task '{task}'");
        }

        var root = new JsonObject();
        foreach ((string name, JsonArray list) in byName)
        {
            root[name] = list;
        }

        File.WriteAllText(Path.Combine(resultPath, "results.json"), root.ToJsonString());
        Console.WriteLine($"Wrote results to {resultPath} ({timer.Elapsed.TotalSeconds:F1} s)");
        return 0;
    }

    private static void RunTables(
        Options o,
        SuryaInference inference,
        List<RgbImage> images,
        List<RgbImage> highres,
        List<string> names,
        Dictionary<string, JsonArray> byName)
    {
        string mode = o.Get("--mode") ?? "simple";
        var crops = new List<RgbImage>();
        var counts = new List<int>();
        var owners = new List<int>();
        if (o.Has("--skip-table-detection"))
        {
            crops.AddRange(highres);
            counts.AddRange(highres.Select(_ => 0));
            owners.AddRange(Enumerable.Range(0, highres.Count));
        }
        else
        {
            List<LayoutResult> layouts = new SuryaLayoutPredictor(inference).Predict(images, highres.Select(h => (h.Width, h.Height)).ToList());
            for (int page = 0; page < layouts.Count; page++)
            {
                foreach (LayoutBox box in layouts[page].Bboxes.Where(b => b.Label is "Table" or "TableOfContents"))
                {
                    crops.Add(SuryaInput.Crop(highres[page], SuryaInput.ExpandBbox(box.Bbox)));
                    counts.Add(box.Count);
                    owners.Add(page);
                }
            }
        }

        List<TableResult> tables = new SuryaTableRecPredictor(inference).Predict(crops, mode, counts);
        var perPage = new Dictionary<int, int>();
        for (int i = 0; i < tables.Count; i++)
        {
            int page = owners[i];
            JsonNode node = JsonSerializer.SerializeToNode(tables[i], SuryaJson.Default.TableResult)!;
            node["table_idx"] = perPage[page] = perPage.GetValueOrDefault(page, -1) + 1;
            string name = names[page];
            if (!byName.TryGetValue(name, out JsonArray? list))
            {
                byName[name] = list = [];
            }

            node["page"] = names.Take(page + 1).Count(n => n == name);
            list.Add(node);
        }
    }

    private static int RunGenerate(Options o)
    {
        string model = o.Get("--model") ?? throw new ArgumentException("--model is required");
        string image = o.Get("--image") ?? throw new ArgumentException("--image is required");
        string prompt = o.Get("--prompt") ?? throw new ArgumentException("--prompt is required");
        using Qwen35Vlm vlm = Qwen35Vlm.Load(model);
        using RgbImage input = RgbImageIO.Read(image);
        using RgbImage scaled = ScaleToFit.Apply(input);
        var profile = new StageProfile();
        GenerationOutput output = vlm.Generate(scaled, prompt, new GenerationOptions { MaxNewTokens = o.GetInt("--max-tokens") ?? 1024 }, profile: profile);
        Console.WriteLine(output.Text);
        Console.Error.WriteLine(string.Join(", ", output.Timings.Select(t => $"{t.Key} {t.Value:F0}")) + $", tokens {output.TokenIds.Count}");
        Console.Error.WriteLine(profile.ToString());
        return 0;
    }

    private static void Progress(string file, long done, long total)
    {
        if (total > 0 && (done == total || done % (64 << 20) < (1 << 20)))
        {
            Console.Error.Write($"\r  downloading {file}: {done >> 20} / {total >> 20} MiB   ");
            if (done == total)
            {
                Console.Error.WriteLine();
            }
        }
    }

    private sealed class Options
    {
        private readonly Dictionary<string, string?> _named = new(StringComparer.Ordinal);

        public List<string> Positional { get; } = [];

        public int Threads => GetInt("--threads") ?? 0;

        private static readonly HashSet<string> Flags =
        [
            "--no-images", "--include-headers-footers", "--no-html", "--paginate-output", "--skip-table-detection",
        ];

        public static Options Parse(string[] args)
        {
            var o = new Options();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i].Replace('_', '-');
                if (a.StartsWith("--", StringComparison.Ordinal))
                {
                    if (Flags.Contains(a))
                    {
                        o._named[a] = null;
                    }
                    else
                    {
                        o._named[a] = i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{a} needs a value");
                    }
                }
                else
                {
                    o.Positional.Add(args[i]);
                }
            }

            return o;
        }

        public bool Has(string name) => _named.ContainsKey(name);

        public string? Get(string name) => _named.TryGetValue(name, out string? v) ? v : null;

        public int? GetInt(string name) => Get(name) is { } v ? int.Parse(v) : null;
    }
}

[JsonSerializable(typeof(PageOcrResult))]
[JsonSerializable(typeof(LayoutResult))]
[JsonSerializable(typeof(TableResult))]
internal sealed partial class SuryaJson : JsonSerializerContext;
