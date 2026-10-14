using System.Diagnostics;
using System.Globalization;
using NanoOcr.Imaging;
using NanoOcr.Onnx;
using NanoOcr.Pipeline;

/// <summary>
/// <c>nano-ocr bench page.png [--runs N] [--threads T] [--warmup W]</c>: warm runs of the OCR
/// path on one page, best of N with the spread, per stage and per ONNX operator.
/// </summary>
/// <remarks>
/// The run starts with a <see cref="MachineProfile"/> header (CPU, ISA, FMA and read ceilings,
/// jitter), so two runs are only compared when their headers agree. The timed runs follow
/// <c>--warmup</c> untimed ones (tiering, pools) and run without the operator profile, whose
/// per-node bookkeeping would otherwise be inside the figure; the profile is a separate pass.
/// </remarks>
internal static class Bench
{
    public static int Run(string[] args)
    {
        string path = args.FirstOrDefault(a => !a.StartsWith("--", StringComparison.Ordinal))
            ?? throw new ArgumentException("bench needs an image");
        int runs = Arg(args, "--runs", 7);
        int threads = Arg(args, "--threads", Environment.ProcessorCount);
        int warmup = Arg(args, "--warmup", 3);
        if (!args.Contains("--no-machine"))
        {
            Console.WriteLine(MachineProfile.Measure().Describe());
        }

        GrayImage gray = ImageIO.ReadPages(path)[0];
        long c0 = Stopwatch.GetTimestamp();
        var engine = new NanoOcrEngine(new EngineOptions { Math = Environment.GetEnvironmentVariable("BENCH_MATH") == "1" });
        double load = Stopwatch.GetElapsedTime(c0).TotalMilliseconds;
        using var _ = NanoOcr.Core.Parallelism.Use(threads);
        c0 = Stopwatch.GetTimestamp();
        engine.Process(gray); // warm-up: tiering, pools
        double first = Stopwatch.GetElapsedTime(c0).TotalMilliseconds;
        for (int i = 0; i < warmup; i++)
        {
            engine.Process(gray);
        }

        Console.WriteLine($"cold: engine {load:0} ms, first page {first:0} ms; {warmup} warm-up runs");

        var samples = new Dictionary<string, List<double>>();

        long alloc0 = GC.GetTotalAllocatedBytes(precise: true);
        int gc0 = GC.CollectionCount(0), gc2 = GC.CollectionCount(2);
        TimeSpan pause0 = GC.GetTotalPauseDuration();
        for (int r = 0; r < runs; r++)
        {
            long t0 = Stopwatch.GetTimestamp();
            Page page = engine.Process(gray);
            double wall = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            foreach (var (k, v) in page.Timings)
            {
                (samples.TryGetValue(k, out var l) ? l : samples[k] = []).Add(v);
            }

            (samples.TryGetValue("wall_ms", out var w) ? w : samples["wall_ms"] = []).Add(wall);
        }

        Console.WriteLine($"allocated {(GC.GetTotalAllocatedBytes(precise: true) - alloc0) / runs / 1048576.0:0} MiB per run; GCs gen0 {GC.CollectionCount(0) - gc0}, gen2 {GC.CollectionCount(2) - gc2}; GC pause {(GC.GetTotalPauseDuration() - pause0).TotalMilliseconds / runs:0.0} ms per run");
        Console.WriteLine($"{path}: {gray.Width}x{gray.Height}, {threads} threads, best of {runs} warm runs");
        foreach (var (k, v) in samples)
        {
            Console.WriteLine($"  {k,-10} best {v.Min(),8:0.0} ms   median {Median(v),8:0.0}   spread {(v.Max() - v.Min()) / v.Min():P0}");
        }

        var skew = new List<double>();
        var orient = new List<double>();
        for (int r = 0; r < runs; r++)
        {
            long t0 = Stopwatch.GetTimestamp();
            NanoOcrEngine.EstimateSkew(gray);
            skew.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            t0 = Stopwatch.GetTimestamp();
            engine.Orientation(gray);
            orient.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }

        var (up, _, _) = engine.Upright(gray);
        var boxes = engine.Detect(up);
        var crop = new List<double>();
        var script = new List<double>();
        var recog = new List<double>();
        for (int r = 0; r < runs; r++)
        {
            long t0 = Stopwatch.GetTimestamp();
            var crops = boxes.Select(b => LineCrops.Crop(up, b, engine.Height)).ToList();
            crop.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            t0 = Stopwatch.GetTimestamp();
            engine.HanziLines(crops);
            script.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            t0 = Stopwatch.GetTimestamp();
            engine.Recognize(up, boxes);
            recog.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }

        Console.WriteLine($"  {"crops",-10} best {crop.Min(),8:0.0} ms   ({boxes.Count} lines)");
        Console.WriteLine($"  {"script",-10} best {script.Min(),8:0.0} ms");
        Console.WriteLine($"  {"recognize",-10} best {recog.Min(),8:0.0} ms   (crops + script + recognizer)");
        Console.WriteLine($"  {"skew",-10} best {skew.Min(),8:0.0} ms   (inside orient_ms)");
        Console.WriteLine($"  {"orient",-10} best {orient.Min(),8:0.0} ms   (the model alone)");
        string[] models = (Environment.GetEnvironmentVariable("BENCH_MODELS") ?? "det,rec,order,orient,script,rec_cjk").Split(",");
        var profiles = models.ToDictionary(m => m, _ => new Dictionary<string, double>());
        foreach (string m in models)
        {
            NanoOcr.Models.ModelStore.Embedded.Session(m).Profile = profiles[m];
        }

        for (int r = 0; r < runs; r++)
        {
            engine.Process(gray);
        }

        foreach (string m in models)
        {
            NanoOcr.Models.ModelStore.Embedded.Session(m).Profile = null;
        }

        int top = int.Parse(Environment.GetEnvironmentVariable("BENCH_TOP") ?? "12", CultureInfo.InvariantCulture);
        Console.WriteLine($"  operators (profiled pass of {runs} pages, per page):");
        foreach (var (m, ops) in profiles)
        {
            double total = ops.Where(kv => !kv.Key.StartsWith('#')).Sum(kv => kv.Value);
            if (total == 0)
            {
                continue;
            }

            Console.WriteLine($"   {m}: {total / runs:0.0} ms");
            foreach (var (k, v) in ops.Where(kv => !kv.Key.StartsWith('#')).OrderByDescending(kv => kv.Value).Take(top))
            {
                Console.WriteLine($"    {k,-34} {v / runs,8:0.00} ms  {v / total,6:P1}  {ops["#" + k] / runs,6:0} calls");
            }
        }

        return 0;
    }

    private static double Median(List<double> v)
    {
        var s = v.Order().ToList();
        return s[s.Count / 2];
    }

    private static int Arg(string[] args, string name, int fallback)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? int.Parse(args[i + 1], CultureInfo.InvariantCulture) : fallback;
    }
}
