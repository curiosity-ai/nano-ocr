using System.Diagnostics;
using System.Globalization;
using IndraOcr.Imaging;
using IndraOcr.Onnx;
using IndraOcr.Pipeline;

/// <summary>
/// <c>indra-ocr bench page.png [--runs N] [--threads T]</c>: warm runs of the OCR path on one
/// page, best of N with the spread, per stage and per ONNX operator.
/// </summary>
internal static class Bench
{
    public static int Run(string[] args)
    {
        string path = args.FirstOrDefault(a => !a.StartsWith("--", StringComparison.Ordinal))
            ?? throw new ArgumentException("bench needs an image");
        int runs = Arg(args, "--runs", 7);
        int threads = Arg(args, "--threads", Environment.ProcessorCount);
        GrayImage gray = ImageIO.ReadPages(path)[0];
        long c0 = Stopwatch.GetTimestamp();
        var engine = new IndraEngine();
        double load = Stopwatch.GetElapsedTime(c0).TotalMilliseconds;
        using var _ = IndraOcr.Core.Parallelism.Use(threads);
        c0 = Stopwatch.GetTimestamp();
        engine.Process(gray); // warm-up: tiering, pools
        double first = Stopwatch.GetElapsedTime(c0).TotalMilliseconds;
        engine.Process(gray);
        Console.WriteLine($"cold: engine {load:0} ms, first page {first:0} ms");

        var samples = new Dictionary<string, List<double>>();
        var ops = new Dictionary<string, double>();
        foreach (string m in (Environment.GetEnvironmentVariable("BENCH_MODELS") ?? "det,rec,order,orient,script,rec_cjk").Split(","))
        {
            IndraOcr.Models.ModelStore.Embedded.Session(m).Profile = ops;
        }

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
            IndraEngine.EstimateSkew(gray);
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
        double total = ops.Where(kv => !kv.Key.StartsWith('#')).Sum(kv => kv.Value);
        Console.WriteLine($"  operators (sum over all runs, {total / runs:0.0} ms per run):");
        foreach (var (k, v) in ops.Where(kv => !kv.Key.StartsWith('#')).OrderByDescending(kv => kv.Value).Take(int.Parse(Environment.GetEnvironmentVariable("BENCH_TOP") ?? "15")))
        {
            Console.WriteLine($"    {k,-28} {v / runs,8:0.0} ms  {v / total,6:P1}  {ops["#" + k] / runs,6:0} calls");
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
