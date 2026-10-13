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
        var engine = new IndraEngine();
        using var _ = IndraOcr.Core.Parallelism.Use(threads);
        engine.Process(gray); // warm-up: tiering, pools
        engine.Process(gray);

        var samples = new Dictionary<string, List<double>>();
        var ops = new Dictionary<string, double>();
        foreach (string m in (Environment.GetEnvironmentVariable("BENCH_MODELS") ?? "det,rec,order,orient,script,rec_cjk").Split(","))
        {
            IndraOcr.Models.ModelStore.Embedded.Session(m).Profile = ops;
        }

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

        Console.WriteLine($"  {"skew",-10} best {skew.Min(),8:0.0} ms   (inside orient_ms)");
        Console.WriteLine($"  {"orient",-10} best {orient.Min(),8:0.0} ms   (the model alone)");
        double total = ops.Where(kv => !kv.Key.StartsWith('#')).Sum(kv => kv.Value);
        Console.WriteLine($"  operators (sum over all runs, {total / runs:0.0} ms per run):");
        foreach (var (k, v) in ops.Where(kv => !kv.Key.StartsWith('#')).OrderByDescending(kv => kv.Value).Take(15))
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
