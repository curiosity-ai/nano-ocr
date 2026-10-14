using System.Diagnostics;
using NanoOcr.Core;

/// <summary><c>nano-ocr bench-gemm [--gemm-256]</c>: GEMM throughput at the shapes the models use.</summary>
internal static class GemmBench
{
    public static int Run(string[] args)
    {
        if (args.Contains("--gemm-256"))
        {
            Gemm.Wide = false;
        }

        Console.WriteLine($"Vector<float>.Count = {System.Numerics.Vector<float>.Count}, hardware accelerated = {System.Numerics.Vector.IsHardwareAccelerated}, 512-bit kernels = {Gemm.Wide}");
        foreach (var (m, n, k, serial) in new[] { (256, 134, 128, true), (512, 512, 512, true), (256, 1742, 128, true), (128, 1742, 256, true), (1678, 1742, 128, true), (32, 448, 144, true), (48, 224, 288, true), (48, 2048, 24, true), (512, 512, 512, false), (1678, 1742, 128, false) })
        {
            var a = new float[m * k];
            var b = new float[k * n];
            var c = new float[m * n];
            var rng = new Random(1);
            for (int i = 0; i < a.Length; i++) { a[i] = (float)rng.NextDouble(); }
            for (int i = 0; i < b.Length; i++) { b[i] = (float)rng.NextDouble(); }
            double best = double.MaxValue;
            int reps = Math.Max(1, (int)(2e8 / (2.0 * m * n * k)));
            for (int s = 0; s < 7; s++)
            {
                long t0 = Stopwatch.GetTimestamp();
                for (int r = 0; r < reps; r++)
                {
                    if (serial) { GemmSerial.Multiply(m, n, k, a, k, b, n, c, n); }
                    else { Gemm.Multiply(m, n, k, a, k, b, n, c, n); }
                }

                best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalSeconds / reps);
            }

            Console.WriteLine($"  {m}x{n}x{k} {(serial ? "1 thread " : "all      ")}: {2.0 * m * n * k / best / 1e9,6:0.0} GFLOP/s");
        }

        return 0;
    }
}
