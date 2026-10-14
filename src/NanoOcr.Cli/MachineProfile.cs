using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// What this machine can do, measured before any model is loaded, so that a benchmark run carries
/// its own ceilings: the CPU model and ISA, the single-thread FMA rate at 256 and 512 bits, the
/// all-thread rate, the read bandwidth from L1, L2 and memory, and the jitter between samples.
/// </summary>
/// <remarks>
/// A shared VM is not a constant: the host's clock and AVX-512 licence state move, a noisy
/// neighbour halves the memory bandwidth, and the VM may land on a different CPU model from one
/// session to the next. Two runs whose headers differ are not comparable; within one run, a stage
/// figure is read against the ceilings printed above it. The method follows the sibling port's
/// <c>MachineProfile</c> (see its <c>measuring-performance</c> skill): eight differently seeded
/// accumulator chains in named locals, results consumed through <see cref="GC.KeepAlive"/>, and
/// <see cref="MethodImplOptions.AggressiveOptimization"/> on every measured loop.
/// </remarks>
internal sealed record MachineProfile(
    string Cpu,
    string Isa,
    int Threads,
    double Fma256,
    double Fma512,
    double FmaAll,
    double Jitter,
    double L1,
    double L2,
    double Dram,
    double CalibrationMs)
{
    public static MachineProfile Measure(double seconds = 1.2)
    {
        long t0 = Stopwatch.GetTimestamp();
        double slice = seconds / 6;
        bool has512 = Avx512F.IsSupported;
        (double f256, double j256) = FmaRate(Fma256Loop, slice);
        (double f512, double j512) = has512 ? FmaRate(Fma512Loop, slice) : (0, 0);
        bool wide = has512 && f512 > f256;
        double all = FmaAllThreads(wide ? Fma512Loop : Fma256Loop, wide ? 16 : 8, slice);
        return new MachineProfile(
            CpuName(),
            IsaSummary(),
            Environment.ProcessorCount,
            f256,
            f512,
            all,
            Math.Max(j256, j512),
            ReadRate(16 * 1024, slice),
            ReadRate(512 * 1024, slice),
            ReadRate(128 * 1024 * 1024, slice),
            Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
    }

    public string Describe()
    {
        var s = new System.Text.StringBuilder();
        s.AppendLine($"machine: {Cpu}, {Threads} threads, {Environment.OSVersion.VersionString}, .NET {Environment.Version}");
        s.AppendLine($"  isa      {Isa}; Vector<float> {Vector<float>.Count * 32} bits, Vector512 accelerated {Vector512.IsHardwareAccelerated}");
        s.AppendLine($"  fma      1 thread {Fma256,6:0.0} GF/s @256  {Fma512,6:0.0} GF/s @512   all threads {FmaAll,6:0.0} GF/s");
        s.AppendLine($"  read     L1 {L1,6:0.0}  L2 {L2,6:0.0}  DRAM {Dram,5:0.0} GB/s (1 thread)");
        s.Append($"  jitter   {Jitter * 100:0.0}% across FMA samples ({CalibrationMs:0} ms calibration)");
        if (Jitter > 0.05)
        {
            s.Append(" — the machine is busy; read stage times as upper bounds");
        }

        return s.ToString();
    }

    private static string CpuName()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/cpuinfo"))
            {
                if (line.StartsWith("model name", StringComparison.Ordinal))
                {
                    return line[(line.IndexOf(':') + 1)..].Trim();
                }
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }

        return System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
    }

    private static string IsaSummary()
    {
        var parts = new List<string>();
        if (Avx2.IsSupported)
        {
            parts.Add("AVX2");
        }

        if (Fma.IsSupported)
        {
            parts.Add("FMA");
        }

        if (Avx512F.IsSupported)
        {
            parts.Add("AVX-512F");
        }

        if (Avx512BW.IsSupported)
        {
            parts.Add("BW");
        }

        if (Avx10v1.IsSupported)
        {
            parts.Add("AVX10.1");
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported)
        {
            parts.Add("NEON");
        }

        return parts.Count == 0 ? "scalar" : string.Join(" ", parts);
    }

    private static (double Best, double Jitter) FmaRate(Func<int, double> body, double seconds)
    {
        const int Samples = 5;
        int iterations = Calibrate(body, seconds / Samples);
        double best = 0, worst = double.MaxValue;
        for (int i = 0; i < Samples; i++)
        {
            double rate = body(iterations);
            best = Math.Max(best, rate);
            worst = Math.Min(worst, rate);
        }

        return (best, best > 0 ? (best - worst) / best : 0);
    }

    private static int Calibrate(Func<int, double> body, double seconds)
    {
        int iterations = 1 << 12;
        for (int attempt = 0; attempt < 24; attempt++)
        {
            long t0 = Stopwatch.GetTimestamp();
            body(iterations);
            double elapsed = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            if (elapsed >= seconds)
            {
                return iterations;
            }

            double factor = elapsed > 1e-6 ? Math.Min(16, seconds / elapsed) : 16;
            iterations = (int)Math.Min(int.MaxValue / 2, iterations * Math.Max(2, factor));
        }

        return iterations;
    }

    private static double FmaAllThreads(Func<int, double> body, int lanes, double seconds)
    {
        int iterations = Calibrate(body, seconds);
        int threads = Environment.ProcessorCount;
        long t0 = Stopwatch.GetTimestamp();
        Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, _ => body(iterations));
        double elapsed = Stopwatch.GetElapsedTime(t0).TotalSeconds;

        // Work over wall clock, not the sum of per-thread rates: a descheduled thread must count.
        return elapsed > 0 ? threads * FlopsPer(iterations, lanes) / elapsed / 1e9 : 0;
    }

    private static double FlopsPer(int iterations, int lanes) => (double)iterations * 8 * lanes * 2;

    /// <summary>Times <see cref="Fma512Chains"/>; the timer calls stay outside the loop's method.</summary>
    /// <remarks>
    /// A call inside the method that holds the chains makes the JIT keep all eight accumulators in
    /// the frame across it, and the loop then measures store forwarding (about half the real rate).
    /// </remarks>
    private static double Fma512Loop(int iterations)
    {
        long t0 = Stopwatch.GetTimestamp();
        GC.KeepAlive(Fma512Chains(iterations));
        double seconds = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        return seconds > 0 ? FlopsPer(iterations, 16) / seconds / 1e9 : 0;
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Fma512Chains(int iterations)
    {
        Vector512<float> x = Vector512.Create(1.0000001f), y = Vector512.Create(0.9999999f);
        Vector512<float> a0 = Vector512.Create(1f), a1 = Vector512.Create(2f), a2 = Vector512.Create(3f), a3 = Vector512.Create(4f);
        Vector512<float> a4 = Vector512.Create(5f), a5 = Vector512.Create(6f), a6 = Vector512.Create(7f), a7 = Vector512.Create(8f);
        for (int i = 0; i < iterations; i++)
        {
            a0 = Vector512.FusedMultiplyAdd(x, y, a0);
            a1 = Vector512.FusedMultiplyAdd(x, y, a1);
            a2 = Vector512.FusedMultiplyAdd(x, y, a2);
            a3 = Vector512.FusedMultiplyAdd(x, y, a3);
            a4 = Vector512.FusedMultiplyAdd(x, y, a4);
            a5 = Vector512.FusedMultiplyAdd(x, y, a5);
            a6 = Vector512.FusedMultiplyAdd(x, y, a6);
            a7 = Vector512.FusedMultiplyAdd(x, y, a7);
        }

        return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
    }

    /// <summary>Times <see cref="Fma256Chains"/>; the timer calls stay outside the loop's method.</summary>
    /// <remarks>
    /// A call inside the method that holds the chains makes the JIT keep all eight accumulators in
    /// the frame across it, and the loop then measures store forwarding (about half the real rate).
    /// </remarks>
    private static double Fma256Loop(int iterations)
    {
        long t0 = Stopwatch.GetTimestamp();
        GC.KeepAlive(Fma256Chains(iterations));
        double seconds = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        return seconds > 0 ? FlopsPer(iterations, 8) / seconds / 1e9 : 0;
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Fma256Chains(int iterations)
    {
        Vector256<float> x = Vector256.Create(1.0000001f), y = Vector256.Create(0.9999999f);
        Vector256<float> a0 = Vector256.Create(1f), a1 = Vector256.Create(2f), a2 = Vector256.Create(3f), a3 = Vector256.Create(4f);
        Vector256<float> a4 = Vector256.Create(5f), a5 = Vector256.Create(6f), a6 = Vector256.Create(7f), a7 = Vector256.Create(8f);
        for (int i = 0; i < iterations; i++)
        {
            a0 = Vector256.FusedMultiplyAdd(x, y, a0);
            a1 = Vector256.FusedMultiplyAdd(x, y, a1);
            a2 = Vector256.FusedMultiplyAdd(x, y, a2);
            a3 = Vector256.FusedMultiplyAdd(x, y, a3);
            a4 = Vector256.FusedMultiplyAdd(x, y, a4);
            a5 = Vector256.FusedMultiplyAdd(x, y, a5);
            a6 = Vector256.FusedMultiplyAdd(x, y, a6);
            a7 = Vector256.FusedMultiplyAdd(x, y, a7);
        }

        return Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
    }

    private static double ReadRate(int bytes, double seconds)
    {
        float[] buffer = new float[bytes / sizeof(float)];
        buffer.AsSpan().Fill(1f);
        int passes = Math.Max(1, (int)(seconds * 4e9 / bytes));
        GC.KeepAlive(Read(buffer, 1));
        double best = 0;
        for (int s = 0; s < 3; s++)
        {
            long t0 = Stopwatch.GetTimestamp();
            GC.KeepAlive(Read(buffer, Math.Max(1, passes / 3)));
            double elapsed = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            best = Math.Max(best, elapsed > 0 ? (double)bytes * Math.Max(1, passes / 3) / elapsed / 1e9 : 0);
        }

        return best;
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Read(float[] buffer, int passes)
    {
        Vector256<float> a0 = Vector256.Create(1f), a1 = Vector256.Create(2f), a2 = Vector256.Create(3f), a3 = Vector256.Create(4f);
        Vector256<float> a4 = Vector256.Create(5f), a5 = Vector256.Create(6f), a6 = Vector256.Create(7f), a7 = Vector256.Create(8f);
        int step = Vector256<float>.Count * 8;
        for (int pass = 0; pass < passes; pass++)
        {
            ref float r = ref buffer[0];
            for (int i = 0; i <= buffer.Length - step; i += step)
            {
                a0 += Vector256.LoadUnsafe(ref r, (nuint)i);
                a1 += Vector256.LoadUnsafe(ref r, (nuint)(i + 8));
                a2 += Vector256.LoadUnsafe(ref r, (nuint)(i + 16));
                a3 += Vector256.LoadUnsafe(ref r, (nuint)(i + 24));
                a4 += Vector256.LoadUnsafe(ref r, (nuint)(i + 32));
                a5 += Vector256.LoadUnsafe(ref r, (nuint)(i + 40));
                a6 += Vector256.LoadUnsafe(ref r, (nuint)(i + 48));
                a7 += Vector256.LoadUnsafe(ref r, (nuint)(i + 56));
            }
        }

        return Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
    }
}
