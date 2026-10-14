namespace NanoOcr.Core;

/// <summary>
/// How many threads the kernels spread one operator across.
/// </summary>
/// <remarks>
/// <para>
/// <c>Parallel.For</c>'s default degree is unbounded, which is a policy for work that blocks.
/// None of the work here blocks, so every kernel caps itself at <see cref="Options"/>, one worker
/// per core by default.
/// </para>
/// <para>
/// The setting is ambient (an <see cref="AsyncLocal{T}"/>) rather than a parameter because the
/// kernels sit several layers below the pipeline. A caller that runs several pages at once —
/// <c>NanoOCR.PredictMany</c> runs one page per core — scopes the kernels inside each page to
/// a single thread with <see cref="Use(ParallelOptions)"/>, which is how Python's one-process-per-core pool
/// is reproduced without processes.
/// </para>
/// </remarks>
public static class Parallelism
{
    private static readonly AsyncLocal<ParallelOptions?> Scoped = new();

    private static ParallelOptions _default = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    /// <summary>The options every kernel uses when nothing narrower is in scope.</summary>
    public static ParallelOptions Default
    {
        get => _default;
        set => _default = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>The options in effect here.</summary>
    public static ParallelOptions Options => Scoped.Value ?? _default;

    /// <summary>Workers the kernels may use here.</summary>
    public static int Workers
    {
        get
        {
            int w = Options.MaxDegreeOfParallelism;
            return w <= 0 ? Environment.ProcessorCount : w;
        }
    }

    /// <summary>Makes <paramref name="options"/> the kernels' options until the scope is disposed.</summary>
    public static Scope Use(ParallelOptions? options) => new(options);

    /// <summary>Makes the kernels use <paramref name="threads"/> threads until the scope is disposed.</summary>
    public static Scope Use(int threads) => new(new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, threads) });

    /// <summary>
    /// Runs <paramref name="body"/> over <c>[0, count)</c> split into contiguous chunks, in parallel
    /// when <paramref name="work"/> (a rough count of multiply-adds) is worth it.
    /// </summary>
    public static void For(int count, long work, Action<int, int> body)
    {
        int workers = Workers;
        int chunks = (int)Math.Min(Math.Min(workers, count), Math.Max(1, work / 65536));
        if (chunks <= 1)
        {
            body(0, count);
            return;
        }

        ForChunks(count, chunks, body);
    }

    /// <summary>
    /// Runs independent jobs of different sizes at once — the recognizer's width batches, say:
    /// up to <see cref="Workers"/> at a time, largest first, each with a share of the workers
    /// proportional to its cost (at least one) as the degree of the kernels inside it. One dominant
    /// job therefore still gets every worker, and many similar ones get one each: parallelism
    /// across jobs where a job is too small to split well, within a job where it is not.
    /// </summary>
    /// <param name="costs">A relative cost per job (e.g. lines × width).</param>
    /// <param name="body">Runs job <c>i</c>; jobs must not depend on each other.</param>
    public static void Jobs(IReadOnlyList<double> costs, Action<int> body)
    {
        int n = costs.Count;
        int workers = Workers;
        if (workers <= 1 || n <= 1)
        {
            for (int i = 0; i < n; i++)
            {
                body(i);
            }

            return;
        }

        double total = Math.Max(costs.Sum(), double.Epsilon);
        int[] order = Enumerable.Range(0, n).OrderByDescending(i => costs[i]).ToArray();
        ParallelOptions current = Options;
        var outer = new ParallelOptions
        {
            MaxDegreeOfParallelism = workers,
            CancellationToken = current.CancellationToken,
            TaskScheduler = current.TaskScheduler ?? TaskScheduler.Default,
        };

        // No buffering: workers take jobs one at a time, in the order given (largest first).
        Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(order, System.Collections.Concurrent.EnumerablePartitionerOptions.NoBuffering), outer, i =>
        {
            int share = (int)Math.Clamp(Math.Round(workers * costs[i] / total), 1, workers);
            using (Use(new ParallelOptions { MaxDegreeOfParallelism = share, CancellationToken = current.CancellationToken }))
            {
                body(i);
            }
        });
    }

    private static void ForChunks(int count, int chunks, Action<int, int> body)
    {
        int size = (count + chunks - 1) / chunks;
        Parallel.For(0, chunks, Options, c =>
        {
            int start = c * size;
            int end = Math.Min(count, start + size);
            if (start < end)
            {
                body(start, end);
            }
        });
    }

    /// <summary>A period during which the kernels use one caller's options.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly ParallelOptions? _previous;
        private readonly bool _entered;

        internal Scope(ParallelOptions? options)
        {
            if (options is null)
            {
                _previous = null;
                _entered = false;
                return;
            }

            _previous = Scoped.Value;
            _entered = true;
            Scoped.Value = options;
        }

        /// <summary>Restores the options that were in scope before.</summary>
        public void Dispose()
        {
            if (_entered)
            {
                Scoped.Value = _previous;
            }
        }
    }
}
