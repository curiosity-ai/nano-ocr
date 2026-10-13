namespace IndraOcr.Core;

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
/// <c>IndraOCR.PredictMany</c> runs one page per core — scopes the kernels inside each page to
/// a single thread with <see cref="Use"/>, which is how Python's one-process-per-core pool
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
