using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NanoOcr.Core;

/// <summary>
/// Single-precision matrix products, <c>C = A·B (+ C)</c>, row-major, on <see cref="Vector{T}"/>
/// (AVX2 on x64, NEON on Arm64), multithreaded through <see cref="Parallelism"/>.
/// </summary>
/// <remarks>
/// <para>
/// The micro-kernel holds a 4-row × 2-vector tile of <c>C</c> in registers for a whole block of
/// the reduction: eight independent fused multiply-add chains, six loads per eight FMAs. The
/// reduction is blocked at <see cref="KBlock"/> and the columns at <see cref="NBlock"/> so the
/// slice of <c>B</c> a tile row sweeps stays in L2 while every row group of <c>A</c> passes over it.
/// </para>
/// <para>
/// Work is split across threads along whichever of the two output dimensions is larger. The
/// convolutions put the pixels in the columns, so they split by pixel; attention products with a
/// short inner dimension split by row.
/// </para>
/// </remarks>
public static unsafe class Gemm
{
    private const int KBlock = 256;
    private const int NBlock = 256;

    /// <summary>
    /// Whether the 512-bit kernels run: the hardware has AVX-512 and the runtime accelerates it.
    /// <see cref="Vector{T}"/> stays 256 bits wide on such machines by default, so the 256-bit
    /// kernels would leave half of every FMA unit idle.
    /// </summary>
    public static readonly bool Wide = Vector512.IsHardwareAccelerated && Environment.GetEnvironmentVariable("NANOOCR_GEMM_256") != "1";

    /// <summary>Columns of one packed strip of <c>B</c> (the micro-kernel's tile width).</summary>
    internal static int StripWidth => Wide ? 32 : 2 * Vector<float>.Count;

    /// <summary>Computes <c>C[M,N] = A[M,K] · B[K,N]</c>, or adds it to <c>C</c> when <paramref name="accumulate"/>.</summary>
    public static void Multiply(
        int m, int n, int k,
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        bool accumulate = false)
    {
        if (m == 0 || n == 0)
        {
            return;
        }

        if (k == 0)
        {
            if (!accumulate)
            {
                for (int i = 0; i < m; i++)
                {
                    c.Slice(i * ldc, n).Clear();
                }
            }

            return;
        }

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            Run(pa, lda, pb, ldb, pc, ldc, m, n, k, accumulate, Segments.None);
        }
    }

    /// <summary>
    /// <c>C_i[M, T] = A[M, K] · B_i[K, T]</c> for <paramref name="batch"/> items laid out one after
    /// another (<c>B_i</c> at <c>i·bStride</c>, <c>C_i</c> at <c>i·cStride</c>, rows <c>T</c> apart):
    /// one product over <c>batch·T</c> columns whose packing reads the items in place, so a 1×1
    /// convolution over a batch of short sequences needs no copy into <c>[K, batch·T]</c> and back.
    /// Each output is the same in-order FMA chain as <see cref="Multiply"/> computes. Requires the
    /// 512-bit kernels (<see cref="Wide"/>).
    /// </summary>
    public static void MultiplyBatched(int m, int t, int batch, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, long bStride, Span<float> c, long cStride)
    {
        if (!Wide)
        {
            throw new NotSupportedException("MultiplyBatched needs the 512-bit kernels.");
        }

        if (m == 0 || t == 0 || batch == 0)
        {
            return;
        }

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            Run(pa, lda, pb, t, pc, t, m, t * batch, k, false, new Segments(t, bStride, cStride));
        }
    }

    /// <summary>How logical columns map to memory: <see cref="Len"/> columns per item, items <see cref="BStride"/> / <see cref="CStride"/> apart.</summary>
    private readonly record struct Segments(int Len, long BStride, long CStride)
    {
        public static readonly Segments None = new(int.MaxValue, 0, 0);

        public bool IsNone => Len == int.MaxValue;

        public long B(int j) => ((j / Len) * BStride) + (j % Len);

        public long C(int j) => ((j / Len) * CStride) + (j % Len);
    }

    private static void Run(float* pa, int lda, float* pb, int ldb, float* pc, int ldc, int m, int n, int k, bool accumulate, Segments seg)
    {
        long work = (long)m * n * k;
        int workers = Parallelism.Workers;
        if (workers <= 1 || work < 1 << 20)
        {
            Region(pa, lda, pb, ldb, pc, ldc, 0, m, 0, n, k, accumulate, seg);
            return;
        }

        RunParallel((nint)pa, lda, (nint)pb, ldb, (nint)pc, ldc, m, n, k, accumulate, workers, seg);
    }

    private static void RunParallel(nint pa, int lda, nint pb, int ldb, nint pc, int ldc, int m, int n, int k, bool accumulate, int workers, Segments seg)
    {
        if (n >= m || m < 8)
        {
            // Split the columns into chunks that are whole multiples of the tile width.
            int step = StripWidth;
            int chunks = Math.Min(workers * 4, Math.Max(1, n / step));
            int size = ((n + chunks - 1) / chunks + step - 1) / step * step;
            chunks = (n + size - 1) / size;
            Parallel.For(0, chunks, Parallelism.Options, ci =>
            {
                int j0 = ci * size;
                int j1 = Math.Min(n, j0 + size);
                Region((float*)pa, lda, (float*)pb, ldb, (float*)pc, ldc, 0, m, j0, j1, k, accumulate, seg);
            });
        }
        else
        {
            int mr = Wide ? 8 : 6;
            int chunks = Math.Min(workers * 4, Math.Max(1, m / mr));
            int size = ((m + chunks - 1) / chunks + mr - 1) / mr * mr;
            chunks = (m + size - 1) / size;
            Parallel.For(0, chunks, Parallelism.Options, ci =>
            {
                int i0 = ci * size;
                int i1 = Math.Min(m, i0 + size);
                Region((float*)pa, lda, (float*)pb, ldb, (float*)pc, ldc, i0, i1, 0, n, k, accumulate, seg);
            });
        }
    }

    /// <summary>Computes the output block rows <c>[i0,i1)</c> × columns <c>[j0,j1)</c>.</summary>
    /// <remarks>
    /// For each block of columns and of the reduction, the slice of <c>B</c> is packed into
    /// contiguous strips of <c>2·vw</c> columns (zero-padded at the edge), so the micro-kernel
    /// streams it with sequential loads; rows of <c>A</c> are broadcast straight from memory.
    /// Six rows at a time hold twelve accumulators in registers. Every output element is still a
    /// single fused multiply-add chain over k in order, so the packing does not change results.
    /// </remarks>
    private static void Region(float* a, int lda, float* b, int ldb, float* c, int ldc, int i0, int i1, int j0, int j1, int k, bool accumulate, Segments seg)
    {
        if (Wide && (i1 - i0 >= 4 || !seg.IsNone))
        {
            Region512(a, lda, b, ldb, c, ldc, i0, i1, j0, j1, k, accumulate, seg);
            return;
        }

        int nr = 2 * Vector<float>.Count;
        int rows = i1 - i0;
        if (rows < 6)
        {
            RegionUnpacked(a, lda, b, ldb, c, ldc, i0, i1, j0, j1, k, accumulate);
            return;
        }

        float[] buffer = System.Buffers.ArrayPool<float>.Shared.Rent(KBlock * (NBlock + nr));
        try
        {
            fixed (float* packed = buffer)
            {
                for (int jb = j0; jb < j1; jb += NBlock)
                {
                    int je = Math.Min(j1, jb + NBlock);
                    int strips = (je - jb + nr - 1) / nr;
                    for (int kb = 0; kb < k; kb += KBlock)
                    {
                        int kc = Math.Min(KBlock, k - kb);
                        bool add = accumulate || kb > 0;
                        Pack(b + (kb * (long)ldb), ldb, jb, je, kc, nr, packed);
                        for (int s = 0; s < strips; s++)
                        {
                            int js = jb + (s * nr);
                            int width = Math.Min(nr, je - js);
                            float* strip = packed + ((long)s * kc * nr);
                            int i = i0;
                            for (; i + 6 <= i1; i += 6)
                            {
                                Tile6(a + (i * (long)lda) + kb, lda, strip, kc, c + (i * (long)ldc) + js, ldc, width, add);
                            }

                            for (; i < i1; i++)
                            {
                                Tile1(a + (i * (long)lda) + kb, strip, kc, c + (i * (long)ldc) + js, width, add);
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>Packs <c>B[0..kc, jb..je]</c> into strips of <paramref name="nr"/> columns, zero-padded.</summary>
    private static void Pack(float* b, int ldb, int jb, int je, int kc, int nr, float* packed)
    {
        int strips = (je - jb + nr - 1) / nr;
        for (int s = 0; s < strips; s++)
        {
            int js = jb + (s * nr);
            int width = Math.Min(nr, je - js);
            float* dst = packed + ((long)s * kc * nr);
            if (width == nr)
            {
                for (int p = 0; p < kc; p++)
                {
                    Buffer.MemoryCopy(b + (p * (long)ldb) + js, dst + (p * nr), nr * sizeof(float), nr * sizeof(float));
                }
            }
            else
            {
                for (int p = 0; p < kc; p++)
                {
                    float* row = b + (p * (long)ldb) + js;
                    float* o = dst + (p * nr);
                    int q = 0;
                    for (; q < width; q++)
                    {
                        o[q] = row[q];
                    }

                    for (; q < nr; q++)
                    {
                        o[q] = 0f;
                    }
                }
            }
        }
    }

    /// <summary>A 6-row tile; partial-width tiles go through a scratch tile so the kernel itself has no edge cases.</summary>
    private static void Tile6(float* a, int lda, float* bp, int kc, float* c, int ldc, int width, bool add)
    {
        int nr = 2 * Vector<float>.Count;
        if (width == nr)
        {
            Kernel6(a, lda, bp, kc, c, ldc, add);
            return;
        }

        EdgeTile6(a, lda, bp, kc, c, ldc, width, add, nr);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void EdgeTile6(float* a, int lda, float* bp, int kc, float* c, int ldc, int width, bool add, int nr)
    {
        float* t = stackalloc float[6 * 64];
        for (int r = 0; r < 6; r++)
        {
            for (int q = 0; q < nr; q++)
            {
                t[(r * nr) + q] = add && q < width ? c[(r * ldc) + q] : 0f;
            }
        }

        Kernel6(a, lda, bp, kc, t, nr, add);
        for (int r = 0; r < 6; r++)
        {
            for (int q = 0; q < width; q++)
            {
                c[(r * ldc) + q] = t[(r * nr) + q];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    private static void Kernel6(float* a, int lda, float* bp, int kc, float* c, int ldc, bool add)
    {
        int vw = Vector<float>.Count;
        float* a0 = a, a1 = a + lda, a2 = a + (2 * lda), a3 = a + (3 * lda), a4 = a + (4 * lda), a5 = a + (5 * lda);
        Vector<float> s00, s01, s10, s11, s20, s21, s30, s31, s40, s41, s50, s51;
        if (add)
        {
            s00 = Load(c); s01 = Load(c + vw);
            s10 = Load(c + ldc); s11 = Load(c + ldc + vw);
            s20 = Load(c + (2 * ldc)); s21 = Load(c + (2 * ldc) + vw);
            s30 = Load(c + (3 * ldc)); s31 = Load(c + (3 * ldc) + vw);
            s40 = Load(c + (4 * ldc)); s41 = Load(c + (4 * ldc) + vw);
            s50 = Load(c + (5 * ldc)); s51 = Load(c + (5 * ldc) + vw);
        }
        else
        {
            s00 = s01 = s10 = s11 = s20 = s21 = s30 = s31 = s40 = s41 = s50 = s51 = Vector<float>.Zero;
        }

        for (int p = 0; p < kc; p++, bp += 2 * vw)
        {
            Vector<float> b0 = Load(bp);
            Vector<float> b1 = Load(bp + vw);
            var v = new Vector<float>(a0[p]);
            s00 = Vector.FusedMultiplyAdd(v, b0, s00);
            s01 = Vector.FusedMultiplyAdd(v, b1, s01);
            v = new Vector<float>(a1[p]);
            s10 = Vector.FusedMultiplyAdd(v, b0, s10);
            s11 = Vector.FusedMultiplyAdd(v, b1, s11);
            v = new Vector<float>(a2[p]);
            s20 = Vector.FusedMultiplyAdd(v, b0, s20);
            s21 = Vector.FusedMultiplyAdd(v, b1, s21);
            v = new Vector<float>(a3[p]);
            s30 = Vector.FusedMultiplyAdd(v, b0, s30);
            s31 = Vector.FusedMultiplyAdd(v, b1, s31);
            v = new Vector<float>(a4[p]);
            s40 = Vector.FusedMultiplyAdd(v, b0, s40);
            s41 = Vector.FusedMultiplyAdd(v, b1, s41);
            v = new Vector<float>(a5[p]);
            s50 = Vector.FusedMultiplyAdd(v, b0, s50);
            s51 = Vector.FusedMultiplyAdd(v, b1, s51);
        }

        Store(c, s00); Store(c + vw, s01);
        Store(c + ldc, s10); Store(c + ldc + vw, s11);
        Store(c + (2 * ldc), s20); Store(c + (2 * ldc) + vw, s21);
        Store(c + (3 * ldc), s30); Store(c + (3 * ldc) + vw, s31);
        Store(c + (4 * ldc), s40); Store(c + (4 * ldc) + vw, s41);
        Store(c + (5 * ldc), s50); Store(c + (5 * ldc) + vw, s51);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile1(float* a, float* bp, int kc, float* c, int width, bool add)
    {
        int vw = Vector<float>.Count;
        float* t = stackalloc float[16];
        Vector<float> s0, s1;
        if (add)
        {
            for (int q = 0; q < 2 * vw; q++)
            {
                t[q] = q < width ? c[q] : 0f;
            }

            s0 = Load(t);
            s1 = Load(t + vw);
        }
        else
        {
            s0 = s1 = Vector<float>.Zero;
        }

        for (int p = 0; p < kc; p++, bp += 2 * vw)
        {
            var v = new Vector<float>(a[p]);
            s0 = Vector.FusedMultiplyAdd(v, Load(bp), s0);
            s1 = Vector.FusedMultiplyAdd(v, Load(bp + vw), s1);
        }

        Store(t, s0);
        Store(t + vw, s1);
        for (int q = 0; q < width; q++)
        {
            c[q] = t[q];
        }
    }

    /// <summary>
    /// <see cref="Region"/> on 512-bit vectors: strips of 32 columns and an 8-row tile (sixteen
    /// accumulators), with 4-, 2- and 1-row tiles for the remainder, so the channel counts these
    /// models use (multiples of 8, and 1678 = 8·209 + 6) rarely leave a row to the 1-row kernel.
    /// Every output is still one in-order FMA chain over k, so the results equal the 256-bit path's.
    /// </summary>
    private static void Region512(float* a, int lda, float* b, int ldb, float* c, int ldc, int i0, int i1, int j0, int j1, int k, bool accumulate, Segments seg)
    {
        const int nr = 32;
        float[] buffer = System.Buffers.ArrayPool<float>.Shared.Rent(KBlock * (NBlock + nr));
        try
        {
            fixed (float* packed = buffer)
            {
                for (int jb = j0; jb < j1; jb += NBlock)
                {
                    int je = Math.Min(j1, jb + NBlock);
                    int strips = (je - jb + nr - 1) / nr;
                    for (int kb = 0; kb < k; kb += KBlock)
                    {
                        int kc = Math.Min(KBlock, k - kb);
                        bool add = accumulate || kb > 0;
                        if (seg.IsNone)
                        {
                            Pack(b + (kb * (long)ldb), ldb, jb, je, kc, nr, packed);
                        }
                        else
                        {
                            PackSegments(b + (kb * (long)ldb), ldb, jb, je, kc, nr, packed, seg);
                        }

                        for (int s = 0; s < strips; s++)
                        {
                            int js = jb + (s * nr);
                            int width = Math.Min(nr, je - js);
                            float* strip = packed + ((long)s * kc * nr);
                            int i = i0;

                            // A strip that straddles two items cannot be stored with one row stride.
                            bool split = !seg.IsNone && js / seg.Len != (js + width - 1) / seg.Len;
                            float* cs = c + (seg.IsNone ? js : seg.C(js));
                            if (width == nr && !split)
                            {
                                for (; i + 8 <= i1; i += 8)
                                {
                                    K512x8(a + (i * (long)lda) + kb, lda, strip, kc, cs + (i * (long)ldc), ldc, add);
                                }

                                if (i + 4 <= i1)
                                {
                                    K512x4(a + (i * (long)lda) + kb, lda, strip, kc, cs + (i * (long)ldc), ldc, add);
                                    i += 4;
                                }

                                if (i + 2 <= i1)
                                {
                                    K512x2(a + (i * (long)lda) + kb, lda, strip, kc, cs + (i * (long)ldc), ldc, add);
                                    i += 2;
                                }

                                if (i < i1)
                                {
                                    K512x1(a + (i * (long)lda) + kb, strip, kc, cs + (i * (long)ldc), add);
                                }
                            }
                            else
                            {
                                Edge512(a + (i * (long)lda) + kb, lda, strip, kc, c + (i * (long)ldc), ldc, i1 - i, js, width, add, seg);
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// A strip narrower than 32 columns, or one split between two items: through a scratch tile,
    /// eight rows at a time; <paramref name="c"/> points at column 0 of the strip's first row.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Edge512(float* a, int lda, float* bp, int kc, float* c, int ldc, int rows, int js, int width, bool add, Segments seg)
    {
        const int nr = 32;
        float* t = stackalloc float[8 * nr];
        long* col = stackalloc long[nr];
        for (int q = 0; q < width; q++)
        {
            col[q] = seg.IsNone ? js + q : seg.C(js + q);
        }

        for (int r0 = 0; r0 < rows; r0 += 8)
        {
            int rr = Math.Min(8, rows - r0);
            for (int r = 0; r < rr; r++)
            {
                for (int q = 0; q < nr; q++)
                {
                    t[(r * nr) + q] = add && q < width ? c[((r0 + r) * (long)ldc) + col[q]] : 0f;
                }
            }

            float* ar = a + (r0 * (long)lda);
            int i = 0;
            if (rr == 8)
            {
                K512x8(ar, lda, bp, kc, t, nr, add);
                i = 8;
            }

            if (i + 4 <= rr)
            {
                K512x4(ar + (i * (long)lda), lda, bp, kc, t + (i * nr), nr, add);
                i += 4;
            }

            if (i + 2 <= rr)
            {
                K512x2(ar + (i * (long)lda), lda, bp, kc, t + (i * nr), nr, add);
                i += 2;
            }

            if (i < rr)
            {
                K512x1(ar + (i * (long)lda), bp, kc, t + (i * nr), add);
            }

            for (int r = 0; r < rr; r++)
            {
                for (int q = 0; q < width; q++)
                {
                    c[((r0 + r) * (long)ldc) + col[q]] = t[(r * nr) + q];
                }
            }
        }
    }

    /// <summary><see cref="Pack"/> for columns laid out in items (<see cref="Segments"/>).</summary>
    private static void PackSegments(float* b, int ldb, int jb, int je, int kc, int nr, float* packed, Segments seg)
    {
        int strips = (je - jb + nr - 1) / nr;
        for (int s = 0; s < strips; s++)
        {
            int js = jb + (s * nr);
            int width = Math.Min(nr, je - js);
            float* dst = packed + ((long)s * kc * nr);
            for (int q0 = 0; q0 < nr;)
            {
                if (q0 >= width)
                {
                    for (int p = 0; p < kc; p++)
                    {
                        new Span<float>(dst + (p * nr) + q0, nr - q0).Clear();
                    }

                    break;
                }

                // A run of columns inside one item is contiguous in every row.
                int j = js + q0;
                int run = Math.Min(width - q0, seg.Len - (j % seg.Len));
                float* src = b + seg.B(j);
                for (int p = 0; p < kc; p++)
                {
                    Buffer.MemoryCopy(src + (p * (long)ldb), dst + (p * nr) + q0, run * sizeof(float), run * sizeof(float));
                }

                q0 += run;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    private static void K512x8(float* a, int lda, float* bp, int kc, float* c, int ldc, bool add)
    {
        float* a0 = a, a1 = a + lda, a2 = a + (2 * lda), a3 = a + (3 * lda);
        float* a4 = a + (4 * lda), a5 = a + (5 * lda), a6 = a + (6 * lda), a7 = a + (7 * lda);
        Vector512<float> s00, s01, s10, s11, s20, s21, s30, s31, s40, s41, s50, s51, s60, s61, s70, s71;
        if (add)
        {
            s00 = Vector512.Load(c); s01 = Vector512.Load(c + 16);
            s10 = Vector512.Load(c + ldc); s11 = Vector512.Load(c + ldc + 16);
            s20 = Vector512.Load(c + (2 * ldc)); s21 = Vector512.Load(c + (2 * ldc) + 16);
            s30 = Vector512.Load(c + (3 * ldc)); s31 = Vector512.Load(c + (3 * ldc) + 16);
            s40 = Vector512.Load(c + (4 * ldc)); s41 = Vector512.Load(c + (4 * ldc) + 16);
            s50 = Vector512.Load(c + (5 * ldc)); s51 = Vector512.Load(c + (5 * ldc) + 16);
            s60 = Vector512.Load(c + (6 * ldc)); s61 = Vector512.Load(c + (6 * ldc) + 16);
            s70 = Vector512.Load(c + (7 * ldc)); s71 = Vector512.Load(c + (7 * ldc) + 16);
        }
        else
        {
            s00 = s01 = s10 = s11 = s20 = s21 = s30 = s31 = Vector512<float>.Zero;
            s40 = s41 = s50 = s51 = s60 = s61 = s70 = s71 = Vector512<float>.Zero;
        }

        for (int p = 0; p < kc; p++, bp += 32)
        {
            Vector512<float> b0 = Vector512.Load(bp);
            Vector512<float> b1 = Vector512.Load(bp + 16);
            Vector512<float> v = Vector512.Create(a0[p]);
            s00 = Vector512.FusedMultiplyAdd(v, b0, s00);
            s01 = Vector512.FusedMultiplyAdd(v, b1, s01);
            v = Vector512.Create(a1[p]);
            s10 = Vector512.FusedMultiplyAdd(v, b0, s10);
            s11 = Vector512.FusedMultiplyAdd(v, b1, s11);
            v = Vector512.Create(a2[p]);
            s20 = Vector512.FusedMultiplyAdd(v, b0, s20);
            s21 = Vector512.FusedMultiplyAdd(v, b1, s21);
            v = Vector512.Create(a3[p]);
            s30 = Vector512.FusedMultiplyAdd(v, b0, s30);
            s31 = Vector512.FusedMultiplyAdd(v, b1, s31);
            v = Vector512.Create(a4[p]);
            s40 = Vector512.FusedMultiplyAdd(v, b0, s40);
            s41 = Vector512.FusedMultiplyAdd(v, b1, s41);
            v = Vector512.Create(a5[p]);
            s50 = Vector512.FusedMultiplyAdd(v, b0, s50);
            s51 = Vector512.FusedMultiplyAdd(v, b1, s51);
            v = Vector512.Create(a6[p]);
            s60 = Vector512.FusedMultiplyAdd(v, b0, s60);
            s61 = Vector512.FusedMultiplyAdd(v, b1, s61);
            v = Vector512.Create(a7[p]);
            s70 = Vector512.FusedMultiplyAdd(v, b0, s70);
            s71 = Vector512.FusedMultiplyAdd(v, b1, s71);
        }

        s00.Store(c); s01.Store(c + 16);
        s10.Store(c + ldc); s11.Store(c + ldc + 16);
        s20.Store(c + (2 * ldc)); s21.Store(c + (2 * ldc) + 16);
        s30.Store(c + (3 * ldc)); s31.Store(c + (3 * ldc) + 16);
        s40.Store(c + (4 * ldc)); s41.Store(c + (4 * ldc) + 16);
        s50.Store(c + (5 * ldc)); s51.Store(c + (5 * ldc) + 16);
        s60.Store(c + (6 * ldc)); s61.Store(c + (6 * ldc) + 16);
        s70.Store(c + (7 * ldc)); s71.Store(c + (7 * ldc) + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    private static void K512x4(float* a, int lda, float* bp, int kc, float* c, int ldc, bool add)
    {
        float* a0 = a, a1 = a + lda, a2 = a + (2 * lda), a3 = a + (3 * lda);
        Vector512<float> s00, s01, s10, s11, s20, s21, s30, s31;
        if (add)
        {
            s00 = Vector512.Load(c); s01 = Vector512.Load(c + 16);
            s10 = Vector512.Load(c + ldc); s11 = Vector512.Load(c + ldc + 16);
            s20 = Vector512.Load(c + (2 * ldc)); s21 = Vector512.Load(c + (2 * ldc) + 16);
            s30 = Vector512.Load(c + (3 * ldc)); s31 = Vector512.Load(c + (3 * ldc) + 16);
        }
        else
        {
            s00 = s01 = s10 = s11 = s20 = s21 = s30 = s31 = Vector512<float>.Zero;
        }

        for (int p = 0; p < kc; p++, bp += 32)
        {
            Vector512<float> b0 = Vector512.Load(bp);
            Vector512<float> b1 = Vector512.Load(bp + 16);
            Vector512<float> v = Vector512.Create(a0[p]);
            s00 = Vector512.FusedMultiplyAdd(v, b0, s00);
            s01 = Vector512.FusedMultiplyAdd(v, b1, s01);
            v = Vector512.Create(a1[p]);
            s10 = Vector512.FusedMultiplyAdd(v, b0, s10);
            s11 = Vector512.FusedMultiplyAdd(v, b1, s11);
            v = Vector512.Create(a2[p]);
            s20 = Vector512.FusedMultiplyAdd(v, b0, s20);
            s21 = Vector512.FusedMultiplyAdd(v, b1, s21);
            v = Vector512.Create(a3[p]);
            s30 = Vector512.FusedMultiplyAdd(v, b0, s30);
            s31 = Vector512.FusedMultiplyAdd(v, b1, s31);
        }

        s00.Store(c); s01.Store(c + 16);
        s10.Store(c + ldc); s11.Store(c + ldc + 16);
        s20.Store(c + (2 * ldc)); s21.Store(c + (2 * ldc) + 16);
        s30.Store(c + (3 * ldc)); s31.Store(c + (3 * ldc) + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    private static void K512x2(float* a, int lda, float* bp, int kc, float* c, int ldc, bool add)
    {
        float* a0 = a, a1 = a + lda;
        Vector512<float> s00, s01, s10, s11;
        if (add)
        {
            s00 = Vector512.Load(c); s01 = Vector512.Load(c + 16);
            s10 = Vector512.Load(c + ldc); s11 = Vector512.Load(c + ldc + 16);
        }
        else
        {
            s00 = s01 = s10 = s11 = Vector512<float>.Zero;
        }

        for (int p = 0; p < kc; p++, bp += 32)
        {
            Vector512<float> b0 = Vector512.Load(bp);
            Vector512<float> b1 = Vector512.Load(bp + 16);
            Vector512<float> v = Vector512.Create(a0[p]);
            s00 = Vector512.FusedMultiplyAdd(v, b0, s00);
            s01 = Vector512.FusedMultiplyAdd(v, b1, s01);
            v = Vector512.Create(a1[p]);
            s10 = Vector512.FusedMultiplyAdd(v, b0, s10);
            s11 = Vector512.FusedMultiplyAdd(v, b1, s11);
        }

        s00.Store(c); s01.Store(c + 16);
        s10.Store(c + ldc); s11.Store(c + ldc + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    private static void K512x1(float* a, float* bp, int kc, float* c, bool add)
    {
        Vector512<float> s0 = add ? Vector512.Load(c) : Vector512<float>.Zero;
        Vector512<float> s1 = add ? Vector512.Load(c + 16) : Vector512<float>.Zero;
        for (int p = 0; p < kc; p++, bp += 32)
        {
            Vector512<float> v = Vector512.Create(a[p]);
            s0 = Vector512.FusedMultiplyAdd(v, Vector512.Load(bp), s0);
            s1 = Vector512.FusedMultiplyAdd(v, Vector512.Load(bp + 16), s1);
        }

        s0.Store(c);
        s1.Store(c + 16);
    }

    /// <summary>The unpacked kernels, for blocks too short in rows for packing to pay.</summary>
    private static void RegionUnpacked(float* a, int lda, float* b, int ldb, float* c, int ldc, int i0, int i1, int j0, int j1, int k, bool accumulate)
    {
        for (int jb = j0; jb < j1; jb += NBlock)
        {
            int je = Math.Min(j1, jb + NBlock);
            for (int kb = 0; kb < k; kb += KBlock)
            {
                int kc = Math.Min(KBlock, k - kb);
                bool add = accumulate || kb > 0;
                int i = i0;
                for (; i + 4 <= i1; i += 4)
                {
                    Rows4(a + (i * (long)lda) + kb, lda, b + (kb * (long)ldb), ldb, c + (i * (long)ldc), ldc, jb, je, kc, add);
                }

                if (i1 - i > 1)
                {
                    // Two or three rows left: pad to four so B is streamed once, not once per row.
                    RowsPadded(a + (i * (long)lda) + kb, lda, b + (kb * (long)ldb), ldb, c + (i * (long)ldc), ldc, jb, je, kc, add, i1 - i);
                    i = i1;
                }

                for (; i < i1; i++)
                {
                    Rows1(a + (i * (long)lda) + kb, b + (kb * (long)ldb), ldb, c + (i * (long)ldc), jb, je, kc, add);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RowsPadded(float* a, int lda, float* b, int ldb, float* c, int ldc, int j0, int j1, int kc, bool add, int rows)
    {
        int width = j1 - j0;
        float[] scratchA = System.Buffers.ArrayPool<float>.Shared.Rent(4 * kc);
        float[] scratchC = System.Buffers.ArrayPool<float>.Shared.Rent(4 * width);
        try
        {
            fixed (float* sa = scratchA)
            fixed (float* sc = scratchC)
            {
                for (int r = 0; r < 4; r++)
                {
                    for (int p = 0; p < kc; p++)
                    {
                        sa[(r * kc) + p] = r < rows ? a[(r * (long)lda) + p] : 0f;
                    }

                    for (int q = 0; q < width; q++)
                    {
                        sc[(r * width) + q] = add && r < rows ? c[(r * (long)ldc) + j0 + q] : 0f;
                    }
                }

                // Rows4 addresses C by absolute column; shift the scratch so column j0 lands at 0.
                Rows4(sa, kc, b, ldb, sc - j0, width, j0, j1, kc, add);
                for (int r = 0; r < rows; r++)
                {
                    for (int q = 0; q < width; q++)
                    {
                        c[(r * (long)ldc) + j0 + q] = sc[(r * width) + q];
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratchA);
            System.Buffers.ArrayPool<float>.Shared.Return(scratchC);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Rows4(float* a, int lda, float* b, int ldb, float* c, int ldc, int j0, int j1, int kc, bool add)
    {
        int vw = Vector<float>.Count;
        float* a0 = a, a1 = a + lda, a2 = a + (2 * lda), a3 = a + (3 * lda);
        float* c0 = c, c1 = c + ldc, c2 = c + (2 * ldc), c3 = c + (3 * ldc);
        int j = j0;
        for (; j + (2 * vw) <= j1; j += 2 * vw)
        {
            Vector<float> s00, s01, s10, s11, s20, s21, s30, s31;
            if (add)
            {
                s00 = Load(c0 + j); s01 = Load(c0 + j + vw);
                s10 = Load(c1 + j); s11 = Load(c1 + j + vw);
                s20 = Load(c2 + j); s21 = Load(c2 + j + vw);
                s30 = Load(c3 + j); s31 = Load(c3 + j + vw);
            }
            else
            {
                s00 = s01 = s10 = s11 = s20 = s21 = s30 = s31 = Vector<float>.Zero;
            }

            float* bp = b + j;
            for (int p = 0; p < kc; p++, bp += ldb)
            {
                Vector<float> b0 = Load(bp);
                Vector<float> b1 = Load(bp + vw);
                var v = new Vector<float>(a0[p]);
                s00 = Vector.FusedMultiplyAdd(v, b0, s00);
                s01 = Vector.FusedMultiplyAdd(v, b1, s01);
                v = new Vector<float>(a1[p]);
                s10 = Vector.FusedMultiplyAdd(v, b0, s10);
                s11 = Vector.FusedMultiplyAdd(v, b1, s11);
                v = new Vector<float>(a2[p]);
                s20 = Vector.FusedMultiplyAdd(v, b0, s20);
                s21 = Vector.FusedMultiplyAdd(v, b1, s21);
                v = new Vector<float>(a3[p]);
                s30 = Vector.FusedMultiplyAdd(v, b0, s30);
                s31 = Vector.FusedMultiplyAdd(v, b1, s31);
            }

            Store(c0 + j, s00); Store(c0 + j + vw, s01);
            Store(c1 + j, s10); Store(c1 + j + vw, s11);
            Store(c2 + j, s20); Store(c2 + j + vw, s21);
            Store(c3 + j, s30); Store(c3 + j + vw, s31);
        }

        for (; j + vw <= j1; j += vw)
        {
            Vector<float> s0, s1, s2, s3;
            if (add)
            {
                s0 = Load(c0 + j); s1 = Load(c1 + j); s2 = Load(c2 + j); s3 = Load(c3 + j);
            }
            else
            {
                s0 = s1 = s2 = s3 = Vector<float>.Zero;
            }

            float* bp = b + j;
            for (int p = 0; p < kc; p++, bp += ldb)
            {
                Vector<float> b0 = Load(bp);
                s0 = Vector.FusedMultiplyAdd(new Vector<float>(a0[p]), b0, s0);
                s1 = Vector.FusedMultiplyAdd(new Vector<float>(a1[p]), b0, s1);
                s2 = Vector.FusedMultiplyAdd(new Vector<float>(a2[p]), b0, s2);
                s3 = Vector.FusedMultiplyAdd(new Vector<float>(a3[p]), b0, s3);
            }

            Store(c0 + j, s0); Store(c1 + j, s1); Store(c2 + j, s2); Store(c3 + j, s3);
        }

        for (; j < j1; j++)
        {
            float s0 = add ? c0[j] : 0f, s1 = add ? c1[j] : 0f, s2 = add ? c2[j] : 0f, s3 = add ? c3[j] : 0f;
            float* bp = b + j;
            for (int p = 0; p < kc; p++, bp += ldb)
            {
                float bv = *bp;
                s0 = MathF.FusedMultiplyAdd(a0[p], bv, s0);
                s1 = MathF.FusedMultiplyAdd(a1[p], bv, s1);
                s2 = MathF.FusedMultiplyAdd(a2[p], bv, s2);
                s3 = MathF.FusedMultiplyAdd(a3[p], bv, s3);
            }

            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Rows1(float* a, float* b, int ldb, float* c, int j0, int j1, int kc, bool add)
    {
        int vw = Vector<float>.Count;
        int j = j0;
        for (; j + (2 * vw) <= j1; j += 2 * vw)
        {
            Vector<float> s0 = add ? Load(c + j) : Vector<float>.Zero;
            Vector<float> s1 = add ? Load(c + j + vw) : Vector<float>.Zero;
            float* bp = b + j;
            for (int p = 0; p < kc; p++, bp += ldb)
            {
                var v = new Vector<float>(a[p]);
                s0 = Vector.FusedMultiplyAdd(v, Load(bp), s0);
                s1 = Vector.FusedMultiplyAdd(v, Load(bp + vw), s1);
            }

            Store(c + j, s0);
            Store(c + j + vw, s1);
        }

        for (; j + vw <= j1; j += vw)
        {
            Vector<float> s0 = add ? Load(c + j) : Vector<float>.Zero;
            float* bp = b + j;
            for (int p = 0; p < kc; p++, bp += ldb)
            {
                s0 = Vector.FusedMultiplyAdd(new Vector<float>(a[p]), Load(bp), s0);
            }

            Store(c + j, s0);
        }

        for (; j < j1; j++)
        {
            float s = add ? c[j] : 0f;
            float* bp = b + j;
            for (int p = 0; p < kc; p++, bp += ldb)
            {
                s = MathF.FusedMultiplyAdd(a[p], *bp, s);
            }

            c[j] = s;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Load(float* p) => Unsafe.ReadUnaligned<Vector<float>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(float* p, Vector<float> v) => Unsafe.WriteUnaligned(p, v);

    /// <summary>Transposes a row-major <c>[rows, cols]</c> matrix into <paramref name="dst"/> (<c>[cols, rows]</c>).</summary>
    public static void Transpose(ReadOnlySpan<float> src, int rows, int cols, Span<float> dst)
    {
        fixed (float* s = src)
        fixed (float* d = dst)
        {
            TransposeBlock(s, d, rows, cols, 0, rows);
        }
    }

    /// <summary>Transposes <paramref name="batch"/> matrices of <c>[rows, cols]</c>, in parallel.</summary>
    public static void TransposeBatch(float[] src, float[] dst, int batch, int rows, int cols)
    {
        const int T = 32;
        int tiles = (rows + T - 1) / T;
        int jobs = batch * tiles;
        Parallelism.For(jobs, (long)batch * rows * cols, (j0, j1) =>
        {
            fixed (float* s = src)
            fixed (float* d = dst)
            {
                for (int j = j0; j < j1; j++)
                {
                    int b = j / tiles, t = j % tiles;
                    long off = (long)b * rows * cols;
                    TransposeBlock(s + off, d + off, rows, cols, t * T, Math.Min(rows, (t + 1) * T));
                }
            }
        });
    }

    private static void TransposeBlock(float* s, float* d, int rows, int cols, int r0, int r1)
    {
        const int T = 32;
        for (int i0 = r0; i0 < r1; i0 += T)
        {
            int i1 = Math.Min(r1, i0 + T);
            for (int j0 = 0; j0 < cols; j0 += T)
            {
                int j1 = Math.Min(cols, j0 + T);
                for (int i = i0; i < i1; i++)
                {
                    float* srow = s + ((long)i * cols);
                    for (int j = j0; j < j1; j++)
                    {
                        d[((long)j * rows) + i] = srow[j];
                    }
                }
            }
        }
    }
}

/// <summary>Entry points for callers that already parallelize at a coarser grain.</summary>
public static unsafe class GemmSerial
{
    /// <summary><see cref="Gemm.Multiply"/> on the calling thread only.</summary>
    public static void Multiply(
        int m, int n, int k,
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        bool accumulate = false)
    {
        using (Parallelism.Use(1))
        {
            Gemm.Multiply(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
        }
    }
}
