using System.Numerics;
using System.Runtime.CompilerServices;

namespace IndraOcr.Core;

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
            long work = (long)m * n * k;
            int workers = Parallelism.Workers;
            if (workers <= 1 || work < 1 << 18)
            {
                Region(pa, lda, pb, ldb, pc, ldc, 0, m, 0, n, k, accumulate);
                return;
            }

            RunParallel((nint)pa, lda, (nint)pb, ldb, (nint)pc, ldc, m, n, k, accumulate, workers);
        }
    }

    private static void RunParallel(nint pa, int lda, nint pb, int ldb, nint pc, int ldc, int m, int n, int k, bool accumulate, int workers)
    {
        int vw = Vector<float>.Count;
        if (n >= m || m < 8)
        {
            // Split the columns into chunks that are whole multiples of the tile width.
            int step = 2 * vw;
            int chunks = Math.Min(workers * 4, Math.Max(1, n / step));
            int size = ((n + chunks - 1) / chunks + step - 1) / step * step;
            chunks = (n + size - 1) / size;
            Parallel.For(0, chunks, Parallelism.Options, ci =>
            {
                int j0 = ci * size;
                int j1 = Math.Min(n, j0 + size);
                Region((float*)pa, lda, (float*)pb, ldb, (float*)pc, ldc, 0, m, j0, j1, k, accumulate);
            });
        }
        else
        {
            int chunks = Math.Min(workers * 4, Math.Max(1, m / 4));
            int size = ((m + chunks - 1) / chunks + 3) / 4 * 4;
            chunks = (m + size - 1) / size;
            Parallel.For(0, chunks, Parallelism.Options, ci =>
            {
                int i0 = ci * size;
                int i1 = Math.Min(m, i0 + size);
                Region((float*)pa, lda, (float*)pb, ldb, (float*)pc, ldc, i0, i1, 0, n, k, accumulate);
            });
        }
    }

    /// <summary>Computes the output block rows <c>[i0,i1)</c> × columns <c>[j0,j1)</c>.</summary>
    private static void Region(float* a, int lda, float* b, int ldb, float* c, int ldc, int i0, int i1, int j0, int j1, int k, bool accumulate)
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

                for (; i < i1; i++)
                {
                    Rows1(a + (i * (long)lda) + kb, b + (kb * (long)ldb), ldb, c + (i * (long)ldc), jb, je, kc, add);
                }
            }
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
        const int T = 32;
        for (int i0 = 0; i0 < rows; i0 += T)
        {
            int i1 = Math.Min(rows, i0 + T);
            for (int j0 = 0; j0 < cols; j0 += T)
            {
                int j1 = Math.Min(cols, j0 + T);
                for (int i = i0; i < i1; i++)
                {
                    for (int j = j0; j < j1; j++)
                    {
                        dst[(j * rows) + i] = src[(i * cols) + j];
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
