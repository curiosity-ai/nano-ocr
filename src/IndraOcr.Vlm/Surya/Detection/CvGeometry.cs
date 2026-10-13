namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>OpenCV's <c>RotatedRect</c>: centre, size and angle in degrees, all float32.</summary>
public readonly record struct RotatedRect(float CenterX, float CenterY, float Width, float Height, float Angle);

/// <summary>
/// The OpenCV 4.11 shape functions <c>heatmap.py</c> calls, ported from their sources with their
/// float32 arithmetic kept operation for operation: <c>cv::convexHull</c> (Sklansky's scan over
/// integer points, <c>convhull.cpp</c>), <c>cv::minAreaRect</c> (rotating calipers,
/// <c>rotcalipers.cpp</c>), <c>cv::boxPoints</c> (<c>RotatedRect::points</c>, <c>types.cpp</c>)
/// and <c>cv::dilate</c> with a rectangular element.
/// </summary>
public static class CvGeometry
{
    /// <summary>
    /// <c>cv2.minAreaRect</c> of integer points (numpy's int64 arrays reach OpenCV as
    /// <c>CV_32S</c>).
    /// </summary>
    public static RotatedRect MinAreaRect(ReadOnlySpan<int> xs, ReadOnlySpan<int> ys)
    {
        int[] hull = ConvexHull(xs, ys);
        int n = hull.Length;
        var px = new float[n];
        var py = new float[n];
        for (int i = 0; i < n; i++)
        {
            px[i] = xs[hull[i]];
            py[i] = ys[hull[i]];
        }

        float cx = 0, cy = 0, width = 0, height = 0, angle = 0;
        if (n > 2)
        {
            Span<float> o = stackalloc float[6];
            RotatingCalipers(px, py, n, o);
            cx = o[0] + ((o[2] + o[4]) * 0.5f);
            cy = o[1] + ((o[3] + o[5]) * 0.5f);
            width = (float)Math.Sqrt(((double)o[2] * o[2]) + ((double)o[3] * o[3]));
            height = (float)Math.Sqrt(((double)o[4] * o[4]) + ((double)o[5] * o[5]));
            angle = (float)Math.Atan2(o[3], o[2]);
        }
        else if (n == 2)
        {
            cx = (px[0] + px[1]) * 0.5f;
            cy = (py[0] + py[1]) * 0.5f;
            double dx = px[1] - px[0];
            double dy = py[1] - py[0];
            width = (float)Math.Sqrt((dx * dx) + (dy * dy));
            height = 0;
            angle = (float)Math.Atan2(dy, dx);
        }
        else if (n == 1)
        {
            cx = px[0];
            cy = py[0];
        }

        // box.angle = (float)(box.angle*180/CV_PI): float · int is float, then a double division.
        angle = (float)((angle * 180f) / Math.PI);
        return new RotatedRect(cx, cy, width, height, angle);
    }

    /// <summary><c>cv2.boxPoints</c> — <c>RotatedRect::points</c>, as <c>[4, 2]</c> float32.</summary>
    public static float[] BoxPoints(RotatedRect box)
    {
        double radians = box.Angle * Math.PI / 180.0;
        float b = (float)Math.Cos(radians) * 0.5f;
        float a = (float)Math.Sin(radians) * 0.5f;
        var pt = new float[8];
        pt[0] = box.CenterX - (a * box.Height) - (b * box.Width);
        pt[1] = box.CenterY + (b * box.Height) - (a * box.Width);
        pt[2] = box.CenterX + (a * box.Height) - (b * box.Width);
        pt[3] = box.CenterY - (b * box.Height) - (a * box.Width);
        pt[4] = (2 * box.CenterX) - pt[0];
        pt[5] = (2 * box.CenterY) - pt[1];
        pt[6] = (2 * box.CenterX) - pt[2];
        pt[7] = (2 * box.CenterY) - pt[3];
        return pt;
    }

    /// <summary>
    /// <c>cv::convexHull(points, hull, clockwise=false, returnPoints=true)</c> for integer points:
    /// the indices, in output order, of the hull's vertices.
    /// </summary>
    public static int[] ConvexHull(ReadOnlySpan<int> xs, ReadOnlySpan<int> ys)
    {
        int total = xs.Length;
        if (total == 0)
        {
            return [];
        }

        // Sort by (x, y), ties by position (the pointer comparison of CHullCmpPoints).
        var pointer = new int[total];
        for (int i = 0; i < total; i++)
        {
            pointer[i] = i;
        }

        int[] xa = xs.ToArray();
        int[] ya = ys.ToArray();
        Array.Sort(pointer, (p, q) =>
        {
            if (xa[p] != xa[q])
            {
                return xa[p] < xa[q] ? -1 : 1;
            }

            if (ya[p] != ya[q])
            {
                return ya[p] < ya[q] ? -1 : 1;
            }

            return p.CompareTo(q);
        });

        int minY = 0, maxY = 0;
        for (int i = 1; i < total; i++)
        {
            int y = ya[pointer[i]];
            if (ya[pointer[minY]] > y)
            {
                minY = i;
            }

            if (ya[pointer[maxY]] < y)
            {
                maxY = i;
            }
        }

        var stack = new int[total + 2];
        var hullbuf = new int[total];
        int nout = 0;

        if (xa[pointer[0]] == xa[pointer[total - 1]] && ya[pointer[0]] == ya[pointer[total - 1]])
        {
            hullbuf[nout++] = 0;
        }
        else
        {
            // Upper half.
            int tlStart = 0;
            int tlCount = Sklansky(xa, ya, pointer, 0, maxY, stack, tlStart, -1, 1);
            int trStart = tlCount;
            int trCount = Sklansky(xa, ya, pointer, total - 1, maxY, stack, trStart, -1, -1);

            // !clockwise: swap.
            (tlStart, trStart) = (trStart, tlStart);
            (tlCount, trCount) = (trCount, tlCount);

            for (int i = 0; i < tlCount - 1; i++)
            {
                hullbuf[nout++] = pointer[stack[tlStart + i]];
            }

            for (int i = trCount - 1; i > 0; i--)
            {
                hullbuf[nout++] = pointer[stack[trStart + i]];
            }

            int stopIdx = trCount > 2 ? stack[trStart + 1] : tlCount > 2 ? stack[tlStart + tlCount - 2] : -1;

            // Lower half. The upper half's stack entries were already copied out.
            int blStart = 0;
            int blCount = Sklansky(xa, ya, pointer, 0, minY, stack, blStart, 1, -1);
            int brStart = blCount;
            int brCount = Sklansky(xa, ya, pointer, total - 1, minY, stack, brStart, 1, 1);

            if (stopIdx >= 0)
            {
                int checkIdx = blCount > 2 ? stack[blStart + 1]
                    : blCount + brCount > 2 ? stack[brStart + 2 - blCount] : -1;
                if (checkIdx == stopIdx || (checkIdx >= 0
                    && xa[pointer[checkIdx]] == xa[pointer[stopIdx]]
                    && ya[pointer[checkIdx]] == ya[pointer[stopIdx]]))
                {
                    // All points collinear: the lower half mirrors the upper one.
                    blCount = Math.Min(blCount, 2);
                    brCount = Math.Min(brCount, 2);
                }
            }

            for (int i = 0; i < blCount - 1; i++)
            {
                hullbuf[nout++] = pointer[stack[blStart + i]];
            }

            for (int i = brCount - 1; i > 0; i--)
            {
                hullbuf[nout++] = pointer[stack[brStart + i]];
            }

            // Cyclic shift so the indices ascend or descend when they nearly do.
            if (nout >= 3)
            {
                int minIdx = 0, maxIdx = 0, lt = 0;
                for (int i = 1; i < nout; i++)
                {
                    int idx = hullbuf[i];
                    lt += hullbuf[i - 1] < idx ? 1 : 0;
                    if (lt > 1 && lt <= i - 2)
                    {
                        break;
                    }

                    if (idx < hullbuf[minIdx])
                    {
                        minIdx = i;
                    }

                    if (idx > hullbuf[maxIdx])
                    {
                        maxIdx = i;
                    }
                }

                int mmdist = Math.Abs(maxIdx - minIdx);
                if ((mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt >= nout - 2))
                {
                    bool ascending = (maxIdx + 1) % nout == minIdx;
                    int i0 = ascending ? minIdx : maxIdx, j = i0;
                    if (i0 > 0)
                    {
                        int i;
                        for (i = 0; i < nout; i++)
                        {
                            int currIdx = stack[i] = hullbuf[j];
                            int nextJ = j + 1 < nout ? j + 1 : 0;
                            int nextIdx = hullbuf[nextJ];
                            if (i < nout - 1 && (ascending != (currIdx < nextIdx)))
                            {
                                break;
                            }

                            j = nextJ;
                        }

                        if (i == nout)
                        {
                            Array.Copy(stack, hullbuf, nout);
                        }
                    }
                }
            }
        }

        return hullbuf.AsSpan(0, nout).ToArray();
    }

    /// <summary><c>Sklansky_&lt;int, int64&gt;</c>; writes into <paramref name="stack"/> from <paramref name="offset"/>.</summary>
    private static int Sklansky(int[] xa, int[] ya, int[] pointer, int start, int end, int[] stack, int offset, int nsign, int sign2)
    {
        int incr = end > start ? 1 : -1;
        int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
        int stacksize = 3;

        if (start == end || (xa[pointer[start]] == xa[pointer[end]] && ya[pointer[start]] == ya[pointer[end]]))
        {
            stack[offset] = start;
            return 1;
        }

        stack[offset] = pprev;
        stack[offset + 1] = pcur;
        stack[offset + 2] = pnext;

        end += incr;

        while (pnext != end)
        {
            int cury = ya[pointer[pcur]];
            int nexty = ya[pointer[pnext]];
            int by = nexty - cury;

            if (Math.Sign(by) != nsign)
            {
                int ax = xa[pointer[pcur]] - xa[pointer[pprev]];
                int bx = xa[pointer[pnext]] - xa[pointer[pcur]];
                int ay = cury - ya[pointer[pprev]];
                long convexity = ((long)ay * bx) - ((long)ax * by);

                if (Math.Sign(convexity) == sign2 && (ax != 0 || ay != 0))
                {
                    pprev = pcur;
                    pcur = pnext;
                    pnext += incr;
                    stack[offset + stacksize] = pnext;
                    stacksize++;
                }
                else
                {
                    if (pprev == start)
                    {
                        pcur = pnext;
                        stack[offset + 1] = pcur;
                        pnext += incr;
                        stack[offset + 2] = pnext;
                    }
                    else
                    {
                        stack[offset + stacksize - 2] = pnext;
                        pcur = pprev;
                        pprev = stack[offset + stacksize - 4];
                        stacksize--;
                    }
                }
            }
            else
            {
                pnext += incr;
                stack[offset + stacksize - 1] = pnext;
            }
        }

        return --stacksize;
    }

    /// <summary>
    /// <c>rotatingCalipers(points, n, CALIPERS_MINAREARECT, out)</c>: <c>out</c> receives the
    /// corner and the two edge vectors.
    /// </summary>
    private static void RotatingCalipers(float[] px, float[] py, int n, Span<float> output)
    {
        float minArea = float.MaxValue;
        var invVectLength = new float[n];
        var vx = new float[n];
        var vy = new float[n];
        int left = 0, bottom = 0, right = 0, top = 0;
        Span<int> seq = stackalloc int[4];
        Span<float> rotX = stackalloc float[4];
        Span<float> rotY = stackalloc float[4];
        float orientation = 0;
        float baseA;
        float baseB = 0;

        float pt0x = px[0], pt0y = py[0];
        float leftX = pt0x, rightX = pt0x, topY = pt0y, bottomY = pt0y;

        for (int i = 0; i < n; i++)
        {
            if (pt0x < leftX)
            {
                leftX = pt0x;
                left = i;
            }

            if (pt0x > rightX)
            {
                rightX = pt0x;
                right = i;
            }

            if (pt0y > topY)
            {
                topY = pt0y;
                top = i;
            }

            if (pt0y < bottomY)
            {
                bottomY = pt0y;
                bottom = i;
            }

            int next = i + 1 < n ? i + 1 : 0;
            float ptx = px[next], pty = py[next];
            double dx = ptx - pt0x;
            double dy = pty - pt0y;
            vx[i] = (float)dx;
            vy[i] = (float)dy;
            invVectLength[i] = (float)(1.0 / Math.Sqrt((dx * dx) + (dy * dy)));
            pt0x = ptx;
            pt0y = pty;
        }

        // Hull orientation.
        {
            double ax = vx[n - 1];
            double ay = vy[n - 1];
            for (int i = 0; i < n; i++)
            {
                double bx = vx[i];
                double by = vy[i];
                double convexity = (ax * by) - (ay * bx);
                if (convexity != 0)
                {
                    orientation = convexity > 0 ? 1f : -1f;
                    break;
                }

                ax = bx;
                ay = by;
            }

            if (orientation == 0)
            {
                throw new InvalidOperationException("rotatingCalipers: degenerate hull");
            }
        }

        baseA = orientation;
        seq[0] = bottom;
        seq[1] = right;
        seq[2] = top;
        seq[3] = left;

        int bufLeft = 0, bufBottom = 0;
        float bufA = 0, bufWidth = 0, bufB = 0, bufHeight = 0;

        for (int k = 0; k < n; k++)
        {
            int mainElement = 0;

            // rot_vect: (a,b), rotate90CW, rotate180, rotate90CCW.
            rotX[0] = vx[seq[0]];
            rotY[0] = vy[seq[0]];
            rotX[1] = vy[seq[1]];
            rotY[1] = -vx[seq[1]];
            rotX[2] = -vx[seq[2]];
            rotY[2] = -vy[seq[2]];
            rotX[3] = -vy[seq[3]];
            rotY[3] = vx[seq[3]];
            for (int i = 1; i < 4; i++)
            {
                // firstVecIsRight(rot[i], rot[main]): rotate90CW(rot[i]) · rot[main] < 0.
                float tx = rotY[i];
                float ty = -rotX[i];
                if ((tx * rotX[mainElement]) + (ty * rotY[mainElement]) < 0)
                {
                    mainElement = i;
                }
            }

            int pindex = seq[mainElement];
            float leadX = vx[pindex] * invVectLength[pindex];
            float leadY = vy[pindex] * invVectLength[pindex];
            switch (mainElement)
            {
                case 0:
                    baseA = leadX;
                    baseB = leadY;
                    break;
                case 1:
                    baseA = leadY;
                    baseB = -leadX;
                    break;
                case 2:
                    baseA = -leadX;
                    baseB = -leadY;
                    break;
                case 3:
                    baseA = -leadY;
                    baseB = leadX;
                    break;
            }

            seq[mainElement] += 1;
            seq[mainElement] = seq[mainElement] == n ? 0 : seq[mainElement];

            float ddx = px[seq[1]] - px[seq[3]];
            float ddy = py[seq[1]] - py[seq[3]];
            float width = (ddx * baseA) + (ddy * baseB);
            ddx = px[seq[2]] - px[seq[0]];
            ddy = py[seq[2]] - py[seq[0]];
            float height = (-ddx * baseB) + (ddy * baseA);
            float area = width * height;
            if (area <= minArea)
            {
                minArea = area;
                bufLeft = seq[3];
                bufA = baseA;
                bufWidth = width;
                bufB = baseB;
                bufHeight = height;
                bufBottom = seq[0];
            }
        }

        float a1 = bufA;
        float b1 = bufB;
        float a2 = -bufB;
        float b2 = bufA;
        float c1 = (a1 * px[bufLeft]) + (py[bufLeft] * b1);
        float c2 = (a2 * px[bufBottom]) + (py[bufBottom] * b2);
        float idet = 1f / ((a1 * b2) - (a2 * b1));
        float x = ((c1 * b2) - (c2 * b1)) * idet;
        float y = ((a1 * c2) - (a2 * c1)) * idet;
        output[0] = x;
        output[1] = y;
        output[2] = a1 * bufWidth;
        output[3] = b1 * bufWidth;
        output[4] = a2 * bufHeight;
        output[5] = b2 * bufHeight;
    }

    /// <summary>
    /// <c>cv2.dilate(mask, getStructuringElement(MORPH_RECT, (k, k)))</c> with the default
    /// anchor <c>(k/2, k/2)</c> and border (which never wins the maximum):
    /// <c>dst(x, y) = max src(x + i, y + j)</c> for <c>i, j ∈ [−k/2, k − 1 − k/2]</c>.
    /// </summary>
    public static byte[] DilateRect(ReadOnlySpan<byte> mask, int width, int height, int kernel)
    {
        int anchor = kernel / 2;
        int lo = -anchor;
        int hi = kernel - 1 - anchor;
        var rows = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> src = mask.Slice(y * width, width);
            Span<byte> dst = rows.AsSpan(y * width, width);
            for (int x = 0; x < width; x++)
            {
                int x0 = Math.Max(0, x + lo);
                int x1 = Math.Min(width - 1, x + hi);
                byte m = 0;
                for (int i = x0; i <= x1 && m == 0; i++)
                {
                    m = src[i];
                }

                dst[x] = m;
            }
        }

        var result = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            int y0 = Math.Max(0, y + lo);
            int y1 = Math.Min(height - 1, y + hi);
            Span<byte> dst = result.AsSpan(y * width, width);
            for (int j = y0; j <= y1; j++)
            {
                ReadOnlySpan<byte> src = rows.AsSpan(j * width, width);
                for (int x = 0; x < width; x++)
                {
                    dst[x] |= src[x];
                }
            }
        }

        return result;
    }
}
