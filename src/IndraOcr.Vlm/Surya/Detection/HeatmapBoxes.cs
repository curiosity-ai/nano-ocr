using IndraOcr.Imaging;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>
/// <c>surya/detection/heatmap.py</c>: from a text heatmap to line polygons (CRAFT's
/// <c>getDetBoxes_core</c>, as Surya modifies it), and <c>clean_boxes</c>
/// (<c>surya/common/util.py</c>).
/// </summary>
public static class HeatmapBoxes
{
    /// <summary>Default thresholds, <c>surya.settings</c>.</summary>
    public const float TextThreshold = 0.6f;

    /// <summary><c>DETECTOR_BLANK_THRESHOLD</c>.</summary>
    public const float BlankThreshold = 0.35f;

    /// <summary><c>DETECTOR_BOX_Y_EXPAND_MARGIN</c>.</summary>
    public const double BoxYExpandMargin = 0.05;

    /// <summary>
    /// <c>get_dynamic_thresholds</c>: scales both thresholds by
    /// <c>√clip(mean(top 10%) / 0.7, 0, 1)</c>, all in float32.
    /// </summary>
    /// <remarks>
    /// numpy takes the top tenth with <c>np.partition</c> and averages it with its pairwise float32
    /// sum. The partition's order depends on numpy's SIMD dispatch (x86-simd-sort on AVX-512), and
    /// a different order can move the float32 mean by an ulp, so this computes the correctly
    /// rounded mean instead (a float64 sum of the float32 values). That equals numpy's value on
    /// every reference page; on a page where numpy's own rounding lands an ulp away, a pixel within
    /// that ulp of a threshold would be classified differently.
    /// </remarks>
    public static (float TextThreshold, float LowText) DynamicThresholds(
        ReadOnlySpan<float> linemap, float textThreshold, float lowText, float typicalTop10Avg = 0.7f)
    {
        int length = linemap.Length;
        int top10Count = (int)(length * 0.9);
        int count = length - top10Count;
        double sum = SumOfLargest(linemap, count);
        float average = (float)(sum / count);
        float ratio = Math.Clamp(average / typicalTop10Avg, 0f, 1f);
        float scaling = MathF.Pow(ratio, 0.5f);
        float low = Math.Clamp(lowText * scaling, 0.1f, 0.6f);
        float text = Math.Clamp(textThreshold * scaling, 0.15f, 0.8f);
        return (text, low);
    }

    /// <summary>
    /// The float64 sum of the <paramref name="count"/> largest values — the multiset
    /// <c>np.partition(values, n − count)[n − count:]</c> holds — found by a two-pass radix select
    /// on the order-preserving integer image of the float bits instead of a sort.
    /// </summary>
    internal static double SumOfLargest(ReadOnlySpan<float> values, int count)
    {
        if (count <= 0)
        {
            return 0;
        }

        // Rank (from the top) the `count`-th largest key, 16 bits at a time.
        var histogram = new int[1 << 16];
        foreach (float v in values)
        {
            histogram[Key(v) >> 16]++;
        }

        int remaining = count;
        int high = (1 << 16) - 1;
        for (; high > 0 && histogram[high] < remaining; high--)
        {
            remaining -= histogram[high];
        }

        Array.Clear(histogram);
        foreach (float v in values)
        {
            uint key = Key(v);
            if (key >> 16 == (uint)high)
            {
                histogram[key & 0xFFFF]++;
            }
        }

        int low = (1 << 16) - 1;
        for (; low > 0 && histogram[low] < remaining; low--)
        {
            remaining -= histogram[low];
        }

        // Everything above the threshold key, plus `remaining` copies of the threshold itself.
        uint threshold = ((uint)high << 16) | (uint)low;
        double sum = 0;
        float thresholdValue = 0;
        foreach (float v in values)
        {
            uint key = Key(v);
            if (key > threshold)
            {
                sum += v;
            }
            else if (key == threshold)
            {
                thresholdValue = v;
            }
        }

        return sum + (remaining * (double)thresholdValue);

        static uint Key(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            return (bits & 0x8000_0000u) != 0 ? ~bits : bits | 0x8000_0000u;
        }
    }

    /// <summary>
    /// <c>detect_boxes</c>: the components of <c>linemap &gt; low_text</c> whose peak reaches the
    /// text threshold, each dilated by <c>√min(w, h)</c> and boxed by <c>minAreaRect</c>
    /// (axis-aligned when the box is nearly square), starting at the corner nearest the origin.
    /// Confidences are the peaks divided by the largest.
    /// </summary>
    /// <returns>Polygons as <c>[4, 2]</c> float32 arrays, and their confidences.</returns>
    public static (List<float[]> Boxes, List<float> Confidences) DetectBoxes(
        ReadOnlySpan<float> linemap, int width, int height, float textThreshold = TextThreshold, float lowText = BlankThreshold)
    {
        (textThreshold, lowText) = DynamicThresholds(linemap, textThreshold, lowText);

        var mask = new bool[width * height];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = linemap[i] > lowText;
        }

        (int[] labels, ConnectedComponents.Stats[] stats) = ConnectedComponents.Label(mask, width, height);

        var boxes = new List<float[]>();
        var confidences = new List<float>();
        float maxConfidence = 0;
        for (int k = 1; k < stats.Length; k++)
        {
            ConnectedComponents.Stats s = stats[k];
            if (s.Area < 10)
            {
                continue;
            }

            int x = s.Left, y = s.Top, w = s.Width, h = s.Height;
            int niter = (int)Math.Sqrt(Math.Min(w, h));
            const int Buffer = 1;
            int sx = Math.Max(0, x - niter - Buffer), sy = Math.Max(0, y - niter - Buffer);
            int ex = Math.Min(width, x + w + niter + Buffer), ey = Math.Min(height, y + h + niter + Buffer);
            int rw = ex - sx, rh = ey - sy;

            var segmap = new byte[rw * rh];
            float lineMax = float.NegativeInfinity;
            bool any = false;
            for (int yy = 0; yy < rh; yy++)
            {
                int row = (sy + yy) * width;
                for (int xx = 0; xx < rw; xx++)
                {
                    if (labels[row + sx + xx] == k)
                    {
                        segmap[(yy * rw) + xx] = 1;
                        lineMax = Math.Max(lineMax, linemap[row + sx + xx]);
                        any = true;
                    }
                }
            }

            if (!any || lineMax < textThreshold)
            {
                continue;
            }

            byte[] dilated = CvGeometry.DilateRect(segmap, rw, rh, Buffer + niter);

            // np.nonzero is row-major; the points are (x, y).
            var xs = new List<int>();
            var ys = new List<int>();
            for (int yy = 0; yy < rh; yy++)
            {
                for (int xx = 0; xx < rw; xx++)
                {
                    if (dilated[(yy * rw) + xx] != 0)
                    {
                        xs.Add(xx + sx);
                        ys.Add(yy + sy);
                    }
                }
            }

            float[] box = CvGeometry.BoxPoints(CvGeometry.MinAreaRect(
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(xs),
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ys)));

            // Align a diamond: a nearly square rotated box becomes the axis-aligned bounds.
            float bw = Norm(box[0] - box[2], box[1] - box[3]);
            float bh = Norm(box[2] - box[4], box[3] - box[5]);
            float boxRatio = Math.Max(bw, bh) / (Math.Min(bw, bh) + 1e-5f);
            if (MathF.Abs(1f - boxRatio) <= 0.1f)
            {
                float left = xs.Min(), right = xs.Max(), top = ys.Min(), bottom = ys.Max();
                box = [left, top, right, top, right, bottom, left, bottom];
            }

            // Clockwise from the corner with the smallest x + y.
            int start = 0;
            float best = box[0] + box[1];
            for (int i = 1; i < 4; i++)
            {
                float sum = box[2 * i] + box[(2 * i) + 1];
                if (sum < best)
                {
                    best = sum;
                    start = i;
                }
            }

            var rolled = new float[8];
            for (int i = 0; i < 4; i++)
            {
                int from = (i + start) % 4;
                rolled[2 * i] = box[2 * from];
                rolled[(2 * i) + 1] = box[(2 * from) + 1];
            }

            maxConfidence = Math.Max(maxConfidence, lineMax);
            confidences.Add(lineMax);
            boxes.Add(rolled);
        }

        if (maxConfidence > 0)
        {
            for (int i = 0; i < confidences.Count; i++)
            {
                confidences[i] /= maxConfidence;
            }
        }

        return (boxes, confidences);
    }

    // np.linalg.norm of a float32 pair: sqrt of the dot product, rounded once.
    private static float Norm(float dx, float dy) => (float)Math.Sqrt((float)(((double)dx * dx) + ((double)dy * dy)));

    /// <summary>
    /// <c>parallel_get_boxes</c> without the map images: detect, rescale from the heatmap's
    /// <c>(width, rows)</c> to the page, clamp, <c>clean_boxes</c>, then expand every box that is
    /// not vertical by <see cref="BoxYExpandMargin"/> of its height and clamp again.
    /// </summary>
    public static List<PolygonBox> GetBoxes(ReadOnlySpan<float> heatmap, int heatmapWidth, int heatmapHeight, int imageWidth, int imageHeight)
    {
        (List<float[]> raw, List<float> confidences) = DetectBoxes(heatmap, heatmapWidth, heatmapHeight);
        var boxes = new List<PolygonBox>(raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            float[] b = raw[i];
            var polygon = new double[4][];
            for (int c = 0; c < 4; c++)
            {
                polygon[c] = [b[2 * c], b[(2 * c) + 1]];
            }

            var box = new PolygonBox(polygon, confidences[i]);
            box.Rescale((heatmapWidth, heatmapHeight), (imageWidth, imageHeight));
            box.FitToBounds(0, 0, imageWidth, imageHeight);
            boxes.Add(box);
        }

        boxes = CleanBoxes(boxes);
        foreach (PolygonBox box in boxes)
        {
            // Skip vertical boxes.
            if (box.Height < 3 * box.Width)
            {
                box.Expand(0, BoxYExpandMargin);
                box.FitToBounds(0, 0, imageWidth, imageHeight);
            }
        }

        return boxes;
    }

    /// <summary>
    /// <c>clean_boxes</c>: drops degenerate boxes and boxes whose bbox lies inside another
    /// (different) one's.
    /// </summary>
    public static List<PolygonBox> CleanBoxes(List<PolygonBox> boxes)
    {
        var result = new List<PolygonBox>();
        double[][] bboxes = boxes.Select(b => b.Bbox).ToArray();
        for (int i = 0; i < boxes.Count; i++)
        {
            double[] box = bboxes[i];
            if (box[2] == box[0] || box[3] == box[1])
            {
                continue;
            }

            bool contained = false;
            for (int j = 0; j < boxes.Count; j++)
            {
                if (SamePolygon(boxes[j], boxes[i]))
                {
                    continue;
                }

                double[] other = bboxes[j];
                if (box.AsSpan().SequenceEqual(other))
                {
                    continue;
                }

                if (box[0] >= other[0] && box[1] >= other[1] && box[2] <= other[2] && box[3] <= other[3])
                {
                    contained = true;
                    break;
                }
            }

            if (!contained)
            {
                result.Add(boxes[i]);
            }
        }

        return result;
    }

    private static bool SamePolygon(PolygonBox a, PolygonBox b)
    {
        for (int i = 0; i < 4; i++)
        {
            if (a.Polygon[i][0] != b.Polygon[i][0] || a.Polygon[i][1] != b.Polygon[i][1])
            {
                return false;
            }
        }

        return true;
    }
}
