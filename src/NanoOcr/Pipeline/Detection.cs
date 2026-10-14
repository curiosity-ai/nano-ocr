using NanoOcr.Imaging;

namespace NanoOcr.Pipeline;

/// <summary>A line box in page pixels: <c>x0, y0, x1, y1</c> and the detector's mean core score.</summary>
public struct LineBox
{
    /// <summary>Left edge.</summary>
    public float X0;

    /// <summary>Top edge.</summary>
    public float Y0;

    /// <summary>Right edge.</summary>
    public float X1;

    /// <summary>Bottom edge.</summary>
    public float Y1;

    /// <summary>Detector score (mean core probability), or 1 for derived boxes.</summary>
    public float Score;

    /// <summary>Creates a box.</summary>
    public LineBox(float x0, float y0, float x1, float y1, float score = 1f)
    {
        X0 = x0;
        Y0 = y0;
        X1 = x1;
        Y1 = y1;
        Score = score;
    }

    /// <summary>Height (<c>y1 - y0</c>) in float32, as numpy computes it.</summary>
    public readonly float Height => Y1 - Y0;

    /// <inheritdoc />
    public override readonly string ToString() => $"[{X0}, {Y0}, {X1}, {Y1}]";
}

/// <summary>
/// The line detector's pre- and post-processing (port of <c>indra_ocr/det_post.py</c>).
/// </summary>
public static class Detection
{
    /// <summary>Core band height as a fraction of the line ink height.</summary>
    public const float CoreFrac = 0.4f;

    /// <summary>The core is shorter than the line by this fraction of its height at each end.</summary>
    public const double CoreXShrink = 0.15;

    /// <summary>The detector's output stride.</summary>
    public const int OutStride = 2;

    /// <summary>Input height and width must divide by this.</summary>
    public const int InputMultiple = 32;

    /// <summary>
    /// <c>page_to_input</c>: the page resized so its long side is <paramref name="longSide"/>
    /// (area when shrinking, linear when enlarging), as an ink map (1 = ink) padded with paper to
    /// multiples of 32.
    /// </summary>
    public static (float[] Ink, int Height, int Width, double Scale) PageToInput(GrayImage gray, int longSide)
    {
        int h = gray.Height, w = gray.Width;
        double scale = (double)longSide / Math.Max(h, w);
        int nh = (int)Math.Max(1, Py.Round(h * scale)), nw = (int)Math.Max(1, Py.Round(w * scale));
        GrayImage img = gray;
        if (nh != h || nw != w)
        {
            img = CvResize.Resize(gray, nw, nh, scale < 1 ? Interpolation.Area : Interpolation.Linear);
        }

        int ph = (nh + InputMultiple - 1) / InputMultiple * InputMultiple;
        int pw = (nw + InputMultiple - 1) / InputMultiple * InputMultiple;
        var ink = new float[ph * pw];
        InkMap(img, ink, pw);
        return (ink, ph, pw, scale);
    }

    /// <summary>Writes <c>1 - gray / 255</c> (float32) into the top-left of a buffer with row stride <paramref name="stride"/>.</summary>
    internal static void InkMap(GrayImage img, Span<float> dst, int stride)
    {
        Span<float> lut = stackalloc float[256];
        for (int v = 0; v < 256; v++)
        {
            lut[v] = 1f - (v / 255f);
        }

        for (int y = 0; y < img.Height; y++)
        {
            ReadOnlySpan<byte> row = img.Pixels.AsSpan(y * img.Width, img.Width);
            Span<float> o = dst.Slice(y * stride, img.Width);
            for (int x = 0; x < row.Length; x++)
            {
                o[x] = lut[row[x]];
            }
        }
    }

    /// <summary>
    /// <c>decode</c>: line boxes (input pixels) from the detector output <c>[3, h, w]</c> (core
    /// logit, distance to the top, distance to the bottom). Each 4-connected component of the
    /// thresholded core is one line; its top and bottom are the mean of the per-pixel distances.
    /// </summary>
    public static List<LineBox> Decode(ReadOnlySpan<float> output, int h, int w, float threshold = 0.5f, int minArea = 2, double growY = 0, double growX = 0)
    {
        int n = h * w;
        ReadOnlySpan<float> logit = output[..n], dtop = output.Slice(n, n), dbot = output.Slice(2 * n, n);
        var prob = new float[n];
        var mask = new bool[n];
        for (int i = 0; i < n; i++)
        {
            prob[i] = 1f / (1f + MathF.Exp(-logit[i]));
            mask[i] = prob[i] > threshold;
        }

        (int[] labels, ConnectedComponents.Stats[] stats) = ConnectedComponents.Label(mask, w, h);
        int count = stats.Length;
        var boxes = new List<LineBox>();
        if (count <= 1)
        {
            return boxes;
        }

        var sumTop = new double[count];
        var sumBot = new double[count];
        var sumScore = new double[count];
        var cnt = new int[count];
        for (int y = 0; y < h; y++)
        {
            float ys = (y + 0.5f) * OutStride;
            for (int x = 0; x < w; x++)
            {
                int i = (y * w) + x;
                int l = labels[i];
                if (l == 0)
                {
                    continue;
                }

                sumTop[l] += ys - dtop[i];
                sumBot[l] += ys + dbot[i];
                sumScore[l] += prob[i];
                cnt[l]++;
            }
        }

        for (int l = 1; l < count; l++)
        {
            if (stats[l].Area < minArea)
            {
                continue;
            }

            double c = Math.Max((float)cnt[l], 1f);
            double y0 = sumTop[l] / c, y1 = sumBot[l] / c, score = sumScore[l] / c;
            double lh = Math.Max(1.0, y1 - y0);
            double ex = Math.Min(CoreXShrink * lh, 0.5 * lh);
            double x0 = (stats[l].Left * OutStride) - ex;
            double x1 = ((stats[l].Left + stats[l].Width) * OutStride) + ex;
            boxes.Add(new LineBox(
                (float)(x0 - (growX * lh)),
                (float)(y0 - (growY * lh)),
                (float)(x1 + (growX * lh)),
                (float)(y1 + (growY * lh)),
                (float)score));
        }

        return boxes;
    }
}
