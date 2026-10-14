namespace NanoOcr.Imaging;

/// <summary>
/// <c>cv2.connectedComponentsWithStats(mask, connectivity=4)</c>: labels in raster order of each
/// component's first pixel, with the bounding box and area of each.
/// </summary>
public static class ConnectedComponents
{
    /// <summary>Statistics of one component.</summary>
    public readonly record struct Stats(int Left, int Top, int Width, int Height, int Area);

    /// <summary>
    /// Labels the nonzero pixels of <paramref name="mask"/> (<c>h × w</c>). Returns the label
    /// image (0 = background, components 1..n-1) and the stats per label (index 0 is the background).
    /// </summary>
    public static (int[] Labels, Stats[] Stats) Label(ReadOnlySpan<bool> mask, int w, int h)
    {
        var labels = new int[w * h];
        var parent = new List<int> { 0 };

        int Find(int x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }

            return x;
        }

        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                if (!mask[row + x])
                {
                    continue;
                }

                int left = x > 0 ? labels[row + x - 1] : 0;
                int up = y > 0 ? labels[row - w + x] : 0;
                if (left == 0 && up == 0)
                {
                    int l = parent.Count;
                    parent.Add(l);
                    labels[row + x] = l;
                }
                else if (left != 0 && up != 0)
                {
                    int a = Find(left), b = Find(up);
                    int root = Math.Min(a, b);
                    parent[a] = root;
                    parent[b] = root;
                    labels[row + x] = root;
                }
                else
                {
                    labels[row + x] = left != 0 ? left : up;
                }
            }
        }

        // Final labels follow the order of each component's smallest provisional label, which is
        // the raster order of its first pixel.
        var final = new int[parent.Count];
        int count = 1;
        for (int i = 1; i < parent.Count; i++)
        {
            int r = Find(i);
            if (r == i)
            {
                final[i] = count++;
            }
        }

        for (int i = 1; i < parent.Count; i++)
        {
            final[i] = final[Find(i)];
        }

        var minX = new int[count];
        var minY = new int[count];
        var maxX = new int[count];
        var maxY = new int[count];
        var area = new int[count];
        Array.Fill(minX, int.MaxValue);
        Array.Fill(minY, int.MaxValue);
        Array.Fill(maxX, -1);
        Array.Fill(maxY, -1);
        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                int l = labels[row + x];
                if (l != 0)
                {
                    l = final[l];
                    labels[row + x] = l;
                }

                area[l]++;
                if (x < minX[l]) { minX[l] = x; }
                if (x > maxX[l]) { maxX[l] = x; }
                if (y < minY[l]) { minY[l] = y; }
                if (y > maxY[l]) { maxY[l] = y; }
            }
        }

        var stats = new Stats[count];
        for (int l = 0; l < count; l++)
        {
            stats[l] = area[l] == 0
                ? new Stats(0, 0, 0, 0, 0)
                : new Stats(minX[l], minY[l], maxX[l] - minX[l] + 1, maxY[l] - minY[l] + 1, area[l]);
        }

        return (labels, stats);
    }
}
