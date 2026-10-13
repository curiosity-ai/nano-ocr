using System.Buffers.Binary;
using System.IO.Compression;

namespace IndraOcr.Imaging;

/// <summary>
/// A baseline TIFF reader for scanned documents, which SkiaSharp does not decode: every page
/// (IFD) of a multi-page file, as grayscale — what <c>cv2.imreadmulti(path, IMREAD_GRAYSCALE)</c>
/// gives the Python pipeline.
/// </summary>
/// <remarks>
/// Covers strips and tiles; no compression, PackBits, LZW (with horizontal predictor), Deflate,
/// and CCITT Group 3 (1-D) and Group 4 fax; bilevel, 4/8/16-bit grayscale, palette, RGB and RGBA.
/// Colour is converted with OpenCV's <c>cvtColor</c> weights (the 14-bit 4899/9617/1868).
/// </remarks>
public static class TiffDecoder
{
    /// <summary>Whether the bytes start with a TIFF header.</summary>
    public static bool IsTiff(ReadOnlySpan<byte> b) =>
        b.Length >= 4 && ((b[0] == 'I' && b[1] == 'I' && b[2] == 42 && b[3] == 0) || (b[0] == 'M' && b[1] == 'M' && b[2] == 0 && b[3] == 42));

    /// <summary>Decodes every page of a TIFF file.</summary>
    public static List<GrayImage> Decode(byte[] data)
    {
        var pages = new List<GrayImage>();
        if (!IsTiff(data))
        {
            return pages;
        }

        bool le = data[0] == 'I';
        uint offset = U32(data, 4, le);
        var seen = new HashSet<uint>();
        while (offset != 0 && offset + 2 <= data.Length && seen.Add(offset))
        {
            var tags = new Dictionary<int, uint[]>();
            int count = U16(data, (int)offset, le);
            for (int i = 0; i < count; i++)
            {
                int e = (int)offset + 2 + (i * 12);
                if (e + 12 > data.Length)
                {
                    break;
                }

                int tag = U16(data, e, le), type = U16(data, e + 2, le);
                uint n = U32(data, e + 4, le);
                int size = type switch { 1 or 2 or 6 or 7 => 1, 3 or 8 => 2, 4 or 9 or 11 => 4, 5 or 10 or 12 => 8, _ => 1 };
                long bytes = (long)size * n;
                int at = bytes <= 4 ? e + 8 : (int)U32(data, e + 8, le);
                var values = new uint[Math.Min(n, 1u << 24)];
                for (int k = 0; k < values.Length && at + ((k + 1) * size) <= data.Length; k++)
                {
                    values[k] = size switch
                    {
                        1 => (uint)data[at + k],
                        2 => (uint)U16(data, at + (k * 2), le),
                        8 => U32(data, at + (k * 8), le),
                        _ => U32(data, at + (k * 4), le),
                    };
                }

                tags[tag] = values;
            }

            offset = (int)offset + 2 + (count * 12) + 4 <= data.Length ? U32(data, (int)offset + 2 + (count * 12), le) : 0;
            try
            {
                GrayImage? page = DecodePage(data, tags, le);
                if (page is not null)
                {
                    pages.Add(page);
                }
            }
            catch (Exception e) when (e is IndexOutOfRangeException or ArgumentException or InvalidDataException or NotSupportedException)
            {
                // A page that cannot be decoded is skipped, as OpenCV skips frames it cannot read.
            }
        }

        return pages;
    }

    private static uint Tag(Dictionary<int, uint[]> tags, int tag, uint fallback) =>
        tags.TryGetValue(tag, out uint[]? v) && v.Length > 0 ? v[0] : fallback;

    private static GrayImage? DecodePage(byte[] data, Dictionary<int, uint[]> tags, bool le)
    {
        int width = (int)Tag(tags, 256, 0), height = (int)Tag(tags, 257, 0);
        if (width <= 0 || height <= 0)
        {
            return null;
        }

        int bits = (int)Tag(tags, 258, 1);
        int compression = (int)Tag(tags, 259, 1);
        int photometric = (int)Tag(tags, 262, 1);
        int spp = (int)Tag(tags, 277, 1);
        int planar = (int)Tag(tags, 284, 1);
        int predictor = (int)Tag(tags, 317, 1);
        int fillOrder = (int)Tag(tags, 266, 1);
        uint t4Options = Tag(tags, 292, 0);
        uint[]? colorMap = tags.GetValueOrDefault(320);
        if (planar != 1 && spp > 1)
        {
            throw new NotSupportedException("planar TIFF");
        }

        int bitsPerPixel = bits * spp;
        bool tiled = tags.ContainsKey(322);
        int tileW = tiled ? (int)Tag(tags, 322, 0) : width;
        int tileH = tiled ? (int)Tag(tags, 323, 0) : (int)Math.Min(Tag(tags, 278, (uint)height), (uint)height);
        uint[] offsets = tags.GetValueOrDefault(tiled ? 324 : 273) ?? [];
        uint[] counts = tags.GetValueOrDefault(tiled ? 325 : 279) ?? [];
        int rowBytes = ((tileW * bitsPerPixel) + 7) / 8;
        int fullRowBytes = ((width * bitsPerPixel) + 7) / 8;
        var raw = new byte[fullRowBytes * height];
        int across = tiled ? (width + tileW - 1) / tileW : 1;
        for (int s = 0; s < offsets.Length; s++)
        {
            int start = (int)offsets[s];
            int len = s < counts.Length ? (int)Math.Min(counts[s], (uint)(data.Length - start)) : data.Length - start;
            if (start < 0 || start >= data.Length || len <= 0)
            {
                continue;
            }

            int ty = tiled ? s / across * tileH : s * tileH;
            int tx = tiled ? s % across * tileW : 0;
            int rows = tiled ? tileH : Math.Min(tileH, height - ty);
            if (rows <= 0)
            {
                continue;
            }

            byte[] block = Decompress(data.AsSpan(start, len), compression, rowBytes * rows, tileW, rows, fillOrder, t4Options, photometric);
            if (predictor == 2)
            {
                UndoPredictor(block, rowBytes, rows, spp, bits);
            }

            for (int y = 0; y < rows && ty + y < height; y++)
            {
                if (tiled)
                {
                    int colBytes = (tx * bitsPerPixel) / 8;
                    int copy = Math.Min(rowBytes, fullRowBytes - colBytes);
                    Array.Copy(block, y * rowBytes, raw, ((ty + y) * fullRowBytes) + colBytes, Math.Max(0, Math.Min(copy, block.Length - (y * rowBytes))));
                }
                else
                {
                    Array.Copy(block, y * rowBytes, raw, (ty + y) * fullRowBytes, Math.Min(rowBytes, Math.Max(0, block.Length - (y * rowBytes))));
                }
            }
        }

        return ToGray(raw, width, height, fullRowBytes, bits, spp, photometric, colorMap, le, compression is 3 or 4);
    }

    private static byte[] Decompress(ReadOnlySpan<byte> src, int compression, int expected, int width, int rows, int fillOrder, uint t4Options, int photometric)
    {
        switch (compression)
        {
            case 1:
                return src.ToArray();
            case 32773:
                return PackBits(src, expected);
            case 5:
                return Lzw(src, expected);
            case 8 or 32946:
                {
                    using var input = new MemoryStream(src.ToArray());
                    using var z = new ZLibStream(input, CompressionMode.Decompress);
                    var output = new byte[expected];
                    int total = 0;
                    while (total < expected)
                    {
                        int r = z.Read(output, total, expected - total);
                        if (r == 0)
                        {
                            break;
                        }

                        total += r;
                    }

                    return output;
                }

            case 2 or 3 or 4:
                {
                    byte[] bytes = src.ToArray();
                    if (fillOrder == 2)
                    {
                        for (int i = 0; i < bytes.Length; i++)
                        {
                            bytes[i] = ReverseBits(bytes[i]);
                        }
                    }

                    return Ccitt.Decode(bytes, width, rows, compression == 4 ? Ccitt.Mode.G4 : compression == 2 ? Ccitt.Mode.Modified : (t4Options & 1) != 0 ? Ccitt.Mode.G3TwoD : Ccitt.Mode.G3);
                }

            default:
                throw new NotSupportedException($"TIFF compression {compression}");
        }
    }

    private static byte ReverseBits(byte b)
    {
        b = (byte)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4));
        b = (byte)(((b & 0xCC) >> 2) | ((b & 0x33) << 2));
        return (byte)(((b & 0xAA) >> 1) | ((b & 0x55) << 1));
    }

    private static byte[] PackBits(ReadOnlySpan<byte> src, int expected)
    {
        var dst = new List<byte>(expected);
        int i = 0;
        while (i < src.Length && dst.Count < expected)
        {
            sbyte n = (sbyte)src[i++];
            if (n >= 0)
            {
                for (int k = 0; k <= n && i < src.Length; k++)
                {
                    dst.Add(src[i++]);
                }
            }
            else if (n != -128 && i < src.Length)
            {
                byte v = src[i++];
                for (int k = 0; k < 1 - n; k++)
                {
                    dst.Add(v);
                }
            }
        }

        return [.. dst];
    }

    private static byte[] Lzw(ReadOnlySpan<byte> src, int expected)
    {
        var output = new List<byte>(expected);
        var table = new List<byte[]>(4096);

        void Reset()
        {
            table.Clear();
            for (int i = 0; i < 256; i++)
            {
                table.Add([(byte)i]);
            }

            table.Add([]);
            table.Add([]);
        }

        Reset();
        int bitPos = 0, codeLen = 9;
        byte[]? prev = null;
        while (true)
        {
            if (bitPos + codeLen > src.Length * 8)
            {
                break;
            }

            int code = 0;
            for (int k = 0; k < codeLen; k++)
            {
                int p = bitPos + k;
                code = (code << 1) | ((src[p >> 3] >> (7 - (p & 7))) & 1);
            }

            bitPos += codeLen;
            if (code == 257)
            {
                break;
            }

            if (code == 256)
            {
                Reset();
                codeLen = 9;
                prev = null;
                continue;
            }

            byte[] entry;
            if (code < table.Count)
            {
                entry = table[code];
                if (prev is not null)
                {
                    table.Add([.. prev, entry[0]]);
                }
            }
            else if (prev is not null)
            {
                entry = [.. prev, prev[0]];
                table.Add(entry);
            }
            else
            {
                break;
            }

            output.AddRange(entry);
            prev = entry;
            if (table.Count + 1 >= (1 << codeLen) && codeLen < 12)
            {
                codeLen++;
            }
        }

        return [.. output];
    }

    private static void UndoPredictor(byte[] block, int rowBytes, int rows, int spp, int bits)
    {
        if (bits == 8)
        {
            for (int y = 0; y < rows; y++)
            {
                int o = y * rowBytes;
                for (int x = spp; x < rowBytes && o + x < block.Length; x++)
                {
                    block[o + x] = (byte)(block[o + x] + block[o + x - spp]);
                }
            }
        }
    }

    private static GrayImage ToGray(byte[] raw, int w, int h, int rowBytes, int bits, int spp, int photometric, uint[]? colorMap, bool le, bool fax)
    {
        var g = new GrayImage(w, h);
        byte[] d = g.Pixels;
        for (int y = 0; y < h; y++)
        {
            int row = y * rowBytes;
            for (int x = 0; x < w; x++)
            {
                byte v;
                if (spp >= 3 && photometric == 2)
                {
                    int bpp = bits / 8 * spp;
                    int o = row + (x * bpp);
                    int r = bits == 16 ? raw[o + (le ? 1 : 0)] : raw[o];
                    int gr = bits == 16 ? raw[o + 2 + (le ? 1 : 0)] : raw[o + 1];
                    int b = bits == 16 ? raw[o + 4 + (le ? 1 : 0)] : raw[o + 2];
                    v = (byte)(((r * 4899) + (gr * 9617) + (b * 1868) + 8192) >> 14);
                }
                else
                {
                    int sample = Sample(raw, row, x * spp, bits, le);
                    if (photometric == 3 && colorMap is not null)
                    {
                        int n = 1 << bits;
                        int r = (int)(colorMap[sample] >> 8), gr = (int)(colorMap[n + sample] >> 8), b = (int)(colorMap[(2 * n) + sample] >> 8);
                        v = (byte)(((r * 4899) + (gr * 9617) + (b * 1868) + 8192) >> 14);
                    }
                    else
                    {
                        int max = (1 << Math.Min(bits, 8)) - 1;
                        int s8 = bits == 16 ? sample >> 8 : bits >= 8 ? sample : sample * 255 / max;
                        // Decoded fax data is "1 = black" in the usual WhiteIsZero files; the
                        // photometric tag decides, for fax as for everything else.
                        bool whiteIsZero = photometric == 0;
                        v = (byte)(whiteIsZero ? 255 - s8 : s8);
                    }
                }

                d[(y * w) + x] = v;
            }
        }

        return g;
    }

    private static int Sample(byte[] raw, int row, int index, int bits, bool le)
    {
        switch (bits)
        {
            case 8:
                return raw[row + index];
            case 16:
                return le ? raw[row + (index * 2)] | (raw[row + (index * 2) + 1] << 8) : (raw[row + (index * 2)] << 8) | raw[row + (index * 2) + 1];
            default:
                int bit = index * bits;
                int b = raw[row + (bit >> 3)];
                int shift = 8 - bits - (bit & 7);
                return (b >> shift) & ((1 << bits) - 1);
        }
    }

    private static int U16(byte[] d, int o, bool le) => le ? BinaryPrimitives.ReadUInt16LittleEndian(d.AsSpan(o)) : BinaryPrimitives.ReadUInt16BigEndian(d.AsSpan(o));

    private static uint U32(byte[] d, int o, bool le) => le ? BinaryPrimitives.ReadUInt32LittleEndian(d.AsSpan(o)) : BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(o));
}
