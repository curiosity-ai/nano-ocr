namespace NanoOcr.Imaging;

/// <summary>
/// CCITT fax decoding (ITU-T T.4 and T.6) for TIFF compressions 2, 3 and 4 — the bilevel
/// compressions scanned documents mostly use. Output rows are packed MSB-first with 1 = black.
/// </summary>
internal static class Ccitt
{
    public enum Mode { Modified, G3, G3TwoD, G4 }

    private static readonly Dictionary<(int Len, int Code), int> White = Build(true);
    private static readonly Dictionary<(int Len, int Code), int> Black = Build(false);

    private static Dictionary<(int, int), int> Build(bool white)
    {
        string[] term = white
            ? ["00110101", "000111", "0111", "1000", "1011", "1100", "1110", "1111", "10011", "10100", "00111", "01000", "001000", "000011", "110100", "110101",
               "101010", "101011", "0100111", "0001100", "0001000", "0010111", "0000011", "0000100", "0101000", "0101011", "0010011", "0100100", "0011000", "00000010", "00000011", "00011010",
               "00011011", "00010010", "00010011", "00010100", "00010101", "00010110", "00010111", "00101000", "00101001", "00101010", "00101011", "00101100", "00101101", "00000100", "00000101", "00001010",
               "00001011", "01010010", "01010011", "01010100", "01010101", "00100100", "00100101", "01011000", "01011001", "01011010", "01011011", "01001010", "01001011", "00110010", "00110011", "00110100"]
            : ["0000110111", "010", "11", "10", "011", "0011", "0010", "00011", "000101", "000100", "0000100", "0000101", "0000111", "00000100", "00000111", "000011000",
               "0000010111", "0000011000", "0000001000", "00001100111", "00001101000", "00001101100", "00000110111", "00000101000", "00000010111", "00000011000", "000011001010", "000011001011", "000011001100", "000011001101", "000001101000", "000001101001",
               "000001101010", "000001101011", "000011010010", "000011010011", "000011010100", "000011010101", "000011010110", "000011010111", "000001101100", "000001101101", "000011011010", "000011011011", "000001010100", "000001010101", "000001010110", "000001010111",
               "000001100100", "000001100101", "000001010010", "000001010011", "000000100100", "000000110111", "000000111000", "000000100111", "000000101000", "000001011000", "000001011001", "000000101011", "000000101100", "000001011010", "000001100110", "000001100111"];
        string[] makeup = white
            ? ["11011", "10010", "010111", "0110111", "00110110", "00110111", "01100100", "01100101", "01101000", "01100111", "011001100", "011001101", "011010010", "011010011",
               "011010100", "011010101", "011010110", "011010111", "011011000", "011011001", "011011010", "011011011", "010011000", "010011001", "010011010", "011000", "010011011"]
            : ["0000001111", "000011001000", "000011001001", "000001011011", "000000110011", "000000110100", "000000110101", "0000001101100", "0000001101101", "0000001001010", "0000001001011", "0000001001100", "0000001001101", "0000001110010",
               "0000001110011", "0000001110100", "0000001110101", "0000001110110", "0000001110111", "0000001010010", "0000001010011", "0000001010100", "0000001010101", "0000001011010", "0000001011011", "0000001100100", "0000001100101"];
        string[] extended = ["00000001000", "00000001100", "00000001101", "000000010010", "000000010011", "000000010100", "000000010101", "000000010110", "000000010111", "000000011100", "000000011101", "000000011110", "000000011111"];
        var d = new Dictionary<(int, int), int>();
        for (int i = 0; i < term.Length; i++)
        {
            d[(term[i].Length, Convert.ToInt32(term[i], 2))] = i;
        }

        for (int i = 0; i < makeup.Length; i++)
        {
            d[(makeup[i].Length, Convert.ToInt32(makeup[i], 2))] = (i + 1) * 64;
        }

        for (int i = 0; i < extended.Length; i++)
        {
            d[(extended[i].Length, Convert.ToInt32(extended[i], 2))] = 1792 + (i * 64);
        }

        return d;
    }

    private sealed class Bits(byte[] data)
    {
        public int Pos;

        public int Peek(int n)
        {
            int v = 0;
            for (int k = 0; k < n; k++)
            {
                int p = Pos + k;
                int bit = p >> 3 < data.Length ? (data[p >> 3] >> (7 - (p & 7))) & 1 : 0;
                v = (v << 1) | bit;
            }

            return v;
        }

        public int Read(int n)
        {
            int v = Peek(n);
            Pos += n;
            return v;
        }

        public bool End => Pos >= data.Length * 8;

        public void Align() => Pos = (Pos + 7) & ~7;
    }

    private static int Run(Bits b, bool white)
    {
        Dictionary<(int, int), int> table = white ? White : Black;
        int total = 0;
        while (!b.End)
        {
            int value = -1;
            for (int len = 2; len <= 13; len++)
            {
                if (table.TryGetValue((len, b.Peek(len)), out value))
                {
                    b.Pos += len;
                    break;
                }

                value = -1;
            }

            if (value < 0)
            {
                throw new InvalidDataException("bad CCITT run code");
            }

            total += value;
            if (value < 64)
            {
                return total;
            }
        }

        return total;
    }

    /// <summary>Skips fill bits and one EOL (000000000001) if present; returns whether one was found.</summary>
    private static bool SkipEol(Bits b)
    {
        int start = b.Pos;
        int zeros = 0;
        while (!b.End && b.Peek(1) == 0)
        {
            b.Pos++;
            zeros++;
        }

        if (zeros >= 11 && !b.End)
        {
            b.Pos++;
            return true;
        }

        b.Pos = start;
        return false;
    }

    public static byte[] Decode(byte[] data, int width, int rows, Mode mode)
    {
        int rowBytes = (width + 7) / 8;
        var output = new byte[rowBytes * rows];
        var bits = new Bits(data);
        var reference = new List<int>();
        var current = new List<int>();
        for (int y = 0; y < rows && !bits.End; y++)
        {
            bool twoD = mode == Mode.G4;
            if (mode is Mode.G3 or Mode.G3TwoD)
            {
                SkipEol(bits);
                if (mode == Mode.G3TwoD)
                {
                    twoD = bits.Read(1) == 0;
                }
            }

            current.Clear();
            try
            {
                if (twoD)
                {
                    DecodeTwoD(bits, width, reference, current);
                }
                else
                {
                    DecodeOneD(bits, width, current);
                }
            }
            catch (InvalidDataException)
            {
                break;
            }

            if (mode == Mode.Modified)
            {
                bits.Align();
            }

            Fill(output.AsSpan(y * rowBytes, rowBytes), current, width);
            (reference, current) = (current, reference);
        }

        return output;
    }

    private static void DecodeOneD(Bits b, int width, List<int> changes)
    {
        int a0 = 0;
        bool white = true;
        while (a0 < width)
        {
            int run = Run(b, white);
            a0 += run;
            changes.Add(Math.Min(a0, width));
            white = !white;
        }
    }

    private static void DecodeTwoD(Bits b, int width, List<int> reference, List<int> changes)
    {
        int a0 = -1;
        bool white = true;
        while (a0 < width)
        {
            // b1: the first change on the reference line right of a0 to the colour opposite a0's.
            int i = 0;
            while (i < reference.Count && (reference[i] <= a0 || (i % 2 == 0) != white))
            {
                i++;
            }

            int b1 = i < reference.Count ? reference[i] : width;
            int b2 = i + 1 < reference.Count ? reference[i + 1] : width;
            if (b.Peek(1) == 1)
            {
                b.Pos += 1;
                Vertical(0);
            }
            else if (b.Peek(3) == 0b011)
            {
                b.Pos += 3;
                Vertical(1);
            }
            else if (b.Peek(3) == 0b010)
            {
                b.Pos += 3;
                Vertical(-1);
            }
            else if (b.Peek(3) == 0b001)
            {
                b.Pos += 3;
                int start = Math.Max(a0, 0);
                int r1 = Run(b, white);
                int r2 = Run(b, !white);
                int a1 = start + r1, a2 = a1 + r2;
                changes.Add(Math.Min(a1, width));
                changes.Add(Math.Min(a2, width));
                a0 = a2;
            }
            else if (b.Peek(4) == 0b0001)
            {
                b.Pos += 4;
                a0 = b2;
            }
            else if (b.Peek(6) == 0b000011)
            {
                b.Pos += 6;
                Vertical(2);
            }
            else if (b.Peek(6) == 0b000010)
            {
                b.Pos += 6;
                Vertical(-2);
            }
            else if (b.Peek(7) == 0b0000011)
            {
                b.Pos += 7;
                Vertical(3);
            }
            else if (b.Peek(7) == 0b0000010)
            {
                b.Pos += 7;
                Vertical(-3);
            }
            else
            {
                throw new InvalidDataException("unsupported CCITT 2-D code");
            }

            void Vertical(int delta)
            {
                int a1 = Math.Clamp(b1 + delta, 0, width);
                changes.Add(a1);
                a0 = a1;
                white = !white;
            }
        }
    }

    /// <summary>Renders a row from its changing elements (starting white), 1 = black.</summary>
    private static void Fill(Span<byte> row, List<int> changes, int width)
    {
        row.Clear();
        bool black = false;
        int x = 0;
        foreach (int c in changes)
        {
            int end = Math.Min(c, width);
            if (black)
            {
                for (int p = x; p < end; p++)
                {
                    row[p >> 3] |= (byte)(0x80 >> (p & 7));
                }
            }

            x = Math.Max(x, end);
            black = !black;
        }

        if (black)
        {
            for (int p = x; p < width; p++)
            {
                row[p >> 3] |= (byte)(0x80 >> (p & 7));
            }
        }
    }
}
