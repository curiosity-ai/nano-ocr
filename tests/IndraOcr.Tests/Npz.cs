using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using IndraOcr.Onnx;

namespace IndraOcr.Tests;

/// <summary>Reads numpy <c>.npz</c> archives (zips of <c>.npy</c> files) into tensors.</summary>
public static partial class Npz
{
    public static Dictionary<string, Tensor> Load(string path)
    {
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        using ZipArchive zip = ZipFile.OpenRead(path);
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            string name = entry.FullName.EndsWith(".npy", StringComparison.Ordinal) ? entry.FullName[..^4] : entry.FullName;
            using Stream s = entry.Open();
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            result[name] = Npy(ms.ToArray());
        }

        return result;
    }

    public static Tensor Npy(byte[] bytes)
    {
        if (bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
        {
            throw new InvalidDataException("Not an npy file.");
        }

        int major = bytes[6];
        int headerLen = major == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8);
        int start = major == 1 ? 10 : 12;
        string header = Encoding.ASCII.GetString(bytes, start, headerLen);
        int data = start + headerLen;
        string descr = Descr().Match(header).Groups[1].Value;
        if (header.Contains("'fortran_order': True", StringComparison.Ordinal))
        {
            throw new NotSupportedException("Fortran-order arrays.");
        }

        string shapeText = Shape().Match(header).Groups[1].Value;
        int[] shape = shapeText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        int n = Tensor.SizeOf(shape);
        switch (descr)
        {
            case "<f4":
                {
                    var a = new float[Math.Max(n, 1)];
                    Buffer.BlockCopy(bytes, data, a, 0, n * 4);
                    return Tensor.FromFloats(a, shape);
                }

            case "<f8":
                {
                    var a = new float[Math.Max(n, 1)];
                    for (int i = 0; i < n; i++)
                    {
                        a[i] = (float)BitConverter.ToDouble(bytes, data + (i * 8));
                    }

                    return Tensor.FromFloats(a, shape);
                }

            case "<i8":
                {
                    var a = new long[Math.Max(n, 1)];
                    Buffer.BlockCopy(bytes, data, a, 0, n * 8);
                    return Tensor.FromLongs(a, shape);
                }

            case "<i4":
                {
                    var a = new long[Math.Max(n, 1)];
                    for (int i = 0; i < n; i++)
                    {
                        a[i] = BitConverter.ToInt32(bytes, data + (i * 4));
                    }

                    return Tensor.FromLongs(a, shape);
                }

            case "|u1":
                {
                    var a = new long[Math.Max(n, 1)];
                    for (int i = 0; i < n; i++)
                    {
                        a[i] = bytes[data + i];
                    }

                    return Tensor.FromLongs(a, shape);
                }

            case "|b1":
                {
                    var a = new bool[Math.Max(n, 1)];
                    for (int i = 0; i < n; i++)
                    {
                        a[i] = bytes[data + i] != 0;
                    }

                    return Tensor.FromBools(a, shape);
                }

            default:
                throw new NotSupportedException($"npy dtype {descr}.");
        }
    }

    [GeneratedRegex(@"'descr':\s*'([^']+)'")]
    private static partial Regex Descr();

    [GeneratedRegex(@"'shape':\s*\(([^)]*)\)")]
    private static partial Regex Shape();
}
