using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Onnx;
using Xunit;

namespace IndraOcr.Tests;

/// <summary>
/// The OpenCV ports against OpenCV itself (fixtures from <c>tools/reference/dump_imaging.py</c>):
/// every operation must reproduce OpenCV's output exactly, to the byte (or the float).
/// </summary>
public class ImagingParityTests(ITestOutputHelper output)
{
    private static string Dir => Path.Combine(TestData.Root, "tests", "data", "reference", "imaging");

    public static TheoryData<string> Cases()
    {
        var data = new TheoryData<string>();
        string json = Path.Combine(Dir, "imaging.json");
        if (!File.Exists(json))
        {
            data.Add("(missing)");
            return data;
        }

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(json));
        foreach (JsonElement c in doc.RootElement.EnumerateArray())
        {
            data.Add(c.GetProperty("name").GetString()!);
        }

        return data;
    }

    private static readonly Lazy<Dictionary<string, Tensor>> Arrays = new(() => Npz.Load(Path.Combine(Dir, "imaging.npz")));

    private static GrayImage Gray(Tensor t)
    {
        long[] v = t.ToLongArray();
        return new GrayImage(t.Shape[1], t.Shape[0], v.Select(x => (byte)x).ToArray());
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesOpenCv(string name)
    {
        Assert.SkipWhen(name == "(missing)", "Imaging fixtures missing; run tools/reference/dump_imaging.py.");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "imaging.json")));
        JsonElement c = doc.RootElement.EnumerateArray().First(e => e.GetProperty("name").GetString() == name);
        string op = c.GetProperty("op").GetString()!;
        string input = c.GetProperty("input").GetString()!;
        JsonElement args = c.GetProperty("args");
        Dictionary<string, Tensor> a = Arrays.Value;
        switch (op)
        {
            case "resize":
                {
                    GrayImage src = Gray(a["in." + input]);
                    var interp = args.GetProperty("interp").GetString() == "area" ? Interpolation.Area : Interpolation.Linear;
                    GrayImage got = CvResize.Resize(src, args.GetProperty("w").GetInt32(), args.GetProperty("h").GetInt32(), interp);
                    AssertBytes(a["out." + name], got.Pixels, maxLevel: 0, maxFraction: 0);
                    break;
                }

            case "warp_replicate":
                {
                    GrayImage src = Gray(a["in." + input]);
                    double deg = args.GetProperty("deg").GetDouble();
                    GrayImage got = CvWarp.WarpReplicate(src, CvWarp.RotationMatrix(src.Width / 2.0, src.Height / 2.0, deg));
                    AssertBytes(a["out." + name], got.Pixels, maxLevel: 0, maxFraction: 0);
                    break;
                }

            case "warp_zero":
                {
                    Tensor ink = a["in.ink"];
                    int h = ink.Shape[0], w = ink.Shape[1];
                    double deg = args.GetProperty("deg").GetDouble();
                    float[] got = CvWarp.WarpConstantZero(ink.ToFloatArray(), w, h, CvWarp.RotationMatrix(w / 2.0, h / 2.0, deg));
                    float[] want = a["out." + name].ToFloatArray();
                    double max = want.Zip(got).Max(p => Math.Abs(p.First - p.Second));
                    output.WriteLine($"{name}: max abs diff {max:E2}");
                    Assert.True(max == 0, $"max abs diff {max}");
                    break;
                }

            case "cc":
                {
                    Tensor mask = a["in.mask"];
                    int h = mask.Shape[0], w = mask.Shape[1];
                    bool[] m = mask.ToLongArray().Select(v => v != 0).ToArray();
                    (int[] labels, ConnectedComponents.Stats[] stats) = ConnectedComponents.Label(m, w, h);
                    Assert.Equal(args.GetProperty("n").GetInt32(), stats.Length);
                    Assert.Equal(a["out.cc_labels"].ToLongArray(), labels.Select(l => (long)l).ToArray());
                    long[] want = a["out.cc_stats"].ToLongArray();
                    for (int l = 1; l < stats.Length; l++)
                    {
                        Assert.Equal(want[(l * 5)..((l * 5) + 5)], new long[] { stats[l].Left, stats[l].Top, stats[l].Width, stats[l].Height, stats[l].Area });
                    }

                    break;
                }

            case "decode":
                {
                    GrayImage got = ImageIO.ReadPages(Path.Combine(Dir, input))[0];
                    AssertBytes(a["out." + name], got.Pixels, maxLevel: 0, maxFraction: 0);
                    break;
                }

            default:
                throw new InvalidOperationException(op);
        }
    }

    private void AssertBytes(Tensor want, byte[] got, int maxLevel, double maxFraction)
    {
        long[] w = want.ToLongArray();
        Assert.True(got.Length >= w.Length, "output too small");
        int diff = 0, worst = 0;
        for (int i = 0; i < w.Length; i++)
        {
            int d = Math.Abs((int)w[i] - got[i]);
            if (d > 0)
            {
                diff++;
                worst = Math.Max(worst, d);
            }
        }

        double fraction = (double)diff / w.Length;
        output.WriteLine($"{diff} of {w.Length} bytes differ ({fraction:P4}), worst {worst}");
        Assert.True(worst <= maxLevel && fraction <= maxFraction, $"{diff} of {w.Length} bytes differ, worst by {worst}");
    }
}
