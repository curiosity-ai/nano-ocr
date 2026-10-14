using System.Text.Json;
using NanoOcr.Models;
using NanoOcr.Onnx;
using Xunit;

namespace NanoOcr.Tests;

/// <summary>
/// Every node output of every model, compared with ONNX Runtime's on the same inputs. The
/// reference dumps come from <c>tools/reference/dump_layers.py</c>; the test is skipped when they
/// have not been generated.
/// </summary>
public class LayerParityTests(ITestOutputHelper output)
{
    public static TheoryData<string> Models => new()
    {
        "det", "rec", "rec_cjk", "script", "mathcls", "orient", "order", "math_enc", "math_dec_init", "math_dec_step",
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void EveryNodeMatchesOnnxRuntime(string model)
    {
        string npz = Path.Combine(TestData.Layers, model + ".npz");
        Assert.SkipUnless(File.Exists(npz), $"Reference dump {npz} missing; run tools/reference/dump_layers.py.");

        Dictionary<string, Tensor> arrays = Npz.Load(npz);
        using JsonDocument names = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestData.Layers, model + ".json")));
        string[] inputs = names.RootElement.GetProperty("inputs").EnumerateArray().Select(e => e.GetString()!).ToArray();
        string[] values = names.RootElement.GetProperty("values").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var expected = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        for (int i = 0; i < values.Length; i++)
        {
            expected[values[i]] = arrays[$"v.{i}"];
        }

        var feed = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        for (int i = 0; i < inputs.Length; i++)
        {
            feed[inputs[i]] = arrays[$"in.{i}"];
        }

        InferenceSession session = ModelStore.Embedded.Session(model);
        var failures = new List<string>();
        int compared = 0;
        double worst = 0;
        string worstName = "";
        session.Run(feed, (name, actual) =>
        {
            if (!expected.TryGetValue(name, out Tensor? reference))
            {
                return;
            }

            compared++;
            string? problem = Compare(reference, actual, out double error);
            if (error > worst)
            {
                worst = error;
                worstName = name;
            }

            if (problem is not null && failures.Count < 10)
            {
                failures.Add($"{name}: {problem}");
            }
        });

        output.WriteLine($"{model}: compared {compared} of {values.Length} node outputs; worst normalized error {worst:E2} at {worstName}");
        Assert.True(compared == values.Length, $"Only {compared} of {values.Length} node outputs were produced.");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>
    /// The fused graph (convolutions with their activation and residual folded in) produces exactly
    /// the unfused graph's outputs: fusion moves work, it must not change arithmetic. The one
    /// exception is the CTC head's <c>SoftmaxPeak</c>, whose sum of exponentials adds the same
    /// terms in another order: its float outputs agree within the layer tolerance, its ids exactly.
    /// </summary>
    [Theory]
    [MemberData(nameof(Models))]
    public void FusedGraphEqualsUnfused(string model)
    {
        string npz = Path.Combine(TestData.Layers, model + ".npz");
        Assert.SkipUnless(File.Exists(npz), $"Reference dump {npz} missing; run tools/reference/dump_layers.py.");
        Dictionary<string, Tensor> arrays = Npz.Load(npz);
        using JsonDocument names = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestData.Layers, model + ".json")));
        string[] inputs = names.RootElement.GetProperty("inputs").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var feed = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        for (int i = 0; i < inputs.Length; i++)
        {
            feed[inputs[i]] = arrays[$"in.{i}"];
        }

        InferenceSession session = ModelStore.Embedded.Session(model);
        Tensor[] fused = session.Run(feed);
        Tensor[] plain = session.RunUnfused(feed);
        output.WriteLine($"{model}: {session.Nodes.Count} nodes, {session.FusedNodes.Count} after fusion");
        bool reordered = session.FusedNodes.Any(n => n.OpType is "SoftmaxPeak" or "ConvSoftmaxPeak");
        for (int o = 0; o < fused.Length; o++)
        {
            Assert.Equal(plain[o].Shape, fused[o].Shape);
            bool exact = plain[o].ToFloatArray().AsSpan().SequenceEqual(fused[o].ToFloatArray());
            if (!exact && reordered && fused[o].Type == ElementType.Float)
            {
                Assert.Null(Compare(plain[o], fused[o], out _));
                continue;
            }

            Assert.True(exact, $"output {o} differs");
        }
    }

    /// <summary>
    /// Floats agree when every element is within <c>1e-3·(|ref| + rms(ref)) + 1e-5</c>: relative for
    /// large values, scaled to the tensor's own magnitude for values near zero. Integers and booleans
    /// must agree exactly.
    /// </summary>
    internal static string? Compare(Tensor expected, Tensor actual, out double error)
    {
        error = 0;
        if (!expected.Shape.SequenceEqual(actual.Shape))
        {
            error = double.PositiveInfinity;
            return $"shape [{string.Join(",", actual.Shape)}], expected [{string.Join(",", expected.Shape)}]";
        }

        if (expected.Type == ElementType.Float)
        {
            float[] e = expected.ToFloatArray();
            float[] a = actual.ToFloatArray();
            double rms = e.Length == 0 ? 0 : Math.Sqrt(e.Average(v => (double)v * v));
            int bad = 0;
            int first = -1;
            for (int i = 0; i < e.Length; i++)
            {
                double tol = (1e-3 * (Math.Abs(e[i]) + rms)) + 1e-5;
                double d = Math.Abs(e[i] - a[i]);
                if (double.IsNaN(a[i]) != double.IsNaN(e[i]))
                {
                    d = double.PositiveInfinity;
                }
                else if (double.IsNaN(e[i]) || e[i] == a[i])
                {
                    d = 0;
                }

                error = Math.Max(error, d / tol * 1e-3);
                if (d > tol)
                {
                    bad++;
                    first = first < 0 ? i : first;
                }
            }

            return bad == 0 ? null : $"{bad} of {e.Length} elements differ; first at {first}: {a[first]} vs {e[first]}";
        }

        long[] el = expected.ToLongArray();
        long[] al = actual.ToLongArray();
        int mismatches = el.Zip(al).Count(p => p.First != p.Second);
        error = mismatches;
        return mismatches == 0 ? null : $"{mismatches} of {el.Length} integer elements differ";
    }
}
