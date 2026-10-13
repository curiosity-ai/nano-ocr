using System.Text;
using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Models;
using IndraOcr.Onnx;

namespace IndraOcr.Pipeline;

/// <summary>
/// Math lines (port of <c>indra_ocr/math.py</c>): a classifier finds lines that hold math, and
/// MathRec reads them as text with inline LaTeX (<c>we have $x _ { i } \leq 1$</c>) with an
/// attention decoder, one token per step, keys and values cached.
/// </summary>
public sealed class MathReader
{
    /// <summary>Classifier probability above which a line is read as math.</summary>
    public const float MathMin = 0.5f;

    /// <summary>Mean token probability a math reading needs to replace a line.</summary>
    public const double MathMinConf = 0.9;

    /// <summary>A line this far below <see cref="MathMinConf"/> after 8 tokens stops decoding.</summary>
    public const double EarlyStopMargin = 0.1;

    private const int Bos = 1, Eos = 2, Special = 4;

    private readonly InferenceSession _cls, _enc, _init, _step;
    private readonly string[] _itos;
    private readonly HashSet<string> _initInputs, _stepInputs;

    private MathReader(ModelStore store)
    {
        _cls = store.Session("mathcls");
        _enc = store.Session("math_enc");
        _init = store.Session("math_dec_init");
        _step = store.Session("math_dec_step");
        using JsonDocument meta = JsonDocument.Parse(store.ReadText("math_meta.json"));
        JsonElement cfg = meta.RootElement.GetProperty("config");
        Height = cfg.GetProperty("height").GetInt32();
        MaxTokens = cfg.GetProperty("max_tokens").GetInt32();
        Layers = cfg.GetProperty("dec_layers").GetInt32();
        Heads = cfg.GetProperty("dec_heads").GetInt32();
        HeadDim = cfg.GetProperty("channels").GetInt32() / Heads;
        _itos = ["", "", "", "", .. meta.RootElement.GetProperty("tokens").EnumerateArray().Select(e => e.GetString()!)];
        _initInputs = [.. _init.InputNames];
        _stepInputs = [.. _step.InputNames];
    }

    /// <summary>Crop height of the math recognizer.</summary>
    public int Height { get; }

    /// <summary>Maximum decoded tokens per line.</summary>
    public int MaxTokens { get; }

    /// <summary>Decoder layers.</summary>
    public int Layers { get; }

    /// <summary>Decoder attention heads.</summary>
    public int Heads { get; }

    /// <summary>Width of one head.</summary>
    public int HeadDim { get; }

    /// <summary>The reader, or <see langword="null"/> when the store lacks a math model.</summary>
    public static MathReader? Load(ModelStore store)
    {
        string[] files = ["mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx", "math_meta.json"];
        return files.All(store.Has) ? new MathReader(store) : null;
    }

    /// <summary>
    /// <c>detokenize</c>: tokens to text; inside math (between <c>$</c> tokens) they are joined by spaces.
    /// </summary>
    public static string Detokenize(IReadOnlyList<string> tokens)
    {
        var parts = new List<string>();
        var math = new List<string>();
        bool inMath = false;
        foreach (string t in tokens)
        {
            if (t == "$")
            {
                if (inMath)
                {
                    parts.Add("$" + string.Join(" ", math) + "$");
                    math.Clear();
                }

                inMath = !inMath;
            }
            else if (inMath)
            {
                math.Add(t);
            }
            else
            {
                parts.Add(t);
            }
        }

        if (inMath && math.Count > 0)
        {
            parts.Add("$" + string.Join(" ", math) + "$");
        }

        return Py.CollapseSpaces(string.Concat(parts));
    }

    /// <summary><c>flags</c>: per recognizer crop, does it hold math?</summary>
    public bool[] Flags(IReadOnlyList<LineCrop> crops, int batch = 64)
    {
        var result = new bool[crops.Count];
        foreach (List<int> idx in LineCrops.WidthBatches(crops.Select(c => c.Width).ToList(), batch))
        {
            (Tensor x, Tensor widths) = LineCrops.PadBatch(idx.Select(j => crops[j]).ToList());
            Tensor[] outs = _cls.Run(new Dictionary<string, Tensor> { ["x"] = x, ["widths"] = widths });
            for (int k = 0; k < idx.Count; k++)
            {
                result[idx[k]] = outs[0].Floats[k] > MathMin;
            }

            Release(outs);
            x.Release();
            widths.Release();
        }

        return result;
    }

    /// <summary><c>read</c>: (text with inline LaTeX, mean token probability) for each box.</summary>
    public (string Text, double Conf)[] Read(GrayImage gray, IReadOnlyList<LineBox> boxes, int batch = 64)
    {
        var crops = boxes.Select(b => LineCrops.Crop(gray, b, Height)).ToList();
        var result = new (string, double)[crops.Count];
        Array.Fill(result, ("", 0.0));
        foreach (List<int> idx in LineCrops.WidthBatches(crops.Select(c => c.Width).ToList(), batch, maxRatio: 4.0, maxArea: 16 * 64 * 640))
        {
            (Tensor x, Tensor widths) = LineCrops.PadBatch(idx.Select(j => crops[j]).ToList());
            Tensor[] enc = _enc.Run(new Dictionary<string, Tensor> { ["x"] = x, ["widths"] = widths });
            x.Release();
            widths.Release();
            Tensor memory = enc[0], mask = enc[1];
            var feed = new Dictionary<string, Tensor> { ["memory"] = memory, ["mask"] = mask };
            Tensor[] init = _init.Run(feed.Where(kv => _initInputs.Contains(kv.Key)).ToDictionary());
            Tensor ck = init[0], cv = init[1];
            int b = idx.Count;
            Tensor sk = Tensor.Float([Layers, b, Heads, 0, HeadDim]);
            Tensor sv = Tensor.Float([Layers, b, Heads, 0, HeadDim]);
            Tensor tok = Tensor.Int64([b]);
            tok.Longs.Fill(Bos);
            var seqs = Enumerable.Range(0, b).Select(_ => new List<int>()).ToArray();
            var probs = Enumerable.Range(0, b).Select(_ => new List<double>()).ToArray();
            var done = new bool[b];
            for (int t = 0; t < MaxTokens - 1; t++)
            {
                var step = new Dictionary<string, Tensor>
                {
                    ["token"] = tok,
                    ["pos"] = Tensor.FromLongs([t]),
                    ["self_k"] = sk,
                    ["self_v"] = sv,
                    ["cross_k"] = ck,
                    ["cross_v"] = cv,
                    ["mask"] = mask,
                };
                Tensor[] outs = _step.Run(step.Where(kv => _stepInputs.Contains(kv.Key)).ToDictionary());
                Tensor logits = outs[0];
                sk.Release();
                sv.Release();
                sk = outs[1];
                sv = outs[2];
                int classes = logits.Shape[1];
                Tensor next = Tensor.Int64([b]);
                for (int r = 0; r < b; r++)
                {
                    ReadOnlySpan<float> row = logits.Floats.Slice(r * classes, classes);
                    int arg = Py.ArgMax(row);
                    next.LongArray[r] = arg;
                    float max = row[arg];
                    var e = new float[classes];
                    for (int c = 0; c < classes; c++)
                    {
                        e[c] = MathF.Exp(row[c] - max);
                    }

                    float top = 1f / Py.PairwiseSum(e);
                    if (done[r])
                    {
                        continue;
                    }

                    probs[r].Add(top);
                    if (arg == Eos)
                    {
                        done[r] = true;
                    }
                    else if (probs[r].Count >= 8 && PyDouble.Mean(probs[r].ToArray()) < MathMinConf - EarlyStopMargin)
                    {
                        done[r] = true;
                    }
                    else if (arg >= Special)
                    {
                        seqs[r].Add(arg);
                    }
                }

                logits.Release();
                tok.Release();
                tok = next;
                if (done.All(d => d))
                {
                    break;
                }
            }

            for (int k = 0; k < b; k++)
            {
                string text = Detokenize(seqs[k].Select(i => i < _itos.Length ? _itos[i] : "").ToList());
                result[idx[k]] = (text, probs[k].Count > 0 ? PyDouble.Mean(probs[k].ToArray()) : 0.0);
            }

            Release([memory, mask, ck, cv, sk, sv, tok]);
        }

        return result;
    }

    private static void Release(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor t in tensors)
        {
            t.Release();
        }
    }
}
