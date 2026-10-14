using System.Buffers;
using NanoOcr.Imaging;
using NanoOcr.Models;
using NanoOcr.Onnx;

namespace NanoOcr.Pipeline;

/// <summary>What a page looks like, as <see cref="PageTypeModel"/> sees it.</summary>
public enum PageType
{
    /// <summary>Printed text: paragraphs, headings, lists, columns — scanned or rendered.</summary>
    Printed,

    /// <summary>Printed, but dominated by tables or forms: invoices, budgets, specifications, questionnaires.</summary>
    Table,

    /// <summary>Mostly handwriting: notes, letters, filled-in pages.</summary>
    Handwritten,

    /// <summary>A camera photo: a scene, a sign, a receipt or a page photographed rather than scanned.</summary>
    Photo,
}

/// <summary>The page-type model's verdict on one page.</summary>
/// <param name="Type">The most likely type.</param>
/// <param name="Probabilities">One probability per <see cref="PageType"/>, in enum order.</param>
public sealed record PageTypePrediction(PageType Type, float[] Probabilities)
{
    /// <summary>The probability of <see cref="Type"/>.</summary>
    public float Confidence => Probabilities[(int)Type];

    /// <summary>The probability of <paramref name="type"/>.</summary>
    public float this[PageType type] => Probabilities[(int)type];

    /// <summary>The prediction for model probabilities in <see cref="PageType"/> order.</summary>
    public static PageTypePrediction FromProbabilities(float[] probabilities) =>
        new((PageType)Py.ArgMax(probabilities), probabilities);
}

/// <summary>
/// The page-type classifier: MobileNetV3-Small (ImageNet weights, timm's
/// <c>mobilenetv3_small_100.lamb_in1k</c>) fine-tuned to tell printed, table, handwritten and photo
/// pages apart from the page image alone. Not part of Indra OCR; trained by
/// <c>tools/pagetype/train.py</c>, which documents the data.
/// </summary>
/// <remarks>
/// <para>
/// It looks at the page rather than at NanoOCR's reading of it, which is what the reading-based
/// <see cref="Routing.PageClassifier"/> rules cannot do: a scene photo where the detector finds no
/// lines, neat handwriting the recognizer reads with fair confidence, a form whose lines the order
/// model does not call a table. The router combines the two.
/// </para>
/// <para>
/// One run is a 512×512 gray image through a 1.5M-parameter network: a few tens of milliseconds,
/// against the hundreds a page's OCR takes.
/// </para>
/// </remarks>
public sealed class PageTypeModel
{
    /// <summary>The model's file name in a <see cref="ModelStore"/>, without <c>.onnx</c>.</summary>
    public const string ModelName = "pagetype";

    /// <summary>Side of the model's square input.</summary>
    public const int Side = 512;

    private readonly InferenceSession _session;
    private readonly bool _fusion;

    private PageTypeModel(InferenceSession session, bool fusion)
    {
        _session = session;
        _fusion = fusion;
    }

    /// <summary>The model in <paramref name="store"/>, or <see langword="null"/> when the store has none.</summary>
    public static PageTypeModel? Load(ModelStore store, bool fusion = true) =>
        store.HasModel(ModelName) ? new PageTypeModel(store.Session(ModelName), fusion) : null;

    /// <summary>
    /// The model input: the page's long side scaled to <paramref name="side"/> with INTER_AREA,
    /// top-left on a white square, as <c>gray / 255</c> (<c>letterbox</c> in <c>tools/pagetype/train.py</c>).
    /// </summary>
    public static float[] Input(GrayImage gray, int side = Side)
    {
        var result = new float[side * side];
        Input(gray, side, result);
        return result;
    }

    private static void Input(GrayImage gray, int side, float[] result)
    {
        int h = gray.Height, w = gray.Width;
        double s = (double)side / Math.Max(h, w);
        int sw = (int)Math.Max(1, Py.Round(w * s)), sh = (int)Math.Max(1, Py.Round(h * s));
        GrayImage small = GrayImage.Rent(sw, sh);
        CvResize.ResizeInto(gray, small, Interpolation.Area);
        result.AsSpan(0, side * side).Fill(1f);
        for (int y = 0; y < Math.Min(sh, side); y++)
        {
            for (int x = 0; x < Math.Min(sw, side); x++)
            {
                result[(y * side) + x] = small.Pixels[(y * sw) + x] / 255f;
            }
        }

        small.Return();
    }

    /// <summary>Classifies an upright page.</summary>
    public PageTypePrediction Predict(GrayImage gray)
    {
        float[] input = ArrayPool<float>.Shared.Rent(Side * Side);
        Input(gray, Side, input);
        Tensor x = Tensor.FromFloats(input, 1, 1, Side, Side);
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        Tensor[] outs = _fusion ? _session.Run(feed) : _session.RunUnfused(feed);
        ArrayPool<float>.Shared.Return(input);
        float[] p = outs[0].ToFloatArray();
        outs[0].Release();
        return PageTypePrediction.FromProbabilities(p);
    }
}
