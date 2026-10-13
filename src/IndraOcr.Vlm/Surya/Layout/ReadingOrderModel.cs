using IndraOcr.Vlm.Core;
using IndraOcr.Vlm.Formats;

namespace IndraOcr.Vlm.Surya.Layout;

/// <summary>
/// The autoregressive reading-order head that pairs with the fast layout detector —
/// <c>ReadingOrderAR</c> of <c>surya/common/order/order_ar.py</c> and <c>OrderPredictor.order_page</c>
/// of <c>surya/common/order/predictor.py</c>.
/// </summary>
/// <remarks>
/// <para>
/// The boxes are put in a y-banded raster order (<c>canonical_order</c>), embedded from their
/// geometry and label, and cross-attend to the detector's projector map through a 3-layer
/// <c>nn.TransformerDecoder</c> (<c>ctx</c>). A second 3-layer decoder (<c>dec</c>) then emits, one step
/// at a time under a causal mask, which raster position is read next; the argmax is constrained
/// to positions not yet emitted, so the result is a permutation.
/// </para>
/// <para>
/// Python re-runs the decoder over the whole prefix every step. Under the causal mask the rows
/// already computed cannot change, so this keeps each layer's keys and values and computes only
/// the new row — the same arithmetic, without the quadratic recomputation.
/// </para>
/// </remarks>
public sealed class ReadingOrderModel
{
    /// <summary><c>LAYOUT_CLASSES</c>: the 19 layout labels, sorted ordinally.</summary>
    public static readonly string[] LayoutClasses = new[]
    {
        "Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer", "Image",
        "Section-Header", "Table", "Text", "Complex-Block", "Code-Block", "Form", "Table-Of-Contents",
        "Figure", "Chemical-Block", "Diagram", "Bibliography", "Blank-Page",
    }.Order(StringComparer.Ordinal).ToArray();

    /// <summary><c>MAX_BOXES</c>: pages with more detections fall back to raster order.</summary>
    public const int MaxBoxes = 128;

    private const float Eps = 1e-5f;
    private readonly int _d;
    private readonly int _heads = 4;
    private readonly Dense _geom, _featProj, _out;
    private readonly float[] _lab, _featPos, _bos, _step;
    private readonly Layer[] _ctx, _dec;

    private ReadingOrderModel(StateDict state, int d, int layers)
    {
        _d = d;
        _geom = state.Dense("geom");
        _featProj = state.Dense("feat_proj");
        _out = state.Dense("out");
        _lab = state.Floats("lab.weight");
        _featPos = state.Floats("feat_pos");
        _bos = state.Floats("bos");
        _step = state.Floats("step.weight");
        _ctx = Enumerable.Range(0, layers).Select(i => new Layer(state, $"ctx.layers.{i}.", d, _heads)).ToArray();
        _dec = Enumerable.Range(0, layers).Select(i => new Layer(state, $"dec.layers.{i}.", d, _heads)).ToArray();
    }

    /// <summary>Loads <c>order_ar.pt</c> from <paramref name="directory"/>.</summary>
    public static ReadingOrderModel Load(string directory)
    {
        TorchCheckpoint ck = TorchCheckpoint.Load(Path.Combine(directory, "order_ar.pt"));
        var state = new StateDict(ck.StateDict("model"));
        return new ReadingOrderModel(state, (int)ck.GetInt("d", 128), (int)ck.GetInt("layers", 3));
    }

    /// <summary>
    /// <c>OrderPredictor.order_page</c>: the reading position of each detection (0 = read first).
    /// </summary>
    /// <param name="features">The detector's projector map, channels-last <c>[F·F, 256]</c>.</param>
    /// <param name="boxes">Pixel <c>(x0, y0, x1, y1)</c> per detection.</param>
    /// <param name="labels">The raw label of each detection.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="trace">Receives <c>order.memory</c> and one <c>order.logits</c> row per step.</param>
    public int[] OrderPage(
        float[] features, IReadOnlyList<(float X0, float Y0, float X1, float Y1)> boxes, IReadOnlyList<string> labels,
        int width, int height, LayoutTrace? trace = null)
    {
        int n = boxes.Count;
        if (n == 0)
        {
            return [];
        }

        if (n == 1)
        {
            return [0];
        }

        if (n > MaxBoxes)
        {
            return RasterPositions(boxes);
        }

        // b1000: numpy float32 arithmetic (a Python int/float meeting a float32 array stays float32).
        float fw = Math.Max(1, width), fh = Math.Max(1, height);
        float[] b1000 = new float[n * 4];
        for (int i = 0; i < n; i++)
        {
            b1000[(i * 4) + 0] = boxes[i].X0 / fw * 1000f;
            b1000[(i * 4) + 1] = boxes[i].Y0 / fh * 1000f;
            b1000[(i * 4) + 2] = boxes[i].X1 / fw * 1000f;
            b1000[(i * 4) + 3] = boxes[i].Y1 / fh * 1000f;
        }

        int[] order = CanonicalOrder(b1000, n);
        float[] geometry = new float[n * 8];
        float[] x = new float[n * _d];
        for (int p = 0; p < n; p++)
        {
            int i = order[p];
            // box_features: /1000, then (x0, y0, x1, y1, cx, cy, w, h), all float32.
            float x0 = b1000[(i * 4) + 0] / 1000f, y0 = b1000[(i * 4) + 1] / 1000f;
            float x1 = b1000[(i * 4) + 2] / 1000f, y1 = b1000[(i * 4) + 3] / 1000f;
            Span<float> g = geometry.AsSpan(p * 8, 8);
            g[0] = x0;
            g[1] = y0;
            g[2] = x1;
            g[3] = y1;
            g[4] = (x0 + x1) / 2f;
            g[5] = (y0 + y1) / 2f;
            g[6] = x1 - x0;
            g[7] = y1 - y0;
        }

        float[] embedded = _geom.Forward(geometry, n);
        for (int p = 0; p < n; p++)
        {
            int label = Array.IndexOf(LayoutClasses, labels[order[p]]);
            if (label < 0)
            {
                label = Array.IndexOf(LayoutClasses, "Text");
            }

            LayoutMath.Add(embedded.AsSpan(p * _d, _d), _lab.AsSpan(label * _d, _d));
        }

        x = embedded;

        // encode(): x cross-attends to feat_proj(fmap) + feat_pos.
        int tokens = features.Length / _featProj.In;
        float[] f = _featProj.Forward(features, tokens);
        LayoutMath.Add(f, _featPos.AsSpan(0, tokens * _d));
        foreach (Layer layer in _ctx)
        {
            x = layer.Forward(x, n, f, tokens);
        }

        trace?.Invoke("order.memory", x);

        // decode(): constrained greedy, incrementally.
        float[] memory = x;
        var caches = _dec.Select(l => new Layer.Cache(l, memory, n, n + 1)).ToArray();
        bool[] used = new bool[n];
        int[] sequence = new int[n];
        float[] input = new float[_d];
        for (int c = 0; c < _d; c++)
        {
            input[c] = _bos[c] + _step[c];
        }

        for (int t = 0; t < n; t++)
        {
            float[] h = input;
            foreach (Layer.Cache cache in caches)
            {
                h = cache.Step(h);
            }

            float[] logits = _out.Forward(h, 1);
            trace?.Invoke("order.logits", logits);

            // logits[k:] = -inf; used positions = -inf; argmax (first maximum).
            int next = -1;
            float best = float.NegativeInfinity;
            for (int k = 0; k < n; k++)
            {
                if (!used[k] && (next < 0 || logits[k] > best))
                {
                    best = logits[k];
                    next = k;
                }
            }

            sequence[t] = next;
            used[next] = true;
            ReadOnlySpan<float> step = _step.AsSpan(Math.Min(t + 1, MaxBoxes) * _d, _d);
            input = new float[_d];
            for (int c = 0; c < _d; c++)
            {
                input[c] = memory[(next * _d) + c] + step[c];
            }
        }

        int[] position = new int[n];
        for (int rank = 0; rank < n; rank++)
        {
            position[order[sequence[rank]]] = rank;
        }

        return position;
    }

    /// <summary>
    /// <c>canonical_order(boxes_1000, n_bands=24)</c>: sort by <c>(int(y0 // band_h), x0, y0, x1)</c>,
    /// with the floor division in float32 as numpy performs it on a <c>float32</c> scalar.
    /// </summary>
    public static int[] CanonicalOrder(float[] b1000, int n)
    {
        float band = (float)(1000.0 / 24);
        var keys = new (int Band, float X0, float Y0, float X1)[n];
        for (int i = 0; i < n; i++)
        {
            keys[i] = ((int)FloorDivide(b1000[(i * 4) + 1], band), b1000[i * 4], b1000[(i * 4) + 1], b1000[(i * 4) + 2]);
        }

        int[] index = Enumerable.Range(0, n).ToArray();
        return index.OrderBy(i => keys[i].Band).ThenBy(i => keys[i].X0).ThenBy(i => keys[i].Y0).ThenBy(i => keys[i].X1).ToArray();
    }

    /// <summary>numpy's <c>npy_floor_dividef</c> (via <c>npy_divmodf</c>).</summary>
    public static float FloorDivide(float a, float b)
    {
        float mod = a % b; // fmodf
        float div = (a - mod) / b;
        if (mod != 0f && (b < 0f) != (mod < 0f))
        {
            div -= 1f;
        }

        float floordiv;
        if (div != 0f)
        {
            floordiv = MathF.Floor(div);
            if (div - floordiv > 0.5f)
            {
                floordiv += 1f;
            }
        }
        else
        {
            floordiv = MathF.CopySign(0f, a / b);
        }

        return floordiv;
    }

    /// <summary><c>_raster_positions</c>: top-to-bottom, then left-to-right, on the pixel boxes.</summary>
    public static int[] RasterPositions(IReadOnlyList<(float X0, float Y0, float X1, float Y1)> boxes)
    {
        int[] idx = Enumerable.Range(0, boxes.Count).OrderBy(i => boxes[i].Y0).ThenBy(i => boxes[i].X0).ToArray();
        int[] position = new int[boxes.Count];
        for (int rank = 0; rank < idx.Length; rank++)
        {
            position[idx[rank]] = rank;
        }

        return position;
    }

    /// <summary>
    /// <c>nn.TransformerDecoderLayer</c> (post-norm, ReLU, <c>d_ff = 4d</c>, eps 1e-5):
    /// <c>x = norm1(x + sa(x)); x = norm2(x + mha(x, mem)); x = norm3(x + ff(x))</c>.
    /// </summary>
    private sealed class Layer
    {
        private readonly MultiHeadAttention _self, _cross;
        private readonly LayerNormLayer _norm1, _norm2, _norm3;
        private readonly Dense _linear1, _linear2;
        private readonly int _heads;

        public Layer(StateDict state, string p, int d, int heads)
        {
            _heads = heads;
            _self = new MultiHeadAttention(state, p + "self_attn", d, heads);
            _cross = new MultiHeadAttention(state, p + "multihead_attn", d, heads);
            _norm1 = state.Norm(p + "norm1", Eps);
            _norm2 = state.Norm(p + "norm2", Eps);
            _norm3 = state.Norm(p + "norm3", Eps);
            _linear1 = state.Dense(p + "linear1");
            _linear2 = state.Dense(p + "linear2");
        }

        /// <summary>The whole layer over <paramref name="rows"/> rows, no mask.</summary>
        public float[] Forward(float[] x, int rows, float[] memory, int memoryRows)
        {
            float[] sa = _self.Forward(x, rows, x, x, rows);
            float[] h = LayoutMath.Sum(x, sa);
            _norm1.Apply(h);
            return CrossAndFeedForward(h, rows, _cross.Forward(h, rows, memory, memory, memoryRows));
        }

        private float[] CrossAndFeedForward(float[] h, int rows, float[] ca)
        {
            LayoutMath.Add(h, ca);
            _norm2.Apply(h);
            float[] ff = _linear1.Forward(h, rows);
            LayoutMath.Relu(ff);
            LayoutMath.Add(h, _linear2.Forward(ff, rows));
            _norm3.Apply(h);
            return h;
        }

        /// <summary>One layer of the causal decoder, stepped a row at a time.</summary>
        public sealed class Cache
        {
            private readonly Layer _layer;
            private readonly float[] _memoryK, _memoryV, _keys, _values;
            private readonly int _memoryRows;
            private int _length;

            public Cache(Layer layer, float[] memory, int memoryRows, int capacity)
            {
                _layer = layer;
                _memoryRows = memoryRows;
                _memoryK = layer._cross.ProjectKey(memory, memoryRows);
                _memoryV = layer._cross.ProjectValue(memory, memoryRows);
                int d = layer._self.Dim;
                _keys = new float[capacity * d];
                _values = new float[capacity * d];
            }

            /// <summary>Feeds the next row; returns this layer's output for it.</summary>
            public float[] Step(float[] x)
            {
                int d = _layer._self.Dim;
                float[] q = _layer._self.ProjectQuery(x, 1);
                _layer._self.ProjectKey(x, 1).CopyTo(_keys, _length * d);
                _layer._self.ProjectValue(x, 1).CopyTo(_values, _length * d);
                _length++;
                float[] context = LayoutMath.Attend(
                    q, 1, _keys.AsSpan(0, _length * d).ToArray(), _values.AsSpan(0, _length * d).ToArray(), _length, _layer._heads, d / _layer._heads, causal: false);
                float[] h = LayoutMath.Sum(x, _layer._self.Output(context, 1));
                _layer._norm1.Apply(h);

                float[] cq = _layer._cross.ProjectQuery(h, 1);
                float[] cross = _layer._cross.Output(_layer._cross.Attend(cq, 1, _memoryK, _memoryV, _memoryRows), 1);
                return _layer.CrossAndFeedForward(h, 1, cross);
            }
        }
    }
}
