namespace NanoOcr.Onnx;

/// <summary>The operator table: ONNX op type → kernel.</summary>
internal static partial class Operators
{
    /// <summary>Runs one node. Inputs may contain <see langword="null"/> for omitted optional inputs.</summary>
    public delegate Tensor[] Kernel(OnnxNode node, List<Tensor?> inputs);

    private static readonly Dictionary<string, Kernel> Table = new(StringComparer.Ordinal)
    {
        // Element-wise arithmetic and logic (Elementwise.cs)
        ["Add"] = (n, x) => [Binary(x[0]!, x[1]!, BinaryOp.Add)],
        ["Sub"] = (n, x) => [Binary(x[0]!, x[1]!, BinaryOp.Sub)],
        ["Mul"] = (n, x) => [Binary(x[0]!, x[1]!, BinaryOp.Mul)],
        ["Div"] = (n, x) => [Binary(x[0]!, x[1]!, BinaryOp.Div)],
        ["Pow"] = (n, x) => [Binary(x[0]!, x[1]!, BinaryOp.Pow)],
        ["Mod"] = (n, x) => [Binary(x[0]!, x[1]!, n.Int("fmod", 0) != 0 ? BinaryOp.FMod : BinaryOp.Mod)],
        ["Less"] = (n, x) => [Compare(x[0]!, x[1]!, CompareOp.Less)],
        ["Greater"] = (n, x) => [Compare(x[0]!, x[1]!, CompareOp.Greater)],
        ["Equal"] = (n, x) => [Compare(x[0]!, x[1]!, CompareOp.Equal)],
        ["And"] = (n, x) => [Logical(x[0]!, x[1]!, LogicOp.And)],
        ["Or"] = (n, x) => [Logical(x[0]!, x[1]!, LogicOp.Or)],
        ["Xor"] = (n, x) => [Logical(x[0]!, x[1]!, LogicOp.Xor)],
        ["Not"] = (n, x) => [Not(x[0]!)],
        ["Where"] = (n, x) => [Where(x[0]!, x[1]!, x[2]!)],
        ["Abs"] = (n, x) => [Unary(x[0]!, UnaryOp.Abs)],
        ["Neg"] = (n, x) => [Unary(x[0]!, UnaryOp.Neg)],
        ["Sqrt"] = (n, x) => [Unary(x[0]!, UnaryOp.Sqrt)],
        ["Exp"] = (n, x) => [Unary(x[0]!, UnaryOp.Exp)],
        ["Sin"] = (n, x) => [Unary(x[0]!, UnaryOp.Sin)],
        ["Cos"] = (n, x) => [Unary(x[0]!, UnaryOp.Cos)],
        ["Erf"] = (n, x) => [Unary(x[0]!, UnaryOp.Erf)],
        ["Reciprocal"] = (n, x) => [Unary(x[0]!, UnaryOp.Reciprocal)],
        ["Relu"] = (n, x) => [Unary(x[0]!, UnaryOp.Relu)],
        ["Sigmoid"] = (n, x) => [Unary(x[0]!, UnaryOp.Sigmoid)],
        ["HardSwish"] = (n, x) => [Unary(x[0]!, UnaryOp.HardSwish)],
        ["Softplus"] = (n, x) => [Unary(x[0]!, UnaryOp.Softplus)],
        ["Tanh"] = (n, x) => [Unary(x[0]!, UnaryOp.Tanh)],
        ["Clip"] = (n, x) => [Clip(n, x)],
        ["HardSigmoid"] = (n, x) => [HardSigmoid(x[0]!, n.Float("alpha", 0.2f), n.Float("beta", 0.5f))],
        ["Cast"] = (n, x) => [Cast(x[0]!, (int)n.Int("to", 1))],
        ["Identity"] = (n, x) => [x[0]!.Reshape(x[0]!.Shape)],

        // Shape algebra (Shapes.cs)
        ["Shape"] = (n, x) => [ShapeOf(n, x[0]!)],
        ["Reshape"] = (n, x) => [Reshape(x[0]!, x[1]!, n.Int("allowzero", 0) != 0)],
        ["Flatten"] = (n, x) => [Flatten(x[0]!, (int)n.Int("axis", 1))],
        ["Squeeze"] = (n, x) => [Squeeze(n, x)],
        ["Unsqueeze"] = (n, x) => [Unsqueeze(n, x)],
        ["Concat"] = (n, x) => [Concat(x, (int)n.Int("axis", 0))],
        ["Split"] = Split,
        ["Slice"] = (n, x) => [Slice(x)],
        ["Gather"] = (n, x) => [Gather(x[0]!, x[1]!, (int)n.Int("axis", 0))],
        ["Transpose"] = (n, x) => [Transpose(x[0]!, n.Ints("perm"))],
        ["Expand"] = (n, x) => [Expand(x[0]!, x[1]!)],
        ["ConstantOfShape"] = (n, x) => [ConstantOfShape(n, x[0]!)],
        ["Range"] = (n, x) => [Range(x[0]!, x[1]!, x[2]!)],
        ["ScatterND"] = (n, x) => [ScatterNd(x[0]!, x[1]!, x[2]!)],
        ["DepthToSpace"] = (n, x) => [DepthToSpace(x[0]!, (int)n.Int("blocksize", 1), n.Str("mode", "DCR"))],

        // Reductions (Reductions.cs)
        ["ReduceSum"] = (n, x) => [Reduce(n, x, ReduceOp.Sum)],
        ["ReduceMean"] = (n, x) => [Reduce(n, x, ReduceOp.Mean)],
        ["ReduceMax"] = (n, x) => [Reduce(n, x, ReduceOp.Max)],
        ["ArgMax"] = (n, x) => [ArgMax(x[0]!, (int)n.Int("axis", 0), n.Int("keepdims", 1) != 0, n.Int("select_last_index", 0) != 0)],
        ["Softmax"] = (n, x) => [Softmax(x[0]!, (int)n.Int("axis", -1))],
        ["GlobalAveragePool"] = (n, x) => [GlobalAveragePool(x[0]!)],

        // Neural-network layers (Layers.cs)
        ["Conv"] = (n, x) => [Conv(n, x)],
        ["MatMul"] = (n, x) => [MatMul(x[0]!, x[1]!)],
        ["Gemm"] = (n, x) => [GemmOp(n, x)],
        ["BatchNormalization"] = (n, x) => [BatchNorm(n, x)],
        ["LayerNormalization"] = (n, x) => [LayerNorm(n, x)],
        ["MaxPool"] = (n, x) => [Pool(n, x[0]!, max: true)],
        ["AveragePool"] = (n, x) => [Pool(n, x[0]!, max: false)],
        ["Resize"] = (n, x) => [Resize(n, x)],
    };

    /// <summary>The kernel for <paramref name="node"/>, or <see langword="null"/> if unsupported.</summary>
    public static Kernel? Resolve(OnnxNode node) =>
        node.Domain is "" or "ai.onnx" && Table.TryGetValue(node.OpType, out Kernel? k) ? k
        : node.Domain == FusedDomain && FusedTable.TryGetValue(node.OpType, out k) ? k
        : null;

    /// <summary>Normalizes a possibly negative axis.</summary>
    internal static int Axis(long axis, int rank) => (int)(axis < 0 ? axis + rank : axis);
}
