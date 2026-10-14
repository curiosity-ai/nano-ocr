using System.Diagnostics;

namespace NanoOcr.Onnx;

/// <summary>
/// Runs an ONNX graph with this library's own kernels — the replacement for ONNX Runtime's
/// <c>InferenceSession</c>. Nodes run in the file's topological order; every intermediate tensor
/// goes back to the pool at its last use.
/// </summary>
public sealed class InferenceSession
{
    private readonly OnnxGraph _graph;
    private readonly Dictionary<string, int> _lastUse = new(StringComparer.Ordinal);
    private readonly HashSet<string> _graphOutputs;
    private readonly Dictionary<string, Tensor> _constants = new(StringComparer.Ordinal);
    private readonly List<OnnxNode> _nodes = [];
    private readonly Operators.Kernel[] _kernels;
    private readonly List<OnnxNode> _fusedNodes;
    private readonly Operators.Kernel[] _fusedKernels;

    /// <summary>Creates a session over a parsed model.</summary>
    public InferenceSession(OnnxModel model)
    {
        Model = model;
        _graph = model.Graph;
        foreach (var (name, t) in _graph.Initializers)
        {
            _constants[name] = t;
        }

        foreach (OnnxNode node in _graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Domain is "" or "ai.onnx")
            {
                _constants[node.Outputs[0]] = ConstantValue(node).Pin();
                continue;
            }

            _nodes.Add(node);
        }

        FoldConstants();
        _kernels = new Operators.Kernel[_nodes.Count];
        for (int i = 0; i < _nodes.Count; i++)
        {
            OnnxNode node = _nodes[i];
            _kernels[i] = Operators.Resolve(node)
                ?? throw new NotSupportedException($"ONNX operator '{node.OpType}' (node '{node.Name}') is not implemented.");
            foreach (string input in node.Inputs)
            {
                if (input.Length > 0)
                {
                    _lastUse[input] = i;
                }
            }
        }

        _graphOutputs = new HashSet<string>(_graph.Outputs.Select(o => o.Name), StringComparer.Ordinal);
        _fusedNodes = Environment.GetEnvironmentVariable("NANOOCR_NO_FUSION") == "1" ? _nodes : Fuse(_nodes, _graphOutputs);
        _fusedKernels = _fusedNodes.Select(n => Operators.Resolve(n)!).ToArray();
        _plan = new Lazy<Plan>(() => BuildPlan(_nodes));
        _fusedPlan = new Lazy<Plan>(() => BuildPlan(_fusedNodes));
        InputNames = _graph.Inputs.Select(i => i.Name).ToArray();
        OutputNames = _graph.Outputs.Select(o => o.Name).ToArray();
    }

    /// <summary>Creates a session from a model file.</summary>
    public InferenceSession(string path)
        : this(OnnxModel.Load(path))
    {
    }

    /// <summary>The model this session runs.</summary>
    public OnnxModel Model { get; }

    /// <summary>Names of the graph inputs, in declaration order.</summary>
    public IReadOnlyList<string> InputNames { get; }

    /// <summary>Names of the graph outputs, in declaration order.</summary>
    public IReadOnlyList<string> OutputNames { get; }

    /// <summary>Nodes that execute (constants folded away), in order, before fusion.</summary>
    public IReadOnlyList<OnnxNode> Nodes => _nodes;

    /// <summary>The nodes a run without an observer executes: <see cref="Nodes"/> after fusion.</summary>
    public IReadOnlyList<OnnxNode> FusedNodes => _fusedNodes;

    /// <summary>Accumulates time per operator type when set.</summary>
    public Dictionary<string, double>? Profile { get; set; }

    /// <summary>
    /// Runs the graph. Returns the graph outputs, in <see cref="OutputNames"/> order. The caller
    /// owns the returned tensors (and may <see cref="Tensor.Release"/> them).
    /// </summary>
    /// <param name="inputs">Graph inputs by name. They are not released.</param>
    /// <param name="observer">
    /// Called with every node output as it is produced (name, tensor), before anything is
    /// released — the hook the per-layer parity tests use. A run with an observer executes the
    /// graph as written, without fusion, so that every node output exists.
    /// </param>
    public Tensor[] Run(IReadOnlyDictionary<string, Tensor> inputs, Action<string, Tensor>? observer = null) =>
        observer is null ? Run(inputs, _fusedNodes, _fusedKernels, _fusedPlan.Value, null) : Run(inputs, _nodes, _kernels, _plan.Value, observer);

    /// <summary>Runs the graph as written, without fusion (for comparing the two).</summary>
    public Tensor[] RunUnfused(IReadOnlyDictionary<string, Tensor> inputs) => Run(inputs, _nodes, _kernels, _plan.Value, null);

    private Tensor[] Run(IReadOnlyDictionary<string, Tensor> inputs, List<OnnxNode> nodes, Operators.Kernel[] kernels, Plan plan, Action<string, Tensor>? observer)
    {
        var values = new Tensor?[plan.SlotCount];
        var owned = new bool[plan.SlotCount];
        foreach (var (slot, t) in plan.Constants)
        {
            values[slot] = t;
        }

        for (int i = 0; i < _graph.Inputs.Count; i++)
        {
            string name = _graph.Inputs[i].Name;
            if (!inputs.TryGetValue(name, out Tensor? t))
            {
                throw new ArgumentException($"Missing graph input '{name}'.", nameof(inputs));
            }

            values[plan.InputSlots[i]] = t;
        }

        if (observer is not null)
        {
            foreach (string name in _folded)
            {
                observer(name, _constants[name]);
            }
        }

        var args = new List<Tensor?>(8);
        for (int i = 0; i < nodes.Count; i++)
        {
            OnnxNode node = nodes[i];
            int[] ins = plan.NodeInputs[i];
            args.Clear();
            for (int k = 0; k < ins.Length; k++)
            {
                int slot = ins[k];
                if (slot < 0)
                {
                    args.Add(null);
                    continue;
                }

                args.Add(values[slot] ?? throw new InvalidOperationException(
                    $"Node '{node.Name}' ({node.OpType}) reads '{node.Inputs[k]}', which no node produced."));
            }

            long start = Profile is null ? 0 : Stopwatch.GetTimestamp();
            Tensor[] results;
            try
            {
                results = kernels[i](node, args);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                string shapes = string.Join(", ", args.Select(a => a?.ToString() ?? "-"));
                throw new InvalidOperationException($"Node '{node.Name}' ({node.OpType}) failed on inputs {shapes}: {e.Message}", e);
            }

            if (Profile is not null)
            {
                double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                string key = node.OpType is "MatMul" or "Transpose" or "Gather" ? $"{node.OpType} {string.Join("x", args[0]!.Shape)}*{string.Join("x", args.Count > 1 && args[1] is not null ? args[1]!.Shape : [])}" : node.OpType is "Mul" ? $"Mul {string.Join("x", args[0]!.Shape)}*{string.Join("x", args[1]!.Shape)}" : node.OpType == "Conv"
                    ? $"Conv {string.Join("x", node.Ints("kernel_shape") ?? [])} g{node.Int("group", 1)} {string.Join("x", args[0]!.Shape)}"
                    : node.OpType;
                Profile[key] = Profile.GetValueOrDefault(key) + ms;
                Profile["#" + key] = Profile.GetValueOrDefault("#" + key) + 1;
            }

            int[] outs = plan.NodeOutputs[i];
            for (int o = 0; o < outs.Length && o < results.Length; o++)
            {
                int slot = outs[o];
                if (slot < 0)
                {
                    results[o].Release();
                    continue;
                }

                observer?.Invoke(node.Outputs[o], results[o]);
                if (plan.LastUse[slot] < 0 && !plan.IsOutput[slot])
                {
                    results[o].Release();
                    continue;
                }

                values[slot] = results[o];
                owned[slot] = true;
            }

            // Release every input whose last reader was this node.
            foreach (int slot in plan.Frees[i])
            {
                if (owned[slot])
                {
                    values[slot]!.Release();
                    values[slot] = null;
                    owned[slot] = false;
                }
            }
        }

        var outputs = new Tensor[plan.OutputSlots.Length];
        for (int o = 0; o < outputs.Length; o++)
        {
            int slot = plan.OutputSlots[o];
            outputs[o] = values[slot] ?? throw new InvalidOperationException($"Output '{_graph.Outputs[o].Name}' was not produced.");
            owned[slot] = false;
        }

        for (int slot = 0; slot < owned.Length; slot++)
        {
            if (owned[slot])
            {
                values[slot]!.Release();
            }
        }

        return outputs;
    }

    /// <summary>
    /// The graph compiled to integer value slots: each node's input and output slots, the slots it
    /// frees, the constants and the graph's inputs and outputs — so a run indexes arrays instead
    /// of hashing names, which matters for the math decoder's 600-node step graph run per token.
    /// </summary>
    private sealed class Plan
    {
        public int SlotCount;
        public List<(int Slot, Tensor Value)> Constants = [];
        public int[] InputSlots = [];
        public int[] OutputSlots = [];
        public int[][] NodeInputs = [];
        public int[][] NodeOutputs = [];
        public int[][] Frees = [];
        public int[] LastUse = [];
        public bool[] IsOutput = [];
    }

    private readonly Lazy<Plan> _plan;
    private readonly Lazy<Plan> _fusedPlan;
    private readonly List<string> _folded = [];

    /// <summary>Node outputs computed once at load because every input was constant.</summary>
    public IReadOnlyList<string> FoldedValues => _folded;

    private Plan BuildPlan(List<OnnxNode> nodes)
    {
        var slots = new Dictionary<string, int>(StringComparer.Ordinal);
        int Slot(string name)
        {
            if (!slots.TryGetValue(name, out int s))
            {
                s = slots.Count;
                slots[name] = s;
            }

            return s;
        }

        var plan = new Plan();
        foreach (var (name, t) in _constants)
        {
            plan.Constants.Add((Slot(name), t));
        }

        plan.InputSlots = _graph.Inputs.Select(i => Slot(i.Name)).ToArray();
        plan.NodeInputs = nodes.Select(n => n.Inputs.Select(x => x.Length == 0 ? -1 : Slot(x)).ToArray()).ToArray();
        plan.NodeOutputs = nodes.Select(n => n.Outputs.Select(x => x.Length == 0 ? -1 : Slot(x)).ToArray()).ToArray();
        plan.OutputSlots = _graph.Outputs.Select(o => Slot(o.Name)).ToArray();
        plan.SlotCount = slots.Count;
        plan.LastUse = new int[plan.SlotCount];
        Array.Fill(plan.LastUse, -1);
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (int s in plan.NodeInputs[i])
            {
                if (s >= 0)
                {
                    plan.LastUse[s] = i;
                }
            }
        }

        plan.IsOutput = new bool[plan.SlotCount];
        foreach (int s in plan.OutputSlots)
        {
            plan.IsOutput[s] = true;
        }

        var frees = Enumerable.Range(0, nodes.Count).Select(_ => new List<int>()).ToArray();
        for (int s = 0; s < plan.SlotCount; s++)
        {
            if (plan.LastUse[s] >= 0 && !plan.IsOutput[s])
            {
                frees[plan.LastUse[s]].Add(s);
            }
        }

        plan.Frees = frees.Select(f => f.ToArray()).ToArray();
        return plan;
    }

    /// <summary>
    /// Folds the element-wise nodes that follow a convolution into it — what ONNX Runtime's
    /// <c>ConvActivationFusion</c> and <c>ConvAddFusion</c> do: <c>Conv → Add(residual)</c> and
    /// <c>Conv [→ Add] → HardSwish | Relu | Sigmoid</c>, wherever each intermediate value has
    /// exactly one reader and is not a graph output. The convolution then applies the residual and
    /// the activation to each stretch of its output while it is still in cache (see
    /// <c>Operators.Epilogue</c>): the same operations in the same order, so the outputs are
    /// identical, minus a full write and re-read of every intermediate tensor. The fused node takes
    /// the place of the last node it absorbed, where the residual is known to exist.
    /// </summary>
    private static List<OnnxNode> Fuse(List<OnnxNode> nodes, HashSet<string> graphOutputs)
    {
        var readers = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (string input in nodes[i].Inputs)
            {
                if (input.Length > 0)
                {
                    (readers.TryGetValue(input, out var l) ? l : readers[input] = []).Add(i);
                }
            }
        }

        bool SoleReader(string value, out int reader)
        {
            reader = -1;
            if (graphOutputs.Contains(value) || !readers.TryGetValue(value, out var l) || l.Count != 1)
            {
                return false;
            }

            reader = l[0];
            return true;
        }

        static bool Default(OnnxNode n) => n.Domain is "" or "ai.onnx";
        var removed = new bool[nodes.Count];
        var replacement = new OnnxNode?[nodes.Count];
        for (int i = 0; i < nodes.Count; i++)
        {
            OnnxNode conv = nodes[i];
            if (conv.OpType != "Conv" || !Default(conv) || conv.Outputs.Count != 1 || conv.Inputs.Count > 3 || removed[i])
            {
                continue;
            }

            string value = conv.Outputs[0];
            int at = i;
            string? residual = null;
            string act = "";
            if (SoleReader(value, out int j) && !removed[j] && nodes[j].OpType == "Add" && Default(nodes[j]) && nodes[j].Inputs.Count == 2
                && nodes[j].Inputs[0] != nodes[j].Inputs[1])
            {
                residual = nodes[j].Inputs[0] == value ? nodes[j].Inputs[1] : nodes[j].Inputs[0];
                value = nodes[j].Outputs[0];
                at = j;
            }

            if (SoleReader(value, out j) && !removed[j] && nodes[j].OpType is "HardSwish" or "Relu" or "Sigmoid" && Default(nodes[j]))
            {
                act = nodes[j].OpType;
                value = nodes[j].Outputs[0];
                at = j;
            }

            if (at == i)
            {
                continue;
            }

            var fused = new OnnxNode { Name = conv.Name, OpType = "Conv", Domain = conv.Domain };
            fused.Inputs.AddRange(conv.Inputs);
            if (residual is not null)
            {
                while (fused.Inputs.Count < 3)
                {
                    fused.Inputs.Add("");
                }

                fused.Inputs.Add(residual);
            }

            fused.Outputs.Add(value);
            foreach (var (k, a) in conv.Attributes)
            {
                fused.Attributes[k] = a;
            }

            if (act.Length > 0)
            {
                fused.Attributes[Operators.FusedActivation] = new OnnxAttribute { Name = Operators.FusedActivation, S = act };
            }

            removed[i] = true;
            if (residual is not null)
            {
                removed[readers[conv.Outputs[0]][0]] = true;
            }

            removed[at] = true;
            replacement[at] = fused;
        }

        var result = new List<OnnxNode>(nodes.Count);
        for (int i = 0; i < nodes.Count; i++)
        {
            if (replacement[i] is not null)
            {
                result.Add(replacement[i]!);
            }
            else if (!removed[i])
            {
                result.Add(nodes[i]);
            }
        }

        return result;
    }

    /// <summary>Runs the graph with inputs given in <see cref="InputNames"/> order.</summary>
    public Tensor[] Run(params Tensor[] inputs)
    {
        if (inputs.Length != InputNames.Count)
        {
            throw new ArgumentException($"The graph takes {InputNames.Count} inputs, got {inputs.Length}.", nameof(inputs));
        }

        var map = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        for (int i = 0; i < inputs.Length; i++)
        {
            map[InputNames[i]] = inputs[i];
        }

        return Run(map);
    }

    /// <summary>
    /// Runs, once at load, every node whose inputs are all constants (a transposed weight, a shape
    /// computed from constants) and keeps the result as a constant — what ONNX Runtime's constant
    /// folding does, so the math decoder does not re-transpose a weight on every token.
    /// </summary>
    private void FoldConstants()
    {
        var keep = new List<OnnxNode>(_nodes.Count);
        foreach (OnnxNode node in _nodes)
        {
            Operators.Kernel? kernel = Operators.Resolve(node);
            bool foldable = kernel is not null && node.Inputs.Count > 0
                && node.Inputs.All(i => i.Length == 0 || _constants.ContainsKey(i))
                && node.Outputs.All(o => !_graph.Outputs.Any(g => g.Name == o));
            if (!foldable)
            {
                keep.Add(node);
                continue;
            }

            var args = node.Inputs.Select(i => i.Length == 0 ? null : _constants[i]).ToList();
            Tensor[] results = kernel!(node, args);
            for (int o = 0; o < node.Outputs.Count && o < results.Length; o++)
            {
                _constants[node.Outputs[o]] = Materialize(results[o]);
                _folded.Add(node.Outputs[o]);
            }
        }

        _nodes.Clear();
        _nodes.AddRange(keep);
    }

    /// <summary>A pinned copy that owns exactly its data (a pooled array may be longer).</summary>
    private static Tensor Materialize(Tensor t) => t.Type switch
    {
        ElementType.Float => Tensor.FromFloats(t.ToFloatArray() is { Length: > 0 } f ? f : [0f], t.Shape),
        ElementType.Int64 => Tensor.FromLongs(t.ToLongArray() is { Length: > 0 } l ? l : [0L], t.Shape),
        _ => Tensor.FromBools(t.Bools.ToArray() is { Length: > 0 } b ? b : [false], t.Shape),
    };

    private static Tensor ConstantValue(OnnxNode node)
    {
        if (node.Attributes.TryGetValue("value", out var a) && a.T is not null)
        {
            return a.T;
        }

        if (node.Attributes.TryGetValue("value_float", out a))
        {
            return Tensor.FromFloats([a.F]);
        }

        if (node.Attributes.TryGetValue("value_floats", out a))
        {
            return Tensor.FromFloats([.. a.Floats], a.Floats.Count);
        }

        if (node.Attributes.TryGetValue("value_int", out a))
        {
            return Tensor.FromLongs([a.I]);
        }

        if (node.Attributes.TryGetValue("value_ints", out a))
        {
            return Tensor.FromLongs([.. a.Ints], a.Ints.Count);
        }

        throw new NotSupportedException($"Constant node '{node.Name}' has no supported value attribute.");
    }
}
