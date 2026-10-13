using System.Diagnostics;

namespace IndraOcr.Onnx;

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

    /// <summary>Nodes that execute (constants folded away), in order.</summary>
    public IReadOnlyList<OnnxNode> Nodes => _nodes;

    /// <summary>Accumulates time per operator type when set.</summary>
    public Dictionary<string, double>? Profile { get; set; }

    /// <summary>
    /// Runs the graph. Returns the graph outputs, in <see cref="OutputNames"/> order. The caller
    /// owns the returned tensors (and may <see cref="Tensor.Release"/> them).
    /// </summary>
    /// <param name="inputs">Graph inputs by name. They are not released.</param>
    /// <param name="observer">
    /// Called with every node output as it is produced (name, tensor), before anything is
    /// released — the hook the per-layer parity tests use.
    /// </param>
    public Tensor[] Run(IReadOnlyDictionary<string, Tensor> inputs, Action<string, Tensor>? observer = null)
    {
        var values = new Dictionary<string, Tensor>(_constants, StringComparer.Ordinal);
        var owned = new HashSet<string>(StringComparer.Ordinal);
        foreach (OnnxValueInfo input in _graph.Inputs)
        {
            if (!inputs.TryGetValue(input.Name, out Tensor? t))
            {
                throw new ArgumentException($"Missing graph input '{input.Name}'.", nameof(inputs));
            }

            values[input.Name] = t;
        }

        var args = new List<Tensor?>(8);
        for (int i = 0; i < _nodes.Count; i++)
        {
            OnnxNode node = _nodes[i];
            args.Clear();
            foreach (string name in node.Inputs)
            {
                if (name.Length == 0)
                {
                    args.Add(null);
                }
                else if (values.TryGetValue(name, out Tensor? t))
                {
                    args.Add(t);
                }
                else
                {
                    throw new InvalidOperationException($"Node '{node.Name}' ({node.OpType}) reads '{name}', which no node produced.");
                }
            }

            long start = Profile is null ? 0 : Stopwatch.GetTimestamp();
            Tensor[] results;
            try
            {
                results = _kernels[i](node, args);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                string shapes = string.Join(", ", args.Select(a => a?.ToString() ?? "-"));
                throw new InvalidOperationException($"Node '{node.Name}' ({node.OpType}) failed on inputs {shapes}: {e.Message}", e);
            }

            if (Profile is not null)
            {
                double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                Profile[node.OpType] = Profile.GetValueOrDefault(node.OpType) + ms;
            }

            for (int o = 0; o < node.Outputs.Count && o < results.Length; o++)
            {
                string name = node.Outputs[o];
                if (name.Length == 0)
                {
                    results[o].Release();
                    continue;
                }

                values[name] = results[o];
                owned.Add(name);
                observer?.Invoke(name, results[o]);
                if (!_lastUse.ContainsKey(name) && !_graphOutputs.Contains(name))
                {
                    values.Remove(name);
                    owned.Remove(name);
                    results[o].Release();
                }
            }

            // Release every input whose last reader was this node.
            foreach (string name in node.Inputs)
            {
                if (name.Length > 0 && _lastUse.TryGetValue(name, out int last) && last == i
                    && owned.Remove(name) && !_graphOutputs.Contains(name))
                {
                    values[name].Release();
                    values.Remove(name);
                }
            }
        }

        var outputs = new Tensor[_graph.Outputs.Count];
        for (int o = 0; o < outputs.Length; o++)
        {
            string name = _graph.Outputs[o].Name;
            outputs[o] = values[name];
            owned.Remove(name);
        }

        foreach (string name in owned)
        {
            values[name].Release();
        }

        return outputs;
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
