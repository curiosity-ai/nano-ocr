using System.Buffers.Binary;

namespace NanoOcr.Onnx;

/// <summary>An attribute of an ONNX node.</summary>
public sealed class OnnxAttribute
{
    /// <summary>Attribute name.</summary>
    public string Name { get; set; } = "";

    /// <summary>Single float value.</summary>
    public float F { get; set; }

    /// <summary>Single integer value.</summary>
    public long I { get; set; }

    /// <summary>Single string value.</summary>
    public string? S { get; set; }

    /// <summary>Tensor value.</summary>
    public Tensor? T { get; set; }

    /// <summary>Subgraph value.</summary>
    public OnnxGraph? G { get; set; }

    /// <summary>Repeated float value.</summary>
    public List<float> Floats { get; } = [];

    /// <summary>Repeated integer value.</summary>
    public List<long> Ints { get; } = [];

    /// <summary>Repeated string value.</summary>
    public List<string> Strings { get; } = [];

    /// <summary>The ONNX AttributeType enum value.</summary>
    public int Kind { get; set; }
}

/// <summary>One operator of an ONNX graph.</summary>
public sealed class OnnxNode
{
    /// <summary>Node name (may be empty).</summary>
    public string Name { get; set; } = "";

    /// <summary>Operator type, e.g. <c>Conv</c>.</summary>
    public string OpType { get; set; } = "";

    /// <summary>Operator domain (empty for the default ONNX domain).</summary>
    public string Domain { get; set; } = "";

    /// <summary>Input value names; an empty string is an omitted optional input.</summary>
    public List<string> Inputs { get; } = [];

    /// <summary>Output value names.</summary>
    public List<string> Outputs { get; } = [];

    /// <summary>Attributes by name.</summary>
    public Dictionary<string, OnnxAttribute> Attributes { get; } = new(StringComparer.Ordinal);

    /// <summary>An integer attribute, or <paramref name="fallback"/>.</summary>
    public long Int(string name, long fallback) => Attributes.TryGetValue(name, out var a) ? a.I : fallback;

    /// <summary>A float attribute, or <paramref name="fallback"/>.</summary>
    public float Float(string name, float fallback) => Attributes.TryGetValue(name, out var a) ? a.F : fallback;

    /// <summary>A string attribute, or <paramref name="fallback"/>.</summary>
    public string Str(string name, string fallback) => Attributes.TryGetValue(name, out var a) && a.S is not null ? a.S : fallback;

    /// <summary>An integer-list attribute, or <see langword="null"/>.</summary>
    public long[]? Ints(string name) => Attributes.TryGetValue(name, out var a) ? [.. a.Ints] : null;

    /// <summary>A float-list attribute, or <see langword="null"/>.</summary>
    public float[]? Floats(string name) => Attributes.TryGetValue(name, out var a) ? [.. a.Floats] : null;

    /// <inheritdoc />
    public override string ToString() => $"{OpType} {Name}";
}

/// <summary>A declared graph input or output.</summary>
public sealed class OnnxValueInfo
{
    /// <summary>Value name.</summary>
    public string Name { get; set; } = "";

    /// <summary>ONNX element data type.</summary>
    public int DataType { get; set; }

    /// <summary>Dimensions; -1 where symbolic or unknown.</summary>
    public List<long> Dims { get; } = [];

    /// <summary>Symbolic dimension names (null where fixed).</summary>
    public List<string?> DimParams { get; } = [];
}

/// <summary>An ONNX graph: nodes in topological order, weights, inputs and outputs.</summary>
public sealed class OnnxGraph
{
    /// <summary>Graph name.</summary>
    public string Name { get; set; } = "";

    /// <summary>Nodes in topological order.</summary>
    public List<OnnxNode> Nodes { get; } = [];

    /// <summary>Weights by name.</summary>
    public Dictionary<string, Tensor> Initializers { get; } = new(StringComparer.Ordinal);

    /// <summary>Graph inputs (initializers excluded).</summary>
    public List<OnnxValueInfo> Inputs { get; } = [];

    /// <summary>Graph outputs.</summary>
    public List<OnnxValueInfo> Outputs { get; } = [];
}

/// <summary>A parsed ONNX model.</summary>
public sealed class OnnxModel
{
    /// <summary>The main graph.</summary>
    public OnnxGraph Graph { get; private set; } = new();

    /// <summary>Opset version of the default domain.</summary>
    public long Opset { get; private set; }

    /// <summary>Producer name recorded in the file.</summary>
    public string Producer { get; private set; } = "";

    /// <summary>Loads a model from a file.</summary>
    public static OnnxModel Load(string path) => Parse(File.ReadAllBytes(path));

    /// <summary>Loads a model from a stream.</summary>
    public static OnnxModel Load(Stream stream)
    {
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return Parse(ms.GetBuffer().AsSpan(0, (int)ms.Length));
    }

    /// <summary>Parses the bytes of a <c>ModelProto</c>.</summary>
    public static OnnxModel Parse(ReadOnlySpan<byte> bytes)
    {
        var model = new OnnxModel();
        var r = new ProtoReader(bytes);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 2 when wire == 2:
                    model.Producer = r.ReadString();
                    break;
                case 7 when wire == 2:
                    model.Graph = ParseGraph(r.ReadBytes());
                    break;
                case 8 when wire == 2:
                    {
                        var op = new ProtoReader(r.ReadBytes());
                        string domain = "";
                        long version = 0;
                        while (op.Next(out int f, out int w))
                        {
                            if (f == 1 && w == 2)
                            {
                                domain = op.ReadString();
                            }
                            else if (f == 2 && w == 0)
                            {
                                version = op.ReadInt64();
                            }
                            else
                            {
                                op.Skip(w);
                            }
                        }

                        if (domain is "" or "ai.onnx")
                        {
                            model.Opset = version;
                        }

                        break;
                    }

                default:
                    r.Skip(wire);
                    break;
            }
        }

        return model;
    }

    private static OnnxGraph ParseGraph(ReadOnlySpan<byte> bytes)
    {
        var graph = new OnnxGraph();
        var inputs = new List<OnnxValueInfo>();
        var r = new ProtoReader(bytes);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    graph.Nodes.Add(ParseNode(r.ReadBytes()));
                    break;
                case 2 when wire == 2:
                    graph.Name = r.ReadString();
                    break;
                case 5 when wire == 2:
                    {
                        (string name, Tensor t) = ParseTensor(r.ReadBytes());
                        graph.Initializers[name] = t.Pin();
                        break;
                    }

                case 11 when wire == 2:
                    inputs.Add(ParseValueInfo(r.ReadBytes()));
                    break;
                case 12 when wire == 2:
                    graph.Outputs.Add(ParseValueInfo(r.ReadBytes()));
                    break;
                default:
                    r.Skip(wire);
                    break;
            }
        }

        graph.Inputs.AddRange(inputs.Where(i => !graph.Initializers.ContainsKey(i.Name)));
        return graph;
    }

    private static OnnxValueInfo ParseValueInfo(ReadOnlySpan<byte> bytes)
    {
        var info = new OnnxValueInfo();
        var r = new ProtoReader(bytes);
        while (r.Next(out int field, out int wire))
        {
            if (field == 1 && wire == 2)
            {
                info.Name = r.ReadString();
            }
            else if (field == 2 && wire == 2)
            {
                // TypeProto -> tensor_type (1) -> elem_type (1), shape (2) -> dim (1)
                var type = new ProtoReader(r.ReadBytes());
                while (type.Next(out int tf, out int tw))
                {
                    if (tf != 1 || tw != 2)
                    {
                        type.Skip(tw);
                        continue;
                    }

                    var tensor = new ProtoReader(type.ReadBytes());
                    while (tensor.Next(out int ef, out int ew))
                    {
                        if (ef == 1 && ew == 0)
                        {
                            info.DataType = (int)tensor.ReadVarint();
                        }
                        else if (ef == 2 && ew == 2)
                        {
                            var shape = new ProtoReader(tensor.ReadBytes());
                            while (shape.Next(out int sf, out int sw))
                            {
                                if (sf != 1 || sw != 2)
                                {
                                    shape.Skip(sw);
                                    continue;
                                }

                                var dim = new ProtoReader(shape.ReadBytes());
                                long value = -1;
                                string? param = null;
                                while (dim.Next(out int df, out int dw))
                                {
                                    if (df == 1 && dw == 0)
                                    {
                                        value = dim.ReadInt64();
                                    }
                                    else if (df == 2 && dw == 2)
                                    {
                                        param = dim.ReadString();
                                    }
                                    else
                                    {
                                        dim.Skip(dw);
                                    }
                                }

                                info.Dims.Add(value);
                                info.DimParams.Add(param);
                            }
                        }
                        else
                        {
                            tensor.Skip(ew);
                        }
                    }
                }
            }
            else
            {
                r.Skip(wire);
            }
        }

        return info;
    }

    private static OnnxNode ParseNode(ReadOnlySpan<byte> bytes)
    {
        var node = new OnnxNode();
        var r = new ProtoReader(bytes);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    node.Inputs.Add(r.ReadString());
                    break;
                case 2 when wire == 2:
                    node.Outputs.Add(r.ReadString());
                    break;
                case 3 when wire == 2:
                    node.Name = r.ReadString();
                    break;
                case 4 when wire == 2:
                    node.OpType = r.ReadString();
                    break;
                case 7 when wire == 2:
                    node.Domain = r.ReadString();
                    break;
                case 5 when wire == 2:
                    {
                        OnnxAttribute a = ParseAttribute(r.ReadBytes());
                        node.Attributes[a.Name] = a;
                        break;
                    }

                default:
                    r.Skip(wire);
                    break;
            }
        }

        return node;
    }

    private static OnnxAttribute ParseAttribute(ReadOnlySpan<byte> bytes)
    {
        var a = new OnnxAttribute();
        var r = new ProtoReader(bytes);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    a.Name = r.ReadString();
                    break;
                case 20 when wire == 0:
                    a.Kind = (int)r.ReadVarint();
                    break;
                case 2 when wire == 5:
                    a.F = r.ReadFixed32Float();
                    break;
                case 3 when wire == 0:
                    a.I = r.ReadInt64();
                    break;
                case 4 when wire == 2:
                    a.S = System.Text.Encoding.UTF8.GetString(r.ReadBytes());
                    break;
                case 5 when wire == 2:
                    a.T = ParseTensor(r.ReadBytes()).Tensor.Pin();
                    break;
                case 6 when wire == 2:
                    a.G = ParseGraph(r.ReadBytes());
                    break;
                case 7:
                    r.ReadFloats(wire, a.Floats);
                    break;
                case 8:
                    r.ReadInt64s(wire, a.Ints);
                    break;
                case 9 when wire == 2:
                    a.Strings.Add(r.ReadString());
                    break;
                default:
                    r.Skip(wire);
                    break;
            }
        }

        return a;
    }

    /// <summary>ONNX TensorProto.DataType values.</summary>
    internal static class DataTypes
    {
        public const int Float = 1, Uint8 = 2, Int8 = 3, Uint16 = 4, Int16 = 5, Int32 = 6, Int64 = 7,
            String = 8, Bool = 9, Float16 = 10, Double = 11, Uint32 = 12, Uint64 = 13, BFloat16 = 16;
    }

    /// <summary>The interpreter's element type for an ONNX data type.</summary>
    internal static ElementType ElementTypeOf(int dataType) => dataType switch
    {
        DataTypes.Float or DataTypes.Float16 or DataTypes.Double or DataTypes.BFloat16 => ElementType.Float,
        DataTypes.Bool => ElementType.Bool,
        _ => ElementType.Int64,
    };

    internal static (string Name, Tensor Tensor) ParseTensor(ReadOnlySpan<byte> bytes)
    {
        var dims = new List<long>();
        int dataType = 0;
        string name = "";
        ReadOnlySpan<byte> raw = default;
        var floats = new List<float>();
        var longs = new List<long>();
        var doubles = new List<double>();
        var r = new ProtoReader(bytes);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1:
                    r.ReadInt64s(wire, dims);
                    break;
                case 2 when wire == 0:
                    dataType = (int)r.ReadVarint();
                    break;
                case 4:
                    r.ReadFloats(wire, floats);
                    break;
                case 5:
                case 7:
                case 11:
                    r.ReadInt64s(wire, longs);
                    break;
                case 8 when wire == 2:
                    name = r.ReadString();
                    break;
                case 9 when wire == 2:
                    raw = r.ReadBytes();
                    break;
                case 10:
                    r.ReadDoubles(wire, doubles);
                    break;
                case 14 when wire == 0:
                    if (r.ReadVarint() != 0)
                    {
                        throw new NotSupportedException($"Tensor '{name}' uses external data, which is not supported.");
                    }

                    break;
                default:
                    r.Skip(wire);
                    break;
            }
        }

        int[] shape = dims.Select(d => checked((int)d)).ToArray();
        int n = Tensor.SizeOf(shape);
        ElementType type = ElementTypeOf(dataType);
        Tensor t;
        switch (type)
        {
            case ElementType.Float:
                {
                    var data = new float[Math.Max(n, 1)];
                    if (!raw.IsEmpty)
                    {
                        DecodeRawFloats(dataType, raw, data.AsSpan(0, n));
                    }
                    else if (dataType == DataTypes.Double)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            data[i] = (float)doubles[i];
                        }
                    }
                    else if (dataType is DataTypes.Float16 or DataTypes.BFloat16)
                    {
                        // Stored in int32_data, one 16-bit pattern per element.
                        for (int i = 0; i < n; i++)
                        {
                            ushort bits = (ushort)longs[i];
                            data[i] = dataType == DataTypes.Float16
                                ? (float)BitConverter.UInt16BitsToHalf(bits)
                                : BitConverter.Int32BitsToSingle(bits << 16);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < n; i++)
                        {
                            data[i] = floats[i];
                        }
                    }

                    t = Tensor.FromFloats(data, shape);
                    break;
                }

            case ElementType.Bool:
                {
                    var data = new bool[Math.Max(n, 1)];
                    for (int i = 0; i < n; i++)
                    {
                        data[i] = raw.IsEmpty ? longs[i] != 0 : raw[i] != 0;
                    }

                    t = Tensor.FromBools(data, shape);
                    break;
                }

            default:
                {
                    var data = new long[Math.Max(n, 1)];
                    if (!raw.IsEmpty)
                    {
                        DecodeRawIntegers(dataType, raw, data.AsSpan(0, n));
                    }
                    else
                    {
                        for (int i = 0; i < n; i++)
                        {
                            data[i] = longs[i];
                        }
                    }

                    t = Tensor.FromLongs(data, shape);
                    break;
                }
        }

        return (name, t);
    }

    private static void DecodeRawFloats(int dataType, ReadOnlySpan<byte> raw, Span<float> target)
    {
        switch (dataType)
        {
            case DataTypes.Float:
                for (int i = 0; i < target.Length; i++)
                {
                    target[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.Slice(i * 4, 4));
                }

                break;
            case DataTypes.Double:
                for (int i = 0; i < target.Length; i++)
                {
                    target[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(raw.Slice(i * 8, 8));
                }

                break;
            case DataTypes.Float16:
                for (int i = 0; i < target.Length; i++)
                {
                    target[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.Slice(i * 2, 2));
                }

                break;
            case DataTypes.BFloat16:
                for (int i = 0; i < target.Length; i++)
                {
                    target[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(i * 2, 2)) << 16);
                }

                break;
            default:
                throw new NotSupportedException($"Float data type {dataType}.");
        }
    }

    private static void DecodeRawIntegers(int dataType, ReadOnlySpan<byte> raw, Span<long> target)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = dataType switch
            {
                DataTypes.Int64 => BinaryPrimitives.ReadInt64LittleEndian(raw.Slice(i * 8, 8)),
                DataTypes.Uint64 => (long)BinaryPrimitives.ReadUInt64LittleEndian(raw.Slice(i * 8, 8)),
                DataTypes.Int32 => BinaryPrimitives.ReadInt32LittleEndian(raw.Slice(i * 4, 4)),
                DataTypes.Uint32 => BinaryPrimitives.ReadUInt32LittleEndian(raw.Slice(i * 4, 4)),
                DataTypes.Int16 => BinaryPrimitives.ReadInt16LittleEndian(raw.Slice(i * 2, 2)),
                DataTypes.Uint16 => BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(i * 2, 2)),
                DataTypes.Int8 => (sbyte)raw[i],
                DataTypes.Uint8 => raw[i],
                _ => throw new NotSupportedException($"Integer data type {dataType}."),
            };
        }
    }
}
