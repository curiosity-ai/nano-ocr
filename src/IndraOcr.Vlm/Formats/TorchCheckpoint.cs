using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using IndraOcr.Vlm.Core;

namespace IndraOcr.Vlm.Formats;

/// <summary>
/// Reader for PyTorch's zip checkpoint format — what <c>torch.save</c> writes since 1.6 — without
/// torch: a ZIP holding <c>&lt;prefix&gt;/data.pkl</c> (a pickle of the saved object) and one
/// <c>&lt;prefix&gt;/data/&lt;key&gt;</c> entry per tensor storage.
/// </summary>
/// <remarks>
/// <para>
/// The pickle is evaluated by a small stack machine that implements the protocol-2 opcodes
/// <c>torch.save</c> emits and exactly the globals a state dict needs: <c>collections.OrderedDict</c>,
/// <c>torch._utils._rebuild_tensor_v2</c> (a strided view of a storage) and the
/// <c>torch.*Storage</c> type tags carried by the persistent ids. Anything else — a pickled
/// optimizer, an arbitrary class — is refused with the global's name rather than half-understood;
/// <c>weights_only=True</c> draws the same line. Scalars, strings, tuples, lists and dicts beside the
/// tensors (the order checkpoint's <c>d</c>, <c>layers</c>, <c>pair_acc</c>, ...) come back as plain
/// CLR values.
/// </para>
/// <para>
/// A persistent id is <c>('storage', storage_type, key, location, numel)</c>; the storage's bytes are
/// the entry <c>data/&lt;key&gt;</c>, little-endian (the archive's <c>byteorder</c> record is checked).
/// A tensor is <c>_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)</c>; a
/// contiguous one is exposed as a slice of its storage, anything else is gathered into a fresh
/// contiguous buffer.
/// </para>
/// </remarks>
public sealed class TorchCheckpoint
{
    private TorchCheckpoint(object? root) => Root = root;

    /// <summary>The unpickled object: dictionaries are <see cref="PyDict"/>, tensors <see cref="WeightTensor"/>.</summary>
    public object? Root { get; }

    /// <summary>Reads a checkpoint written by <c>torch.save</c>.</summary>
    /// <param name="path">The <c>.pt</c> / <c>.pth</c> file.</param>
    public static TorchCheckpoint Load(string path)
    {
        using ZipArchive zip = ZipFile.OpenRead(path);
        ZipArchiveEntry pickle = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal) || e.FullName == "data.pkl")
            ?? throw new InvalidDataException($"{path} is not a torch zip checkpoint (no data.pkl).");
        string prefix = pickle.FullName[..^"data.pkl".Length];

        ZipArchiveEntry? order = zip.GetEntry(prefix + "byteorder");
        if (order is not null && ReadAll(order) is { } bo && Encoding.ASCII.GetString(bo).Trim() != "little")
        {
            throw new NotSupportedException($"{path} was saved big-endian.");
        }

        var storages = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        byte[] Storage(string key)
        {
            if (!storages.TryGetValue(key, out byte[]? bytes))
            {
                ZipArchiveEntry entry = zip.GetEntry(prefix + "data/" + key)
                    ?? throw new InvalidDataException($"{path}: storage '{key}' is missing.");
                bytes = ReadAll(entry);
                storages[key] = bytes;
            }

            return bytes;
        }

        var machine = new Unpickler(ReadAll(pickle), Storage);
        return new TorchCheckpoint(machine.Run());
    }

    /// <summary>
    /// The tensors of the state dict at <paramref name="key"/> of the root dictionary (or the root
    /// itself when <paramref name="key"/> is <see langword="null"/>), by parameter name.
    /// </summary>
    public Dictionary<string, WeightTensor> StateDict(string? key = "model")
    {
        object? node = Root;
        if (key is not null)
        {
            node = (Root as PyDict ?? throw new InvalidDataException("The checkpoint's root is not a dict."))[key];
        }

        var dict = node as PyDict ?? throw new InvalidDataException($"'{key}' is not a dict.");
        var result = new Dictionary<string, WeightTensor>(StringComparer.Ordinal);
        foreach ((object k, object? v) in dict.Items)
        {
            if (k is string name && v is WeightTensor t)
            {
                result[name] = t with { Name = name };
            }
        }

        return result;
    }

    /// <summary>A scalar entry of the root dictionary, or <paramref name="fallback"/> when absent.</summary>
    public long GetInt(string key, long fallback) =>
        Root is PyDict d && d.TryGetValue(key, out object? v) && v is long l ? l : fallback;

    private static byte[] ReadAll(ZipArchiveEntry entry)
    {
        using Stream s = entry.Open();
        byte[] buffer = new byte[entry.Length];
        s.ReadExactly(buffer);
        return buffer;
    }

    /// <summary>An insertion-ordered Python dict.</summary>
    public sealed class PyDict
    {
        private readonly Dictionary<object, object?> _map = [];
        private readonly List<object> _keys = [];

        /// <summary>The entries in insertion order.</summary>
        public IEnumerable<(object Key, object? Value)> Items => _keys.Select(k => (k, _map[k]));

        /// <summary>The value at <paramref name="key"/>.</summary>
        public object? this[object key]
        {
            get => _map[key];
            set
            {
                if (!_map.ContainsKey(key))
                {
                    _keys.Add(key);
                }

                _map[key] = value;
            }
        }

        /// <summary>Looks up <paramref name="key"/>.</summary>
        public bool TryGetValue(object key, out object? value) => _map.TryGetValue(key, out value);
    }

    /// <summary>A <c>module.name</c> reference produced by <c>GLOBAL</c>.</summary>
    private sealed record PyGlobal(string Module, string Name)
    {
        public string FullName => Module + "." + Name;
    }

    /// <summary>A storage resolved from a persistent id.</summary>
    private sealed record StorageRef(byte[] Bytes, DType Dtype);

    private sealed class Mark;

    /// <summary>The pickle stack machine (the subset of protocol 2+ that <c>torch.save</c> emits).</summary>
    private sealed class Unpickler(byte[] data, Func<string, byte[]> storage)
    {
        private readonly List<object?> _stack = [];
        private readonly Dictionary<long, object?> _memo = [];
        private static readonly Mark MarkObject = new();
        private int _pos;

        public object? Run()
        {
            while (true)
            {
                byte op = data[_pos++];
                switch (op)
                {
                    case 0x80: _pos++; break; // PROTO
                    case 0x95: _pos += 8; break; // FRAME
                    case 0x2e: return Pop(); // STOP
                    case 0x28: Push(MarkObject); break; // MARK
                    case 0x4e: Push(null); break; // NONE
                    case 0x88: Push(true); break; // NEWTRUE
                    case 0x89: Push(false); break; // NEWFALSE
                    case 0x4b: Push((long)data[_pos++]); break; // BININT1
                    case 0x4d: Push((long)BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(_pos))); _pos += 2; break; // BININT2
                    case 0x4a: Push((long)BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(_pos))); _pos += 4; break; // BININT
                    case 0x8a: // LONG1
                    {
                        int n = data[_pos++];
                        long v = 0;
                        for (int i = n - 1; i >= 0; i--)
                        {
                            v = (v << 8) | data[_pos + i];
                        }

                        if (n is > 0 and < 8 && (data[_pos + n - 1] & 0x80) != 0)
                        {
                            v -= 1L << (8 * n);
                        }

                        _pos += n;
                        Push(v);
                        break;
                    }

                    case 0x47: Push(BinaryPrimitives.ReadDoubleBigEndian(data.AsSpan(_pos))); _pos += 8; break; // BINFLOAT
                    case 0x58: Push(ReadString(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(_pos)), 4)); break; // BINUNICODE
                    case 0x8c: Push(ReadString(data[_pos], 1)); break; // SHORT_BINUNICODE
                    case 0x8d: Push(ReadString((int)BinaryPrimitives.ReadInt64LittleEndian(data.AsSpan(_pos)), 8)); break; // BINUNICODE8
                    case 0x43: Push(ReadBytes(data[_pos], 1)); break; // SHORT_BINBYTES
                    case 0x42: Push(ReadBytes(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(_pos)), 4)); break; // BINBYTES
                    case 0x7d: Push(new PyDict()); break; // EMPTY_DICT
                    case 0x5d: Push(new List<object?>()); break; // EMPTY_LIST
                    case 0x29: Push(Array.Empty<object?>()); break; // EMPTY_TUPLE
                    case 0x85: Push(new[] { Pop() }); break; // TUPLE1
                    case 0x86: { object? b = Pop(), a = Pop(); Push(new[] { a, b }); break; } // TUPLE2
                    case 0x87: { object? c = Pop(), b = Pop(), a = Pop(); Push(new[] { a, b, c }); break; } // TUPLE3
                    case 0x74: Push(PopMark().ToArray()); break; // TUPLE
                    case 0x6c: Push(PopMark()); break; // LIST
                    case 0x61: { object? v = Pop(); ((List<object?>)Peek()!).Add(v); break; } // APPEND
                    case 0x65: { List<object?> items = PopMark(); ((List<object?>)Peek()!).AddRange(items); break; } // APPENDS
                    case 0x73: { object? v = Pop(), k = Pop(); ((PyDict)Peek()!)[k!] = v; break; } // SETITEM
                    case 0x75: // SETITEMS
                    {
                        List<object?> items = PopMark();
                        var dict = (PyDict)Peek()!;
                        for (int i = 0; i < items.Count; i += 2)
                        {
                            dict[items[i]!] = items[i + 1];
                        }

                        break;
                    }

                    case 0x71: _memo[data[_pos++]] = Peek(); break; // BINPUT
                    case 0x72: _memo[BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(_pos))] = Peek(); _pos += 4; break; // LONG_BINPUT
                    case 0x94: _memo[_memo.Count] = Peek(); break; // MEMOIZE
                    case 0x68: Push(_memo[data[_pos++]]); break; // BINGET
                    case 0x6a: Push(_memo[BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(_pos))]); _pos += 4; break; // LONG_BINGET
                    case 0x63: // GLOBAL
                    {
                        string module = ReadLine();
                        string name = ReadLine();
                        Push(new PyGlobal(module, name));
                        break;
                    }

                    case 0x93: { var name = (string)Pop()!; var module = (string)Pop()!; Push(new PyGlobal(module, name)); break; } // STACK_GLOBAL
                    case 0x51: Push(PersistentLoad(Pop())); break; // BINPERSID
                    case 0x52: { var args = (object?[])Pop()!; object? f = Pop(); Push(Reduce(f, args)); break; } // REDUCE
                    case 0x81: { var args = (object?[])Pop()!; object? cls = Pop(); Push(Reduce(cls, args)); break; } // NEWOBJ
                    case 0x62: Pop(); break; // BUILD: a state dict's `_metadata`; nothing needs it
                    default:
                        throw new NotSupportedException($"Pickle opcode 0x{op:x2} at {_pos - 1} is not supported.");
                }
            }
        }

        private object PersistentLoad(object? pid)
        {
            if (pid is not object?[] { Length: >= 5 } t || t[0] is not "storage")
            {
                throw new NotSupportedException("Unknown persistent id in the checkpoint.");
            }

            var type = (PyGlobal)t[1]!;
            DType dtype = type.Name switch
            {
                "FloatStorage" => DType.Float32,
                "DoubleStorage" => DType.Float64,
                "HalfStorage" => DType.Float16,
                "BFloat16Storage" => DType.BFloat16,
                "LongStorage" => DType.Int64,
                "IntStorage" => DType.Int32,
                "CharStorage" => DType.Int8,
                "ByteStorage" => DType.UInt8,
                "BoolStorage" => DType.Bool,
                _ => throw new NotSupportedException($"Storage type {type.FullName} is not supported."),
            };
            return new StorageRef(storage((string)t[2]!), dtype);
        }

        private static object? Reduce(object? callable, object?[] args)
        {
            var g = callable as PyGlobal ?? throw new NotSupportedException("REDUCE of a non-global.");
            switch (g.FullName)
            {
                case "collections.OrderedDict":
                case "builtins.dict":
                    return new PyDict();
                case "torch._utils._rebuild_tensor_v2":
                case "torch._utils._rebuild_tensor":
                    return RebuildTensor((StorageRef)args[0]!, (long)args[1]!, (object?[])args[2]!, (object?[])args[3]!);
                default:
                    throw new NotSupportedException($"The checkpoint pickles {g.FullName}, which this reader does not evaluate.");
            }
        }

        private static WeightTensor RebuildTensor(StorageRef storage, long offset, object?[] size, object?[] stride)
        {
            int[] shape = size.Select(s => checked((int)(long)s!)).ToArray();
            long[] strides = stride.Select(s => (long)s!).ToArray();
            int elementSize = storage.Dtype.ByteSize();
            int count = Tensor.ElementCount(shape);

            bool contiguous = true;
            long expected = 1;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                if (shape[d] != 1 && strides[d] != expected)
                {
                    contiguous = false;
                }

                expected *= shape[d];
            }

            if (contiguous)
            {
                return new WeightTensor(string.Empty, storage.Dtype, shape, storage.Bytes.AsMemory(checked((int)(offset * elementSize)), count * elementSize));
            }

            // A strided view: gather it, element by element, into its own buffer.
            byte[] gathered = new byte[count * elementSize];
            int[] index = new int[shape.Length];
            for (int i = 0; i < count; i++)
            {
                long source = offset;
                for (int d = 0; d < shape.Length; d++)
                {
                    source += index[d] * strides[d];
                }

                storage.Bytes.AsSpan(checked((int)(source * elementSize)), elementSize).CopyTo(gathered.AsSpan(i * elementSize));
                for (int d = shape.Length - 1; d >= 0; d--)
                {
                    if (++index[d] < shape[d])
                    {
                        break;
                    }

                    index[d] = 0;
                }
            }

            return new WeightTensor(string.Empty, storage.Dtype, shape, gathered);
        }

        private void Push(object? value) => _stack.Add(value);

        private object? Pop()
        {
            object? v = _stack[^1];
            _stack.RemoveAt(_stack.Count - 1);
            return v;
        }

        private object? Peek() => _stack[^1];

        private List<object?> PopMark()
        {
            int mark = _stack.FindLastIndex(o => ReferenceEquals(o, MarkObject));
            List<object?> items = _stack.GetRange(mark + 1, _stack.Count - mark - 1);
            _stack.RemoveRange(mark, _stack.Count - mark);
            return items;
        }

        private string ReadString(int length, int prefix)
        {
            _pos += prefix;
            string s = Encoding.UTF8.GetString(data, _pos, length);
            _pos += length;
            return s;
        }

        private byte[] ReadBytes(int length, int prefix)
        {
            _pos += prefix;
            byte[] b = data.AsSpan(_pos, length).ToArray();
            _pos += length;
            return b;
        }

        private string ReadLine()
        {
            int end = Array.IndexOf(data, (byte)'\n', _pos);
            string s = Encoding.ASCII.GetString(data, _pos, end - _pos);
            _pos = end + 1;
            return s;
        }
    }
}
