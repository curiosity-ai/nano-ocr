using System.Buffers.Binary;
using System.Text;

namespace NanoOcr.Onnx;

/// <summary>
/// A minimal forward-only reader of the protocol-buffers wire format, enough to decode an ONNX
/// <c>ModelProto</c> without a dependency on a protobuf runtime (and without reflection, so the
/// library stays trimmable and native-AOT publishable).
/// </summary>
internal ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _position;

    public ProtoReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _position = 0;
    }

    public readonly bool End => _position >= _data.Length;

    /// <summary>Reads the next field key; returns false at the end of the message.</summary>
    public bool Next(out int field, out int wireType)
    {
        if (_position >= _data.Length)
        {
            field = 0;
            wireType = 0;
            return false;
        }

        ulong key = ReadVarint();
        field = (int)(key >> 3);
        wireType = (int)(key & 7);
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            byte b = _data[_position++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return result;
            }

            shift += 7;
            if (shift > 63)
            {
                throw new InvalidDataException("Malformed varint in ONNX model.");
            }
        }
    }

    public long ReadInt64() => (long)ReadVarint();

    public float ReadFixed32Float()
    {
        float value = BinaryPrimitives.ReadSingleLittleEndian(_data.Slice(_position, 4));
        _position += 4;
        return value;
    }

    public double ReadFixed64Double()
    {
        double value = BinaryPrimitives.ReadDoubleLittleEndian(_data.Slice(_position, 8));
        _position += 8;
        return value;
    }

    public ReadOnlySpan<byte> ReadBytes()
    {
        int length = checked((int)ReadVarint());
        ReadOnlySpan<byte> slice = _data.Slice(_position, length);
        _position += length;
        return slice;
    }

    /// <summary>Offset of the next byte in the underlying buffer.</summary>
    public readonly int Position => _position;

    public string ReadString() => Encoding.UTF8.GetString(ReadBytes());

    public void Skip(int wireType)
    {
        switch (wireType)
        {
            case 0:
                ReadVarint();
                break;
            case 1:
                _position += 8;
                break;
            case 2:
                ReadBytes();
                break;
            case 5:
                _position += 4;
                break;
            default:
                throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
    }

    /// <summary>Reads a repeated int64 field that may be packed (wire type 2) or not.</summary>
    public void ReadInt64s(int wireType, List<long> target)
    {
        if (wireType == 2)
        {
            var inner = new ProtoReader(ReadBytes());
            while (!inner.End)
            {
                target.Add(inner.ReadInt64());
            }
        }
        else
        {
            target.Add(ReadInt64());
        }
    }

    /// <summary>Reads a repeated float field that may be packed (wire type 2) or not.</summary>
    public void ReadFloats(int wireType, List<float> target)
    {
        if (wireType == 2)
        {
            ReadOnlySpan<byte> bytes = ReadBytes();
            for (int i = 0; i + 4 <= bytes.Length; i += 4)
            {
                target.Add(BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice(i, 4)));
            }
        }
        else
        {
            target.Add(ReadFixed32Float());
        }
    }

    /// <summary>Reads a repeated double field that may be packed (wire type 2) or not.</summary>
    public void ReadDoubles(int wireType, List<double> target)
    {
        if (wireType == 2)
        {
            ReadOnlySpan<byte> bytes = ReadBytes();
            for (int i = 0; i + 8 <= bytes.Length; i += 8)
            {
                target.Add(BinaryPrimitives.ReadDoubleLittleEndian(bytes.Slice(i, 8)));
            }
        }
        else
        {
            target.Add(ReadFixed64Double());
        }
    }
}
