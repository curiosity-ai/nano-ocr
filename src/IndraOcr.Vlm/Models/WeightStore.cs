using System.Collections.Concurrent;
using System.Text.Json;
using IndraOcr.Vlm.Core;
using IndraOcr.Vlm.Formats;

namespace IndraOcr.Vlm.Models;

/// <summary>
/// The tensors of a checkpoint directory — one <c>model.safetensors</c>, or the shards a
/// <c>model.safetensors.index.json</c> names — memory-mapped and kept in their on-disk dtype.
/// </summary>
public sealed class WeightStore : IDisposable
{
    private readonly SafetensorsFile[] _files;
    private readonly Dictionary<string, SafetensorsFile> _byName = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, float[]> _vectors = new(StringComparer.Ordinal);

    private WeightStore(SafetensorsFile[] files)
    {
        _files = files;
        foreach (SafetensorsFile file in files)
        {
            foreach (string name in file.Names)
            {
                _byName[name] = file;
            }
        }
    }

    /// <summary>Opens the checkpoint in <paramref name="directory"/>.</summary>
    public static WeightStore Open(string directory)
    {
        string single = Path.Combine(directory, "model.safetensors");
        if (File.Exists(single))
        {
            return new WeightStore([SafetensorsFile.Open(single)]);
        }

        string index = Path.Combine(directory, "model.safetensors.index.json");
        if (!File.Exists(index))
        {
            throw new FileNotFoundException($"No model.safetensors or index in '{directory}'.");
        }

        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(index));
        string[] shards = document.RootElement.GetProperty("weight_map").EnumerateObject()
            .Select(p => p.Value.GetString()!).Distinct().Order(StringComparer.Ordinal).ToArray();
        return new WeightStore(shards.Select(s => SafetensorsFile.Open(Path.Combine(directory, s))).ToArray());
    }

    /// <summary>Whether a tensor exists.</summary>
    public bool Contains(string name) => _byName.ContainsKey(name);

    /// <summary>A tensor's raw view.</summary>
    public WeightTensor Tensor(string name) =>
        _byName.TryGetValue(name, out SafetensorsFile? file) ? file[name] : throw new KeyNotFoundException(name);

    /// <summary>A 2-D weight, <c>[out, in]</c>, in its on-disk dtype.</summary>
    public WeightMatrix Matrix(string name) => Tensor(name).AsMatrix();

    /// <summary>A weight of any rank viewed as <c>[shape[0], rest]</c>.</summary>
    public WeightMatrix MatrixFlattened(string name)
    {
        WeightTensor tensor = Tensor(name);
        int rows = tensor.Shape[0];
        return tensor.AsMatrix(rows, tensor.ElementCount / rows);
    }

    /// <summary>A small tensor widened to float32 (cached).</summary>
    public float[] Vector(string name) => _vectors.GetOrAdd(name, static (key, self) => self.Tensor(key).ToFloats(), this);

    /// <inheritdoc/>
    public void Dispose()
    {
        _vectors.Clear();
        foreach (SafetensorsFile file in _files)
        {
            file.Dispose();
        }
    }
}
