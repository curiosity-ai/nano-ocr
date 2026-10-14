using System.Collections.Concurrent;
using System.Reflection;
using NanoOcr.Onnx;

namespace NanoOcr.Models;

/// <summary>
/// Where the models come from: the copies embedded in this assembly (the default), or a folder
/// laid out like the Hugging Face repository (<c>det.onnx</c>, <c>rec.onnx</c>, <c>meta.json</c>, …).
/// Sessions are parsed once and shared; running them is thread-safe.
/// </summary>
public sealed class ModelStore
{
    private const string ResourcePrefix = "NanoOcr.Models.";

    private static readonly Lazy<ModelStore> EmbeddedStore = new(() => new ModelStore(null));

    private readonly string? _directory;
    private readonly ConcurrentDictionary<string, Lazy<InferenceSession>> _sessions = new(StringComparer.Ordinal);

    private ModelStore(string? directory) => _directory = directory;

    /// <summary>The models compiled into the library.</summary>
    public static ModelStore Embedded => EmbeddedStore.Value;

    /// <summary>
    /// The models in <paramref name="directory"/>; <see langword="null"/> or empty gives the
    /// <c>NANO_OCR_MODEL_DIR</c> environment variable's folder (<c>INDRA_OCR_MODEL_DIR</c>, the
    /// original's name, is still honoured), or else the embedded models.
    /// </summary>
    public static ModelStore FromDirectory(string? directory)
    {
        directory = string.IsNullOrEmpty(directory) ? (Environment.GetEnvironmentVariable("NANO_OCR_MODEL_DIR") is { Length: > 0 } nano ? nano : Environment.GetEnvironmentVariable("INDRA_OCR_MODEL_DIR")) : directory;
        if (string.IsNullOrEmpty(directory))
        {
            return Embedded;
        }

        if (!Directory.Exists(directory))
        {
            throw new ModelException($"The model folder {directory} does not exist.");
        }

        return new ModelStore(directory);
    }

    /// <summary>Folder the models are read from, or <see langword="null"/> for the embedded set.</summary>
    public string? Folder => _directory;

    /// <summary>Whether the store holds <paramref name="file"/> (e.g. <c>orient.onnx</c>).</summary>
    public bool Has(string file) => _directory is null
        ? typeof(ModelStore).Assembly.GetManifestResourceInfo(ResourcePrefix + file) is not null
        : File.Exists(Path.Combine(_directory, file));

    /// <summary>Opens a model file for reading.</summary>
    public Stream Open(string file)
    {
        if (_directory is not null)
        {
            string path = Path.Combine(_directory, file);
            if (!File.Exists(path))
            {
                throw new ModelException($"The model folder {_directory} lacks {file}.");
            }

            return File.OpenRead(path);
        }

        return typeof(ModelStore).Assembly.GetManifestResourceStream(ResourcePrefix + file)
            ?? throw new ModelException($"The embedded model {file} is missing from {typeof(ModelStore).Assembly.GetName().Name}.");
    }

    /// <summary>Reads a text file (e.g. <c>meta.json</c>).</summary>
    public string ReadText(string file)
    {
        using var reader = new StreamReader(Open(file));
        return reader.ReadToEnd();
    }

    /// <summary>The shared session for model <paramref name="name"/> (without <c>.onnx</c>).</summary>
    public InferenceSession Session(string name) =>
        _sessions.GetOrAdd(name, n => new Lazy<InferenceSession>(() =>
        {
            using Stream s = Open(n + ".onnx");
            return new InferenceSession(OnnxModel.Load(s));
        })).Value;

    /// <summary>Whether model <paramref name="name"/> (without <c>.onnx</c>) is present.</summary>
    public bool HasModel(string name) => Has(name + ".onnx");
}

/// <summary>The models are not available: a missing folder or file.</summary>
public sealed class ModelException(string message) : IOException(message);
