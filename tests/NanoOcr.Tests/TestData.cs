namespace NanoOcr.Tests;

/// <summary>Locates the test corpus, the reference dumps and the models.</summary>
public static class TestData
{
    public static string Root { get; } = FindRoot();

    public static string Documents => Path.Combine(Root, "tests", "data", "documents");

    public static string Layers => Path.Combine(Root, "tests", "data", "reference", "layers");

    public static string Pipeline => Path.Combine(Root, "tests", "data", "reference", "pipeline");

    private static string FindRoot()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "NanoOcr.slnx")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return dir ?? throw new InvalidOperationException("Cannot find the repository root (NanoOcr.slnx).");
    }
}
