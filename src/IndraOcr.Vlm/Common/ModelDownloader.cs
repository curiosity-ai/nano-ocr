using System.Net;
using System.Net.Http.Headers;

namespace IndraOcr.Vlm.Common;

/// <summary>
/// Fetches checkpoint files from the Hugging Face Hub (or a mirror) into a local cache, resuming
/// interrupted transfers. The vision-language checkpoints are 1.4 GB (Surya 2) and 10.6 GB
/// (Chandra 2) — too large to embed in an assembly — so they are downloaded once on first use.
/// </summary>
public static class ModelDownloader
{
    /// <summary>The cache root: <c>INDRA_OCR_CACHE</c>, else <c>~/.cache/indra-ocr</c>.</summary>
    public static string CacheRoot =>
        Environment.GetEnvironmentVariable("INDRA_OCR_CACHE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "indra-ocr");

    /// <summary>The Hub endpoint: <c>HF_ENDPOINT</c>, else https://huggingface.co.</summary>
    public static string Endpoint => (Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? "https://huggingface.co").TrimEnd('/');

    /// <summary>
    /// Ensures <paramref name="files"/> of <paramref name="repository"/> are present and complete,
    /// downloading what is missing, and returns the local directory.
    /// </summary>
    /// <param name="repository">For example <c>datalab-to/chandra-ocr-2</c>.</param>
    /// <param name="files">Paths inside the repository.</param>
    /// <param name="revision">Branch, tag or commit.</param>
    /// <param name="progress">Receives <c>(file, bytes so far, total bytes)</c>.</param>
    /// <param name="cancellation">Cancels the transfer.</param>
    public static async Task<string> EnsureAsync(
        string repository,
        IEnumerable<string> files,
        string revision = "main",
        Action<string, long, long>? progress = null,
        CancellationToken cancellation = default)
    {
        string directory = Path.Combine(CacheRoot, "hf", repository.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(directory);
        using var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = true })
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("indra-ocr-dotnet/0.1");
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        foreach (string file in files)
        {
            string target = Path.Combine(directory, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            string url = $"{Endpoint}/{repository}/resolve/{revision}/{file}";
            await DownloadAsync(client, url, target, file, progress, cancellation).ConfigureAwait(false);
        }

        return directory;
    }

    /// <summary>Synchronous <see cref="EnsureAsync"/>.</summary>
    public static string Ensure(string repository, IEnumerable<string> files, Action<string, long, long>? progress = null) =>
        EnsureAsync(repository, files, progress: progress).GetAwaiter().GetResult();

    /// <summary>
    /// Datalab's model store, which Surya's <c>S3DownloaderMixin</c> reads for <c>s3://</c>
    /// checkpoints: <c>DATALAB_MODELS_ENDPOINT</c>, else https://models.datalab.to.
    /// </summary>
    public static string DatalabEndpoint =>
        (Environment.GetEnvironmentVariable("DATALAB_MODELS_ENDPOINT") ?? "https://models.datalab.to").TrimEnd('/');

    /// <summary>
    /// Ensures <paramref name="files"/> of the Datalab checkpoint at <paramref name="path"/> (for
    /// example <c>text_detection/2025_05_07</c>, from <c>s3://text_detection/2025_05_07</c>) are
    /// present under <c>&lt;cache&gt;/datalab/&lt;path&gt;</c>, and returns that directory.
    /// </summary>
    public static async Task<string> EnsureDatalabAsync(
        string path,
        IEnumerable<string> files,
        Action<string, long, long>? progress = null,
        CancellationToken cancellation = default)
    {
        path = path.Trim('/');
        string directory = Path.Combine(CacheRoot, "datalab", path.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(directory);
        using var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = true })
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("indra-ocr-dotnet/0.1");
        foreach (string file in files)
        {
            string target = Path.Combine(directory, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            await DownloadAsync(client, $"{DatalabEndpoint}/{path}/{file}", target, file, progress, cancellation).ConfigureAwait(false);
        }

        return directory;
    }

    /// <summary>Synchronous <see cref="EnsureDatalabAsync"/>.</summary>
    public static string EnsureDatalab(string path, IEnumerable<string> files, Action<string, long, long>? progress = null) =>
        EnsureDatalabAsync(path, files, progress).GetAwaiter().GetResult();

    private static async Task DownloadAsync(
        HttpClient client,
        string url,
        string target,
        string name,
        Action<string, long, long>? progress,
        CancellationToken cancellation)
    {
        string complete = target + ".complete";
        if (File.Exists(target) && File.Exists(complete))
        {
            return;
        }

        string partial = target + ".partial";
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                long have = File.Exists(partial) ? new FileInfo(partial).Length : 0;
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (have > 0)
                {
                    request.Headers.Range = new RangeHeaderValue(have, null);
                }

                using HttpResponseMessage response = await client
                    .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellation).ConfigureAwait(false);
                if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
                {
                    break;
                }

                response.EnsureSuccessStatusCode();
                bool resumed = response.StatusCode == HttpStatusCode.PartialContent;
                long total = (resumed ? have : 0) + (response.Content.Headers.ContentLength ?? 0);
                await using (var output = new FileStream(partial, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write))
                await using (Stream input = await response.Content.ReadAsStreamAsync(cancellation).ConfigureAwait(false))
                {
                    var buffer = new byte[1 << 20];
                    long done = resumed ? have : 0;
                    int read;
                    while ((read = await input.ReadAsync(buffer, cancellation).ConfigureAwait(false)) > 0)
                    {
                        await output.WriteAsync(buffer.AsMemory(0, read), cancellation).ConfigureAwait(false);
                        done += read;
                        progress?.Invoke(name, done, total);
                    }

                    if (total > 0 && done < total)
                    {
                        throw new IOException($"transfer of {name} ended at {done} of {total} bytes");
                    }
                }

                break;
            }
            catch (Exception e) when (e is HttpRequestException or IOException && attempt < 8)
            {
                await Task.Delay(TimeSpan.FromSeconds(Math.Min(60, 2 << attempt)), cancellation).ConfigureAwait(false);
            }
        }

        File.Move(partial, target, overwrite: true);
        await File.WriteAllTextAsync(complete, DateTime.UtcNow.ToString("O"), cancellation).ConfigureAwait(false);
    }
}
