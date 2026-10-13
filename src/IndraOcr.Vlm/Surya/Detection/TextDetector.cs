using System.Diagnostics;
using IndraOcr.Vlm.Common;
using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya.Detection;

/// <summary>
/// Surya's text-line detector — <c>DetectionPredictor</c> (<c>surya/detection/__init__.py</c>)
/// run in-process: split tall pages into chunks, resize and normalise each, run the EfficientViT
/// segmentation model, upsample its maps to the processor size, stitch the chunks back together,
/// and turn the text heatmap into line polygons.
/// </summary>
public sealed class TextDetector
{
    /// <summary>Where Datalab publishes the checkpoint (<c>DETECTOR_MODEL_CHECKPOINT</c>).</summary>
    public const string CheckpointPath = "text_detection/2025_05_07";

    /// <summary>The files <see cref="Download"/> fetches.</summary>
    public static readonly string[] CheckpointFiles = ["config.json", "preprocessor_config.json", "model.safetensors"];

    private readonly EfficientVitModel _model;

    private TextDetector(EfficientVitModel model)
    {
        _model = model;
        Config = model.Config;
        Preprocessor = new DetectionPreprocessor(Config);
    }

    /// <summary>The checkpoint's configuration.</summary>
    public DetectionConfig Config { get; }

    /// <summary>The input side (chunking, resizes, normalisation).</summary>
    public DetectionPreprocessor Preprocessor { get; }

    /// <summary>The segmentation model.</summary>
    public EfficientVitModel Model => _model;

    /// <summary>Receives each chunk's model output (chunk index, maps); the parity tests use it.</summary>
    internal Action<int, SegmentationOutput>? ChunkObserver { get; set; }

    /// <summary>Wall-clock milliseconds of the last <see cref="Detect(RgbImage, bool)"/> by stage.</summary>
    public Dictionary<string, double> Timings { get; } = new(StringComparer.Ordinal);

    /// <summary>Loads the detector from a checkpoint directory.</summary>
    public static TextDetector Load(string directory) => new(EfficientVitModel.Load(directory));

    /// <summary>
    /// Downloads the checkpoint from <c>models.datalab.to</c> into the cache (once) and loads it.
    /// </summary>
    public static TextDetector Download(Action<string, long, long>? progress = null) =>
        Load(ModelDownloader.EnsureDatalab(CheckpointPath, CheckpointFiles, progress));

    /// <summary>Detects the text lines of one page.</summary>
    /// <param name="image">The page.</param>
    /// <param name="includeMaps">Whether to return the stitched heatmap and affinity map.</param>
    public TextDetectionResult Detect(RgbImage image, bool includeMaps = false)
    {
        Timings.Clear();
        var clock = Stopwatch.StartNew();
        (float[] heatmap, float[] affinity, int rows) = Heatmaps(image);
        Timings["model"] = clock.Elapsed.TotalMilliseconds;
        clock.Restart();
        List<PolygonBox> boxes = HeatmapBoxes.GetBoxes(heatmap, Config.ImageWidth, rows, image.Width, image.Height);
        Timings["boxes"] = clock.Elapsed.TotalMilliseconds;
        return new TextDetectionResult
        {
            Bboxes = boxes,
            Heatmap = includeMaps ? heatmap : null,
            AffinityMap = includeMaps ? affinity : null,
            HeatmapHeight = rows,
            HeatmapWidth = Config.ImageWidth,
            ImageBbox = [0, 0, image.Width, image.Height],
        };
    }

    /// <summary>Detects the text lines of each page.</summary>
    public List<TextDetectionResult> Detect(IEnumerable<RgbImage> images, bool includeMaps = false) =>
        images.Select(image => Detect(image, includeMaps)).ToList();

    /// <summary>
    /// <c>batch_detection</c> for one page: the stitched text heatmap and affinity map,
    /// <c>[rows, processor_width]</c>, where <c>rows</c> is the processor height for a page of one
    /// chunk, and otherwise <c>processor_height</c> per full chunk plus the last chunk's height in
    /// page pixels — Surya cuts the last chunk's map at its page height, not at its share of the
    /// processor's rows.
    /// </summary>
    public (float[] Heatmap, float[] Affinity, int Rows) Heatmaps(RgbImage image)
    {
        int width = Config.ImageWidth;
        int height = Config.ImageHeight;
        List<ImageChunk> chunks = Preprocessor.Split(image);
        try
        {
            int rows = 0;
            for (int i = 0; i < chunks.Count; i++)
            {
                rows += i > 0 && chunks[i].Height < height ? chunks[i].Height : height;
            }

            var heatmap = new float[rows * width];
            var affinity = new float[rows * width];
            int offset = 0;
            for (int i = 0; i < chunks.Count; i++)
            {
                float[][] maps = ChunkMaps(chunks[i].Image, i);
                int keep = i > 0 && chunks[i].Height < height ? chunks[i].Height : height;
                maps[0].AsSpan(0, keep * width).CopyTo(heatmap.AsSpan(offset * width));
                if (maps.Length > 1)
                {
                    maps[1].AsSpan(0, keep * width).CopyTo(affinity.AsSpan(offset * width));
                }

                offset += keep;
            }

            return (heatmap, affinity, rows);
        }
        finally
        {
            foreach (ImageChunk chunk in chunks)
            {
                chunk.Image.Dispose();
            }
        }
    }

    /// <summary>
    /// One chunk through <c>prepare_image</c>, the model and the upsample to the processor size:
    /// one <c>[height · width]</c> map per label.
    /// </summary>
    public float[][] ChunkMaps(RgbImage chunk) => ChunkMaps(chunk, 0);

    private float[][] ChunkMaps(RgbImage chunk, int index)
    {
        int width = Config.ImageWidth;
        int height = Config.ImageHeight;
        using RgbImage resized = Preprocessor.Resize(chunk);
        float[] pixels = Preprocessor.Normalize(resized);
        SegmentationOutput output = _model.Forward(pixels, height, width);
        ChunkObserver?.Invoke(index, output);
        return Upsample(output, height, width);
    }

    /// <summary>
    /// <c>F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)</c> of
    /// the model's quarter-resolution maps (skipped, as in Surya, when they already have that size).
    /// </summary>
    public static float[][] Upsample(SegmentationOutput output, int height, int width)
    {
        int plane = output.Height * output.Width;
        var maps = new float[output.Labels][];
        for (int c = 0; c < output.Labels; c++)
        {
            ReadOnlySpan<float> source = output.Maps.AsSpan(c * plane, plane);
            if (output.Height == height && output.Width == width)
            {
                maps[c] = source.ToArray();
                continue;
            }

            maps[c] = new float[height * width];
            ConvKernels.ResizePlane(output.Maps, c * plane, output.Height, output.Width, maps[c], height, width);
        }

        return maps;
    }
}
