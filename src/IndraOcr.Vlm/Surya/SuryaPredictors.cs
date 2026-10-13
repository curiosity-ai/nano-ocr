using IndraOcr.Pdf;
using IndraOcr.Vlm.Chandra;
using IndraOcr.Vlm.Common;
using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya;

/// <summary>Image helpers with PIL's semantics.</summary>
internal static class PilCrop
{
    /// <summary><c>Image.crop(box)</c>: coordinates rounded half-to-even; out-of-range pixels are black.</summary>
    public static RgbImage Crop(RgbImage image, double x0, double y0, double x1, double y1) =>
        ChandraOutput.Crop(
            image,
            (int)Math.Round(x0),
            (int)Math.Round(y0),
            (int)Math.Round(x1),
            (int)Math.Round(y1));
}

/// <summary>
/// <c>surya/layout/__init__.py</c>: the layout prompt on whole pages, parsed into canonical,
/// blank-filtered blocks in pixel coordinates.
/// </summary>
public sealed class SuryaLayoutPredictor(SuryaInference inference)
{
    /// <summary><c>SURYA_MAX_TOKENS_LAYOUT</c>.</summary>
    public int MaxTokens { get; init; } = 3072;

    /// <summary><c>SURYA_GUIDED_LAYOUT</c>: constrain the output to the layout JSON schema.</summary>
    public bool Guided { get; init; } = true;

    /// <summary>
    /// Runs layout on pages. <paramref name="targetSizes"/> denormalises the boxes to other
    /// sizes (layout runs on a low-DPI render, OCR on a high-DPI one).
    /// </summary>
    public List<LayoutResult> Predict(IReadOnlyList<RgbImage> images, IReadOnlyList<(int Width, int Height)>? targetSizes = null)
    {
        if (targetSizes is not null && targetSizes.Count != images.Count)
        {
            throw new ArgumentException("target_image_sizes must match images length");
        }

        var results = new List<LayoutResult>();
        for (int index = 0; index < images.Count; index++)
        {
            RgbImage image = images[index];
            SuryaResponse output = inference.Generate(new SuryaRequest
            {
                Image = image,
                Prompt = SuryaPrompts.Layout,
                MaxTokens = MaxTokens,
                Constraint = Guided ? JsonArrayConstraint.SuryaLayout : null,
            });
            (int w, int h) = targetSizes?[index] ?? (image.Width, image.Height);
            results.Add(Parse(image, output, w, h));
        }

        return results;
    }

    /// <summary>The parsing half of <c>LayoutPredictor.__call__</c>.</summary>
    public static LayoutResult Parse(RgbImage image, SuryaResponse output, int w, int h)
    {
        double[] pageBox = [0, 0, w, h];
        if (output.Error || output.Raw.Length == 0)
        {
            return new LayoutResult { Bboxes = [], ImageBbox = pageBox, Raw = output.Raw, Error = true };
        }

        List<ParsedLayoutBlock> parsed;
        try
        {
            parsed = SuryaParsers.ParseLayout(output.Raw);
        }
        catch (Exception e) when (e is FormatException or System.Text.Json.JsonException)
        {
            return new LayoutResult { Bboxes = [], ImageBbox = pageBox, Raw = output.Raw, Error = true };
        }

        double confidence = output.MeanTokenProb ?? 1.0;
        var boxes = new List<LayoutBox>();
        foreach (ParsedLayoutBlock block in parsed)
        {
            string canonical = SuryaPrompts.Canonical(block.Label);
            if (SuryaPrompts.TextLabels.Contains(canonical))
            {
                double[] imageBox = SuryaParsers.DenormBbox(block.Bbox, image.Width, image.Height);
                int x0 = Math.Max(0, (int)imageBox[0]), y0 = Math.Max(0, (int)imageBox[1]);
                int x1 = Math.Max(0, (int)imageBox[2]), y1 = Math.Max(0, (int)imageBox[3]);
                if (x1 > x0 && y1 > y0)
                {
                    using RgbImage crop = ChandraOutput.Crop(image, x0, y0, x1, y1);
                    if (BlankRegion.IsBlank(crop))
                    {
                        continue;
                    }
                }
            }

            double[] pixel = SuryaParsers.DenormBbox(block.Bbox, w, h);
            boxes.Add(new LayoutBox
            {
                Polygon = PolygonBox.FromBox(pixel[0], pixel[1], pixel[2], pixel[3]),
                Label = canonical,
                RawLabel = block.Label,
                Position = boxes.Count,
                Count = block.Count,
                Confidence = confidence,
            });
        }

        return new LayoutResult { Bboxes = boxes, ImageBbox = pageBox, Raw = output.Raw, Error = false };
    }
}

/// <summary>
/// <c>surya/recognition/__init__.py</c>: full-page OCR (one request per page, falling back to
/// layout plus per-block OCR when the output loops or fails to parse), or block mode.
/// </summary>
public sealed class SuryaRecognitionPredictor(SuryaInference inference)
{
    /// <summary><c>SURYA_MAX_TOKENS_FULL_PAGE</c>.</summary>
    public int MaxTokensFullPage { get; init; } = 12288;

    /// <summary><c>SURYA_MAX_TOKENS_BLOCK_CEILING</c>.</summary>
    public int MaxTokensBlockCeiling { get; init; } = 8192;

    /// <summary><c>SURYA_FULLPAGE_REGEN</c>: re-decode a failed full page at rising temperature first.</summary>
    public bool FullPageRegen { get; init; }

    /// <summary>
    /// <c>RecognitionPredictor.__call__</c>: full-page mode unless <paramref name="layouts"/> are
    /// given (or <paramref name="fullPage"/> says otherwise).
    /// </summary>
    public List<PageOcrResult> Predict(IReadOnlyList<RgbImage> images, IReadOnlyList<LayoutResult>? layouts = null, bool? fullPage = null)
    {
        if (images.Count == 0)
        {
            return [];
        }

        fullPage ??= layouts is null;
        if (fullPage.Value)
        {
            return FullPage(images, layouts);
        }

        if (layouts is null)
        {
            throw new ArgumentException("layout_results required when full_page=False");
        }

        if (layouts.Count != images.Count)
        {
            throw new ArgumentException($"images and layout_results must be same length ({images.Count} vs {layouts.Count})");
        }

        var results = new List<PageOcrResult>();
        for (int page = 0; page < images.Count; page++)
        {
            RgbImage image = images[page];
            var blocks = new List<BlockOcrResult>();
            foreach (LayoutBox box in layouts[page].Bboxes)
            {
                if (SuryaPrompts.SkipCanonicalLabels.Contains(box.Label))
                {
                    blocks.Add(new BlockOcrResult
                    {
                        Polygon = box.Polygon, Label = box.Label, RawLabel = box.RawLabel, ReadingOrder = box.Position,
                        Html = string.Empty, Skipped = true, Confidence = 1.0,
                    });
                    continue;
                }

                using RgbImage crop = CropBlock(image, box.Polygon);
                SuryaResponse output = inference.Generate(new SuryaRequest
                {
                    Image = crop,
                    Prompt = SuryaPrompts.Block,
                    MaxTokens = SuryaPrompts.ImageTokenBudget(box.Count, MaxTokensBlockCeiling),
                });

                blocks.Add(output.Error
                    ? new BlockOcrResult
                    {
                        Polygon = box.Polygon, Label = box.Label, RawLabel = box.RawLabel, ReadingOrder = box.Position,
                        Html = string.Empty, Error = true, Confidence = 0.0,
                    }
                    : new BlockOcrResult
                    {
                        Polygon = box.Polygon, Label = box.Label, RawLabel = box.RawLabel, ReadingOrder = box.Position,
                        Html = SuryaParsers.CleanBlockHtml(output.Raw), Confidence = output.MeanTokenProb ?? 1.0,
                    });
            }

            results.Add(new PageOcrResult { Blocks = blocks, ImageBbox = [0, 0, image.Width, image.Height] });
        }

        return results;
    }

    /// <summary><c>_crop_block</c>: the polygon's box padded by 4 pixels, clamped to the image.</summary>
    public static RgbImage CropBlock(RgbImage image, double[][] polygon, int pad = 4)
    {
        int x0 = Math.Max(0, (int)polygon.Min(p => p[0]) - pad);
        int y0 = Math.Max(0, (int)polygon.Min(p => p[1]) - pad);
        int x1 = Math.Min(image.Width, (int)polygon.Max(p => p[0]) + pad);
        int y1 = Math.Min(image.Height, (int)polygon.Max(p => p[1]) + pad);
        return x1 <= x0 || y1 <= y0
            ? ChandraOutput.Crop(image, 0, 0, 1, 1)
            : ChandraOutput.Crop(image, x0, y0, x1, y1);
    }

    private List<PageOcrResult> FullPage(IReadOnlyList<RgbImage> images, IReadOnlyList<LayoutResult>? fallbackLayouts)
    {
        var results = new PageOcrResult?[images.Count];
        (float Temperature, float? TopP)[] rounds = FullPageRegen
            ? [(0f, null), .. Enumerable.Range(0, 6).Select(n => (Math.Min(0.2f * (n + 1), 0.8f), (float?)0.95f))]
            : [(0f, null)];

        List<int> pending = Enumerable.Range(0, images.Count).ToList();
        for (int round = 0; round < rounds.Length && pending.Count > 0; round++)
        {
            var still = new List<int>();
            foreach (int i in pending)
            {
                SuryaResponse output = inference.Generate(new SuryaRequest
                {
                    Image = images[i],
                    Prompt = SuryaPrompts.HighAccuracyBbox,
                    MaxTokens = MaxTokensFullPage,
                    Temperature = round > 0 ? rounds[round].Temperature : null,
                    TopP = rounds[round].TopP,
                });
                PageOcrResult? page = BuildPage(output, images[i]);
                if (page is not null)
                {
                    results[i] = page;
                }
                else
                {
                    still.Add(i);
                }
            }

            pending = still;
        }

        if (pending.Count > 0)
        {
            List<RgbImage> fallbackImages = pending.Select(i => images[i]).ToList();
            List<LayoutResult> fallback = fallbackLayouts is not null
                ? pending.Select(i => fallbackLayouts[i]).ToList()
                : new SuryaLayoutPredictor(inference).Predict(fallbackImages);
            List<PageOcrResult> blockResults = Predict(fallbackImages, fallback, fullPage: false);
            for (int k = 0; k < pending.Count; k++)
            {
                results[pending[k]] = blockResults[k];
            }
        }

        return results.Select((r, i) => r ?? new PageOcrResult { Blocks = [], ImageBbox = [0, 0, images[i].Width, images[i].Height] }).ToList();
    }

    /// <summary><c>_build_page</c>: a full-page output to blocks, or null when it should be retried.</summary>
    public static PageOcrResult? BuildPage(SuryaResponse? output, RgbImage image)
    {
        int w = image.Width, h = image.Height;
        double[] pageBox = [0, 0, w, h];
        if (output is null || output.Error)
        {
            return null;
        }

        if (output.Raw.Length == 0)
        {
            return BlankRegion.IsBlank(image) ? new PageOcrResult { Blocks = [], ImageBbox = pageBox } : null;
        }

        if (RepeatDetector.Detect(output.Raw))
        {
            return null;
        }

        List<ParsedFullPageBlock> parsed;
        try
        {
            parsed = SuryaParsers.ParseFullPageHtml(output.Raw);
        }
        catch (Exception)
        {
            return null;
        }

        double confidence = output.MeanTokenProb ?? 1.0;
        var blocks = new List<BlockOcrResult>();
        for (int index = 0; index < parsed.Count; index++)
        {
            ParsedFullPageBlock item = parsed[index];
            double x0 = item.Bbox[0] / 1000 * w, y0 = item.Bbox[1] / 1000 * h;
            double x1 = item.Bbox[2] / 1000 * w, y1 = item.Bbox[3] / 1000 * h;
            string canonical = SuryaPrompts.Canonical(item.Label);
            bool skipped = SuryaPrompts.SkipCanonicalLabels.Contains(canonical);
            blocks.Add(new BlockOcrResult
            {
                Polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                Label = canonical,
                RawLabel = item.Label,
                ReadingOrder = index,
                Html = skipped ? string.Empty : item.Html,
                Skipped = skipped,
                Confidence = confidence,
            });
        }

        // _drop_blank_text_blocks
        var kept = new List<BlockOcrResult>();
        foreach (BlockOcrResult block in blocks)
        {
            if (!SuryaPrompts.TextLabels.Contains(block.Label) || block.Skipped || block.Error)
            {
                kept.Add(block);
                continue;
            }

            using RgbImage crop = CropBlock(image, block.Polygon);
            if (!BlankRegion.IsBlank(crop))
            {
                kept.Add(block);
            }
        }

        return new PageOcrResult { Blocks = kept, ImageBbox = pageBox };
    }
}

/// <summary>
/// <c>surya/table_rec/__init__.py</c>: rows and columns (cells by intersection), or the table's
/// full HTML.
/// </summary>
public sealed class SuryaTableRecPredictor(SuryaInference inference)
{
    /// <summary><c>SURYA_MAX_TOKENS_TABLE_REC</c>.</summary>
    public int MaxTokens { get; init; } = 3072;

    /// <summary><c>SURYA_GUIDED_TABLE_REC</c>.</summary>
    public bool Guided { get; init; }

    /// <summary><c>SURYA_MAX_TOKENS_BLOCK_CEILING</c>.</summary>
    public int MaxTokensBlockCeiling { get; init; } = 8192;

    /// <summary><c>__call__(images, mode)</c>.</summary>
    public List<TableResult> Predict(IReadOnlyList<RgbImage> images, string mode = "simple", IReadOnlyList<int>? counts = null) =>
        mode == "full" ? PredictFull(images, counts) : PredictSimple(images);

    /// <summary><c>predict_simple</c>.</summary>
    public List<TableResult> PredictSimple(IReadOnlyList<RgbImage> images)
    {
        var results = new List<TableResult>();
        foreach (RgbImage image in images)
        {
            SuryaResponse output = inference.Generate(new SuryaRequest
            {
                Image = image,
                Prompt = SuryaPrompts.TableRec,
                MaxTokens = MaxTokens,
                Constraint = Guided ? JsonArrayConstraint.SuryaTableRec : null,
            });
            results.Add(ParseSimple(output, image.Width, image.Height));
        }

        return results;
    }

    /// <summary>The parsing half of <c>predict_simple</c>.</summary>
    public static TableResult ParseSimple(SuryaResponse output, int w, int h)
    {
        double[] pageBox = [0, 0, w, h];
        TableResult Failed() => new() { Rows = [], Cols = [], Cells = [], ImageBbox = pageBox, Raw = output.Raw, Mode = "simple", Error = true };
        if (output.Error || output.Raw.Length == 0)
        {
            return Failed();
        }

        List<ParsedTableElement> elements;
        try
        {
            elements = SuryaParsers.ParseTableRec(output.Raw);
        }
        catch (Exception e) when (e is FormatException or System.Text.Json.JsonException)
        {
            return Failed();
        }

        var rows = new List<TableRow>();
        var cols = new List<TableCol>();
        foreach (ParsedTableElement element in elements)
        {
            double[] b = SuryaParsers.DenormBbox(element.Bbox, w, h);
            double[][] polygon = PolygonBox.FromBox(b[0], b[1], b[2], b[3]);
            if (element.Label == "Row")
            {
                rows.Add(new TableRow { Polygon = polygon, RowId = rows.Count });
            }
            else
            {
                cols.Add(new TableCol { Polygon = polygon, ColId = cols.Count });
            }
        }

        var cells = new List<TableCell>();
        foreach (TableRow row in rows)
        {
            foreach (TableCol col in cols)
            {
                double[] r = row.Bbox, c = col.Bbox;
                double x0 = Math.Max(r[0], c[0]), y0 = Math.Max(r[1], c[1]);
                double x1 = Math.Min(r[2], c[2]), y1 = Math.Min(r[3], c[3]);
                if (x1 <= x0 || y1 <= y0)
                {
                    continue;
                }

                cells.Add(new TableCell { Polygon = PolygonBox.FromBox(x0, y0, x1, y1), RowId = row.RowId, ColId = col.ColId, CellId = cells.Count });
            }
        }

        return new TableResult { Rows = rows, Cols = cols, Cells = cells, ImageBbox = pageBox, Raw = output.Raw, Mode = "simple" };
    }

    /// <summary><c>predict_full</c>: the block prompt on the table crop.</summary>
    public List<TableResult> PredictFull(IReadOnlyList<RgbImage> images, IReadOnlyList<int>? counts = null)
    {
        var results = new List<TableResult>();
        for (int i = 0; i < images.Count; i++)
        {
            RgbImage image = images[i];
            int count = counts?[i] ?? 0;
            SuryaResponse output = inference.Generate(new SuryaRequest
            {
                Image = image,
                Prompt = SuryaPrompts.Block,
                MaxTokens = SuryaPrompts.ImageTokenBudget(count, MaxTokensBlockCeiling, floor: 1024),
            });
            double[] pageBox = [0, 0, image.Width, image.Height];
            results.Add(output.Error
                ? new TableResult { Rows = [], Cols = [], Cells = [], ImageBbox = pageBox, Raw = output.Raw, Mode = "full", Error = true }
                : new TableResult { Rows = [], Cols = [], Cells = [], ImageBbox = pageBox, Raw = output.Raw, Html = SuryaParsers.CleanBlockHtml(output.Raw), Mode = "full" });
        }

        return results;
    }
}

/// <summary><c>surya/input/load.py</c> and <c>surya/input/processing.py</c>.</summary>
public static class SuryaInput
{
    /// <summary><c>IMAGE_DPI</c>: layout and detection.</summary>
    public const int ImageDpi = 96;

    /// <summary><c>IMAGE_DPI_HIGHRES</c>: recognition and tables.</summary>
    public const int ImageDpiHighRes = 192;

    /// <summary><c>get_page_images</c>: pages rendered at <paramref name="dpi"/>, annotations not drawn.</summary>
    public static List<RgbImage> LoadPdf(string path, IReadOnlyList<int>? pages = null, int dpi = ImageDpi)
    {
        using PdfDocument document = PdfDocument.Open(path);
        if (pages is { Count: > 0 } && pages.Any(p => p < 0 || p >= document.PageCount))
        {
            throw new ArgumentException($"Invalid page range: [{string.Join(", ", pages)}]");
        }

        var images = new List<RgbImage>();
        foreach (int index in pages is { Count: > 0 } ? pages : Enumerable.Range(0, document.PageCount))
        {
            using PdfPage page = document.GetPage(index);
            (byte[] rgb, int width, int height) = page.RenderRgb(dpi / 72.0, drawAnnotations: false);
            images.Add(RgbImage.Wrap(rgb, width, height));
        }

        return images;
    }

    /// <summary><c>load_from_file</c>: a PDF's pages or one image, with the name stem.</summary>
    public static (List<RgbImage> Images, List<string> Names) LoadFile(string path, IReadOnlyList<int>? pages = null, int dpi = ImageDpi)
    {
        string name = Path.GetFileName(path).Split('.')[0];
        if (PdfDocument.IsPdf(path))
        {
            List<RgbImage> images = LoadPdf(path, pages, dpi);
            return (images, Enumerable.Repeat(name, images.Count).ToList());
        }

        return ([RgbImageIO.Read(path)], [name]);
    }

    /// <summary><c>load_from_folder</c>: every non-hidden file; unreadable images are skipped.</summary>
    public static (List<RgbImage> Images, List<string> Names) LoadFolder(string folder, IReadOnlyList<int>? pages = null, int dpi = ImageDpi)
    {
        var images = new List<RgbImage>();
        var names = new List<string>();
        foreach (string path in Directory.EnumerateFiles(folder).Where(p => !Path.GetFileName(p).StartsWith('.')))
        {
            try
            {
                (List<RgbImage> i, List<string> n) = LoadFile(path, pages, dpi);
                images.AddRange(i);
                names.AddRange(n);
            }
            catch (InvalidDataException)
            {
            }
        }

        return (images, names);
    }

    /// <summary>The CLI's <c>parse_range_str</c>.</summary>
    public static List<int> ParseRange(string range) => ChandraOcr.ParseRange(range);

    /// <summary><c>expand_bbox</c> (table crops).</summary>
    public static double[] ExpandBbox(double[] bbox, double factor = 0.01) =>
        [bbox[0] * (1 - factor), bbox[1] * (1 - factor), bbox[2] * (1 + factor), bbox[3] * (1 + factor)];

    /// <summary>A crop with PIL's float-box rounding.</summary>
    public static RgbImage Crop(RgbImage image, double[] box) => PilCrop.Crop(image, box[0], box[1], box[2], box[3]);
}
