using System.Text.Json.Serialization;

namespace IndraOcr.Vlm.Surya;

/// <summary><c>surya.common.polygon.PolygonBox</c>: four corners, clockwise from top-left.</summary>
public class PolygonBox
{
    /// <summary>The corners <c>[[x, y], ...]</c>.</summary>
    [JsonPropertyName("polygon")]
    [JsonPropertyOrder(-10)]
    public required double[][] Polygon { get; set; }

    /// <summary>Confidence, when the producer has one.</summary>
    [JsonPropertyName("confidence")]
    [JsonPropertyOrder(-9)]
    public double? Confidence { get; set; }

    /// <summary><c>[x_min, y_min, x_max, y_max]</c> of the polygon.</summary>
    [JsonPropertyName("bbox")]
    [JsonPropertyOrder(100)]
    public double[] Bbox =>
    [
        Polygon.Min(p => p[0]), Polygon.Min(p => p[1]), Polygon.Max(p => p[0]), Polygon.Max(p => p[1]),
    ];

    /// <summary>The polygon of an axis-aligned box.</summary>
    public static double[][] FromBox(double x0, double y0, double x1, double y1) =>
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

/// <summary><c>surya.layout.schema.LayoutBox</c>.</summary>
public sealed class LayoutBox : PolygonBox
{
    /// <summary>Canonical label (<see cref="SuryaPrompts.Relabel"/>).</summary>
    [JsonPropertyName("label")]
    public required string Label { get; init; }

    /// <summary>The label the model emitted.</summary>
    [JsonPropertyName("raw_label")]
    public required string RawLabel { get; init; }

    /// <summary>Reading-order index.</summary>
    [JsonPropertyName("position")]
    public int Position { get; set; }

    /// <summary>The model's estimate of the block's OCR token count.</summary>
    [JsonPropertyName("count")]
    public int Count { get; init; }
}

/// <summary><c>surya.layout.schema.LayoutResult</c>.</summary>
public sealed class LayoutResult
{
    /// <summary>The blocks, in reading order.</summary>
    [JsonPropertyName("bboxes")]
    public required List<LayoutBox> Bboxes { get; init; }

    /// <summary><c>[0, 0, width, height]</c>.</summary>
    [JsonPropertyName("image_bbox")]
    public required double[] ImageBbox { get; init; }

    /// <summary>The model's raw output.</summary>
    [JsonPropertyName("raw")]
    public string? Raw { get; init; }

    /// <summary>Whether generation or parsing failed.</summary>
    [JsonPropertyName("error")]
    public bool Error { get; init; }
}

/// <summary><c>surya.recognition.schema.BlockOCRResult</c>.</summary>
public sealed class BlockOcrResult : PolygonBox
{
    /// <summary>Canonical label.</summary>
    [JsonPropertyName("label")]
    public required string Label { get; init; }

    /// <summary>The label the model emitted.</summary>
    [JsonPropertyName("raw_label")]
    public string RawLabel { get; init; } = string.Empty;

    /// <summary>0-based position in the layout.</summary>
    [JsonPropertyName("reading_order")]
    public int ReadingOrder { get; init; }

    /// <summary>The block's HTML (empty when skipped).</summary>
    [JsonPropertyName("html")]
    public string Html { get; init; } = string.Empty;

    /// <summary>Whether the label is one that is not OCR'd.</summary>
    [JsonPropertyName("skipped")]
    public bool Skipped { get; init; }

    /// <summary>Whether generation failed.</summary>
    [JsonPropertyName("error")]
    public bool Error { get; init; }
}

/// <summary><c>surya.recognition.schema.PageOCRResult</c>.</summary>
public sealed class PageOcrResult
{
    /// <summary>The blocks.</summary>
    [JsonPropertyName("blocks")]
    public required List<BlockOcrResult> Blocks { get; init; }

    /// <summary><c>[0, 0, width, height]</c>.</summary>
    [JsonPropertyName("image_bbox")]
    public required double[] ImageBbox { get; init; }

    /// <summary>1-based page number within its document (the CLI's <c>page</c>).</summary>
    [JsonPropertyName("page")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int Page { get; set; }
}

/// <summary><c>surya.table_rec.schema.TableRow</c>.</summary>
public sealed class TableRow : PolygonBox
{
    /// <summary>Row index.</summary>
    [JsonPropertyName("row_id")]
    public int RowId { get; init; }
}

/// <summary><c>surya.table_rec.schema.TableCol</c>.</summary>
public sealed class TableCol : PolygonBox
{
    /// <summary>Column index.</summary>
    [JsonPropertyName("col_id")]
    public int ColId { get; init; }
}

/// <summary><c>surya.table_rec.schema.TableCell</c>: a row × column intersection.</summary>
public sealed class TableCell : PolygonBox
{
    /// <summary>Row index.</summary>
    [JsonPropertyName("row_id")]
    public int RowId { get; init; }

    /// <summary>Column index.</summary>
    [JsonPropertyName("col_id")]
    public int ColId { get; init; }

    /// <summary>Cell index.</summary>
    [JsonPropertyName("cell_id")]
    public int CellId { get; init; }
}

/// <summary><c>surya.table_rec.schema.TableResult</c>.</summary>
public sealed class TableResult
{
    /// <summary>Rows (simple mode).</summary>
    [JsonPropertyName("rows")]
    public required List<TableRow> Rows { get; init; }

    /// <summary>Columns (simple mode).</summary>
    [JsonPropertyName("cols")]
    public required List<TableCol> Cols { get; init; }

    /// <summary>Cells derived geometrically (simple mode).</summary>
    [JsonPropertyName("cells")]
    public required List<TableCell> Cells { get; init; }

    /// <summary><c>[0, 0, width, height]</c>.</summary>
    [JsonPropertyName("image_bbox")]
    public required double[] ImageBbox { get; init; }

    /// <summary>The model's raw output.</summary>
    [JsonPropertyName("raw")]
    public string? Raw { get; init; }

    /// <summary>The table HTML (full mode).</summary>
    [JsonPropertyName("html")]
    public string? Html { get; init; }

    /// <summary><c>"simple"</c> or <c>"full"</c>.</summary>
    [JsonPropertyName("mode")]
    public string Mode { get; init; } = "simple";

    /// <summary>Whether generation or parsing failed.</summary>
    [JsonPropertyName("error")]
    public bool Error { get; init; }
}
