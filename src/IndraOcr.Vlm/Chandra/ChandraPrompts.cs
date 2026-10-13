namespace IndraOcr.Vlm.Chandra;

/// <summary><c>chandra/prompts.py</c>, character for character — the model's training-time contract.</summary>
public static class ChandraPrompts
{
    /// <summary><c>ALLOWED_TAGS</c>.</summary>
    public static readonly string[] AllowedTags =
    [
        "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th", "div", "pre", "h1", "h2", "h3",
        "h4", "h5", "ul", "ol", "li", "input", "a", "span", "img", "hr", "tbody", "small", "caption", "strong", "thead",
        "big", "code", "chem",
    ];

    /// <summary><c>ALLOWED_ATTRIBUTES</c>.</summary>
    public static readonly string[] AllowedAttributes =
    [
        "class", "colspan", "rowspan", "display", "checked", "type", "border", "value", "style", "href", "alt", "align",
        "data-bbox", "data-label",
    ];

    /// <summary><c>PROMPT_ENDING</c> (the lists are formatted as Python's <c>repr</c> of a list).</summary>
    public static readonly string PromptEnding = $"""
        Only use these tags {PyList(AllowedTags)}, and these attributes {PyList(AllowedAttributes)}.

        Guidelines:
        * Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
        * Tables: Use colspan and rowspan attributes to match table structure.
        * Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
        * Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
        * Forms: Mark checkboxes and radio buttons properly.
        * Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
        * Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
        * Lists: Preserve indents and proper list markers.
        * Use the simplest possible HTML structure that accurately represents the content of the block.
        * Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
        """;

    /// <summary><c>OCR_LAYOUT_PROMPT</c>: OCR to HTML arranged as labelled, boxed layout blocks.</summary>
    public static readonly string OcrLayout = $"""
        OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

        Use the following labels:
        - Caption
        - Footnote
        - Equation-Block
        - List-Group
        - Page-Header
        - Page-Footer
        - Image
        - Section-Header
        - Table
        - Text
        - Complex-Block
        - Code-Block
        - Form
        - Table-Of-Contents
        - Figure
        - Chemical-Block
        - Diagram
        - Bibliography
        - Blank-Page

        {PromptEnding}
        """;

    /// <summary><c>OCR_PROMPT</c>: OCR to HTML without layout.</summary>
    public static readonly string Ocr = $"""
        OCR this image to HTML.

        {PromptEnding}
        """;

    /// <summary><c>PROMPT_MAPPING</c>.</summary>
    public static string ForType(string promptType) => promptType switch
    {
        "ocr_layout" => OcrLayout,
        "ocr" => Ocr,
        _ => throw new ArgumentException($"unknown prompt type '{promptType}'", nameof(promptType)),
    };

    private static string PyList(string[] items) => "[" + string.Join(", ", items.Select(i => $"'{i}'")) + "]";
}
