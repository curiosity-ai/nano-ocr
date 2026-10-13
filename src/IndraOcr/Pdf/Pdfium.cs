using System.Runtime.InteropServices;

namespace IndraOcr.Pdf;

/// <summary>
/// The PDFium entry points the pipeline uses — the same calls pypdfium2 makes, so the text layer
/// and the renders match the Python implementation. PDFium is not thread-safe: every call goes
/// through <see cref="Lock"/>.
/// </summary>
internal static partial class Pdfium
{
    private const string Lib = "pdfium";

    /// <summary>Serializes all PDFium calls in the process.</summary>
    public static readonly Lock Lock = new();

    private static bool _initialized;

    public const int FPDF_ANNOT = 0x01;
    public const int FPDF_GRAYSCALE = 0x08;
    public const int FPDFBitmap_Gray = 1;
    public const int FPDFBitmap_BGR = 2;
    public const int FLATTEN_FAIL = 0;
    public const int FLAT_NORMALDISPLAY = 0;
    public const int FPDF_PAGEOBJ_TEXT = 1;
    public const int FPDF_PAGEOBJ_FORM = 5;
    public const int FPDF_TEXTRENDERMODE_INVISIBLE = 3;
    public const uint FPDF_ERR_PASSWORD = 4;

    public static void EnsureInitialized()
    {
        lock (Lock)
        {
            if (!_initialized)
            {
                FPDF_InitLibrary();
                _initialized = true;
            }
        }
    }

    [LibraryImport(Lib)]
    public static partial void FPDF_InitLibrary();

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial IntPtr FPDF_LoadMemDocument64(IntPtr data, nuint size, string? password);

    [LibraryImport(Lib)]
    public static partial void FPDF_CloseDocument(IntPtr doc);

    [LibraryImport(Lib)]
    public static partial uint FPDF_GetLastError();

    [LibraryImport(Lib)]
    public static partial int FPDF_GetPageCount(IntPtr doc);

    [LibraryImport(Lib)]
    public static partial IntPtr FPDF_LoadPage(IntPtr doc, int index);

    [LibraryImport(Lib)]
    public static partial void FPDF_ClosePage(IntPtr page);

    [LibraryImport(Lib)]
    public static partial float FPDF_GetPageWidthF(IntPtr page);

    [LibraryImport(Lib)]
    public static partial float FPDF_GetPageHeightF(IntPtr page);

    [LibraryImport(Lib)]
    public static partial IntPtr FPDFBitmap_CreateEx(int width, int height, int format, IntPtr firstScan, int stride);

    [LibraryImport(Lib)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool FPDFBitmap_FillRect(IntPtr bitmap, int left, int top, int width, int height, uint color);

    [LibraryImport(Lib)]
    public static partial IntPtr FPDFBitmap_GetBuffer(IntPtr bitmap);

    [LibraryImport(Lib)]
    public static partial int FPDFBitmap_GetStride(IntPtr bitmap);

    [LibraryImport(Lib)]
    public static partial void FPDFBitmap_Destroy(IntPtr bitmap);

    [LibraryImport(Lib)]
    public static partial void FPDF_RenderPageBitmap(IntPtr bitmap, IntPtr page, int startX, int startY, int sizeX, int sizeY, int rotate, int flags);

    [LibraryImport(Lib)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool FPDF_PageToDevice(IntPtr page, int startX, int startY, int sizeX, int sizeY, int rotate, double pageX, double pageY, out int deviceX, out int deviceY);

    [LibraryImport(Lib)]
    public static partial int FPDFPage_Flatten(IntPtr page, int flag);

    [LibraryImport(Lib)]
    public static partial IntPtr FPDFText_LoadPage(IntPtr page);

    [LibraryImport(Lib)]
    public static partial void FPDFText_ClosePage(IntPtr textPage);

    [LibraryImport(Lib)]
    public static partial int FPDFText_CountChars(IntPtr textPage);

    [LibraryImport(Lib)]
    public static partial uint FPDFText_GetUnicode(IntPtr textPage, int index);

    [LibraryImport(Lib)]
    public static partial float FPDFText_GetCharAngle(IntPtr textPage, int index);

    [LibraryImport(Lib)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool FPDFText_GetCharBox(IntPtr textPage, int index, out double left, out double right, out double bottom, out double top);

    [LibraryImport(Lib)]
    public static partial int FPDFPage_CountObjects(IntPtr page);

    [LibraryImport(Lib)]
    public static partial IntPtr FPDFPage_GetObject(IntPtr page, int index);

    [LibraryImport(Lib)]
    public static partial int FPDFPageObj_GetType(IntPtr obj);

    [LibraryImport(Lib)]
    public static partial int FPDFFormObj_CountObjects(IntPtr form);

    [LibraryImport(Lib)]
    public static partial IntPtr FPDFFormObj_GetObject(IntPtr form, uint index);

    [LibraryImport(Lib)]
    public static partial int FPDFTextObj_GetTextRenderMode(IntPtr text);
}
