using System.Runtime.InteropServices;
using IndraOcr.Imaging;

namespace IndraOcr.Pdf;

/// <summary>The PDF cannot be opened: corrupt, not a PDF, or locked by a password.</summary>
public sealed class PdfException(string message, bool needsPassword = false) : IOException(message)
{
    /// <summary>Whether the failure was a missing or wrong password.</summary>
    public bool NeedsPassword { get; } = needsPassword;
}

/// <summary>An open PDF document (PDFium). Dispose to release it.</summary>
public sealed class PdfDocument : IDisposable
{
    private readonly GCHandle _pin;
    private IntPtr _doc;

    private PdfDocument(byte[] bytes, string? password)
    {
        Pdfium.EnsureInitialized();
        _pin = GCHandle.Alloc(bytes, GCHandleType.Pinned);
        lock (Pdfium.Lock)
        {
            _doc = Pdfium.FPDF_LoadMemDocument64(_pin.AddrOfPinnedObject(), (nuint)bytes.Length, password);
            if (_doc == IntPtr.Zero)
            {
                uint err = Pdfium.FPDF_GetLastError();
                _pin.Free();
                throw err == Pdfium.FPDF_ERR_PASSWORD
                    ? new PdfException("password-protected PDF (pass a password)", needsPassword: true)
                    : new PdfException($"cannot open the PDF (pdfium error {err})");
            }

            PageCount = Pdfium.FPDF_GetPageCount(_doc);
        }
    }

    /// <summary>Number of pages.</summary>
    public int PageCount { get; }

    /// <summary>Opens a PDF from bytes.</summary>
    public static PdfDocument Open(byte[] bytes, string? password = null) => new(bytes, password);

    /// <summary>Opens a PDF file.</summary>
    public static PdfDocument Open(string path, string? password = null) => new(File.ReadAllBytes(path), password);

    /// <summary>Whether the bytes say PDF (the header may follow a little junk, as Python's check allows).</summary>
    public static bool IsPdf(ReadOnlySpan<byte> head)
    {
        ReadOnlySpan<byte> marker = "%PDF-"u8;
        return head[..Math.Min(head.Length, 1024)].IndexOf(marker) >= 0;
    }

    /// <summary>Whether the file at <paramref name="path"/> is a PDF.</summary>
    public static bool IsPdf(string path)
    {
        using FileStream f = File.OpenRead(path);
        var buf = new byte[1024];
        int n = f.ReadAtLeast(buf, 1024, throwOnEndOfStream: false);
        return IsPdf(buf.AsSpan(0, n));
    }

    /// <summary>Loads page <paramref name="index"/>.</summary>
    public PdfPage GetPage(int index)
    {
        ObjectDisposedException.ThrowIf(_doc == IntPtr.Zero, this);
        lock (Pdfium.Lock)
        {
            IntPtr page = Pdfium.FPDF_LoadPage(_doc, index);
            if (page == IntPtr.Zero)
            {
                throw new PdfException($"cannot load page {index + 1}");
            }

            return new PdfPage(page);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_doc != IntPtr.Zero)
        {
            lock (Pdfium.Lock)
            {
                Pdfium.FPDF_CloseDocument(_doc);
            }

            _doc = IntPtr.Zero;
            _pin.Free();
        }
    }
}

/// <summary>A loaded PDF page. Dispose to release it.</summary>
public sealed class PdfPage : IDisposable
{
    internal PdfPage(IntPtr handle)
    {
        Handle = handle;
        lock (Pdfium.Lock)
        {
            Width = Pdfium.FPDF_GetPageWidthF(handle);
            Height = Pdfium.FPDF_GetPageHeightF(handle);
        }
    }

    internal IntPtr Handle { get; private set; }

    /// <summary>Width in points (as rendered: rotation applied).</summary>
    public float Width { get; }

    /// <summary>Height in points.</summary>
    public float Height { get; }

    /// <summary>
    /// Renders the page in grayscale at <paramref name="scale"/> pixels per point, as pypdfium2's
    /// <c>page.render(scale=scale, grayscale=True)</c> does: a white-filled 8-bit bitmap of
    /// <c>ceil(size · scale)</c> pixels, <c>FPDF_GRAYSCALE | FPDF_ANNOT</c>.
    /// </summary>
    public GrayImage Render(double scale)
    {
        int w = (int)Math.Ceiling(Width * scale), h = (int)Math.Ceiling(Height * scale);
        var image = new GrayImage(w, h);
        lock (Pdfium.Lock)
        {
            IntPtr bitmap = Pdfium.FPDFBitmap_CreateEx(w, h, Pdfium.FPDFBitmap_Gray, IntPtr.Zero, 0);
            if (bitmap == IntPtr.Zero)
            {
                throw new PdfException($"cannot allocate a {w}x{h} bitmap");
            }

            try
            {
                Pdfium.FPDFBitmap_FillRect(bitmap, 0, 0, w, h, 0xFFFFFFFF);
                Pdfium.FPDF_RenderPageBitmap(bitmap, Handle, 0, 0, w, h, 0, Pdfium.FPDF_GRAYSCALE | Pdfium.FPDF_ANNOT);
                IntPtr buffer = Pdfium.FPDFBitmap_GetBuffer(bitmap);
                int stride = Pdfium.FPDFBitmap_GetStride(bitmap);
                for (int y = 0; y < h; y++)
                {
                    Marshal.Copy(buffer + (y * stride), image.Pixels, y * w, w);
                }
            }
            finally
            {
                Pdfium.FPDFBitmap_Destroy(bitmap);
            }
        }

        return image;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Handle != IntPtr.Zero)
        {
            lock (Pdfium.Lock)
            {
                Pdfium.FPDF_ClosePage(Handle);
            }

            Handle = IntPtr.Zero;
        }
    }
}
