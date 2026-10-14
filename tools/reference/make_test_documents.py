"""Render the small test corpus the parity tests run on.

    python tools/reference/make_test_documents.py tests/data/documents

Writes:
  page_two_column.png   a two-column article page: heading, paragraphs, a list, a table
  page_rotated.png      the same page turned a quarter turn (exercises orient.onnx)
  page_skewed.png       the same page tilted by 1.5 degrees (exercises the deskew)
  page_chinese.png      a few lines of simplified Chinese mixed with Latin (script.onnx, rec_cjk.onnx)
  page_math.png         lines with inline formulas (mathcls.onnx and the math decoder)
  digital.pdf           a born-digital two-page PDF with a text layer (the text-layer path)
  scanned.pdf           page_two_column.png wrapped in a PDF with no text layer (the OCR path)
  tiff/*.tif            the page as TIFF in every compression a scanner writes (raw, PackBits,
                        LZW, Deflate, CCITT G3 and G4), RGB, palette, and multi-page files
  inputs/*              small files for the input-handling tests: five blank pages, a poster
                        page, a password-locked PDF (password "pw"), unreadable files

The pages are deterministic: the same fonts and layout give the same pixels.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

PARAS = [
    "Optical character recognition turns the image of a page into text that a computer can search, "
    "index and edit. A fast recognizer reads a line at a time and needs only a small network.",
    "The detector finds every line of text on the page and returns its box. Each box is cropped, "
    "scaled to a fixed height and read by the recognizer, one character per column of features.",
    "Reading order decides which line comes next. In a two-column article the reader goes down the "
    "left column first and then continues at the top of the right column.",
    "Layout classes group lines into blocks: headings, paragraphs, lists, tables and captions. Page "
    "headers and footers are left out of the markdown unless they are asked for.",
]

LIST = ["1. Detect the lines of text.", "2. Recognize each line.", "3. Order the lines and blocks.",
        "4. Write markdown."]

TABLE = [["Engine", "Pages/s", "Score"], ["Indra", "49.4", "62.1"], ["Tesseract", "2.9", "19.3"],
         ["PaddleOCR", "4.3", "35.8"]]


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def two_column_page() -> Image.Image:
    W, H = 1700, 2200
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(SERIF_BOLD, 54)
    head = ImageFont.truetype(SERIF_BOLD, 36)
    body = ImageFont.truetype(SERIF, 28)
    small = ImageFont.truetype(SANS, 22)
    d.text((150, 70), "Journal of Document Engineering", font=small, fill=0)
    d.text((1350, 70), "Vol. 12", font=small, fill=0)
    d.text((150, 160), "Fast Line Recognition on the CPU", font=title, fill=0)
    col_w, gap, x0 = 640, 120, 150
    cols = [x0, x0 + col_w + gap]
    y = [300, 300]
    lh = 40

    def para(ci, text):
        for ln in wrap(d, text, body, col_w):
            d.text((cols[ci], y[ci]), ln, font=body, fill=0)
            y[ci] += lh
        y[ci] += 24

    d.text((cols[0], y[0]), "1. Introduction", font=head, fill=0)
    y[0] += 64
    para(0, PARAS[0])
    para(0, PARAS[1])
    d.text((cols[0], y[0]), "2. Method", font=head, fill=0)
    y[0] += 64
    para(0, PARAS[2])
    for item in LIST:
        d.text((cols[0] + 20, y[0]), item, font=body, fill=0)
        y[0] += lh
    para(1, PARAS[3])
    d.text((cols[1], y[1]), "3. Results", font=head, fill=0)
    y[1] += 64
    para(1, "Table 1 compares the throughput of three engines on the same machine.")
    cw = [260, 190, 170]
    ty = y[1]
    for r, row in enumerate(TABLE):
        x = cols[1]
        for c, cell in enumerate(row):
            d.text((x + 8, ty + 6), cell, font=body, fill=0)
            x += cw[c]
        ty += 52
    for r in range(len(TABLE) + 1):
        d.line([(cols[1], y[1] + r * 52), (cols[1] + sum(cw), y[1] + r * 52)], fill=0, width=2)
    y[1] = ty + 30
    para(1, "The fastest engine reads a page in about a quarter of a second on four cores.")
    d.text((W // 2 - 10, H - 120), "7", font=small, fill=0)
    return img


def chinese_page() -> Image.Image:
    img = Image.new("L", (1400, 700), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(CJK, 40)
    lat = ImageFont.truetype(SERIF, 34)
    lines = ["光学字符识别把页面的图像变成文字。", "检测器找到每一行文字并返回它的框。",
             "识别器一次读取一行。", "This line is in English."]
    for i, t in enumerate(lines):
        d.text((100, 100 + i * 120), t, font=f if i < 3 else lat, fill=0)
    return img


def math_page() -> Image.Image:
    img = Image.new("L", (1600, 800), 255)
    d = ImageDraw.Draw(img)
    body = ImageFont.truetype(SERIF, 34)
    lines = ["For every x we have x² + y² ≤ 1 when the point lies in the disk.",
             "The sum ∑ aᵢ converges if |aᵢ| ≤ C/i² for some C > 0.",
             "Let f(x) = αx + β and g(x) = eˣ.",
             "Plain text without any formula at all."]
    for i, t in enumerate(lines):
        d.text((100, 100 + i * 140), t, font=body, fill=0)
    return img


def digital_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    W, H = letter
    for page in range(2):
        c.setFont("Times-Bold", 20)
        c.drawString(72, H - 90, f"Section {page + 1}: Digital documents")
        c.setFont("Times-Roman", 11)
        y = H - 130
        for p in PARAS[page * 2: page * 2 + 2]:
            words, line = p.split(), ""
            for w in words:
                t = (line + " " + w).strip()
                if c.stringWidth(t, "Times-Roman", 11) > W - 144:
                    c.drawString(72, y, line)
                    y -= 15
                    line = w
                else:
                    line = t
            c.drawString(72, y, line)
            y -= 28
        for item in LIST:
            c.drawString(90, y, item)
            y -= 15
        y -= 20
        for row in TABLE:
            x = 72
            for cell, w in zip(row, [150, 100, 100]):
                c.drawString(x, y, cell)
                x += w
            y -= 18
        c.setFont("Times-Roman", 9)
        c.drawString(W / 2, 40, str(page + 1))
        c.showPage()
    c.save()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page = two_column_page()
    page.save(out / "page_two_column.png")
    page.rotate(90, expand=True, fillcolor=255).save(out / "page_rotated.png")
    page.rotate(1.5, resample=Image.BICUBIC, expand=False, fillcolor=255).save(out / "page_skewed.png")
    chinese_page().save(out / "page_chinese.png")
    math_page().save(out / "page_math.png")
    digital_pdf(out / "digital.pdf")
    import datetime

    fixed = datetime.datetime(2026, 1, 1)
    page.convert("RGB").save(out / "scanned.pdf", resolution=200.0, creationDate=fixed, modDate=fixed)
    tiffs(page, out / "tiff")
    inputs(out / "inputs")


def tiffs(page: Image.Image, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    half = page.resize((page.width // 2, page.height // 2), Image.BILINEAR)
    quarter = page.resize((page.width // 4, page.height // 4), Image.BILINEAR)
    for comp in ("raw", "packbits", "tiff_lzw", "tiff_adobe_deflate"):
        (quarter if comp == "raw" else half).save(out / f"gray_{comp}.tif", compression=comp)
    bw = half.point(lambda v: 255 if v > 128 else 0).convert("1")
    for comp in ("group3", "group4", "packbits"):
        bw.save(out / f"bilevel_{comp}.tif", compression=comp)
    half.convert("RGB").save(out / "rgb_lzw.tif", compression="tiff_lzw")
    quarter.convert("P").save(out / "palette.tif", compression="packbits")
    frames = [half, half.rotate(90, expand=True, fillcolor=255), Image.new("L", (300, 400), 255)]
    frames[0].save(out / "multipage.tif", save_all=True, append_images=frames[1:], compression="tiff_lzw")
    bwf = [f.point(lambda v: 255 if v > 128 else 0).convert("1") for f in frames]
    bwf[0].save(out / "multipage_g4.tif", save_all=True, append_images=bwf[1:], compression="group4")


def inputs(out: Path) -> None:
    import pikepdf
    import pypdfium2 as pdfium

    out.mkdir(parents=True, exist_ok=True)

    def pdf(path, sizes):
        doc = pdfium.PdfDocument.new()
        for w, h in sizes:
            doc.new_page(w, h)
        doc.save(str(path))

    pdf(out / "five.pdf", [(200, 300)] * 5)
    pdf(out / "poster.pdf", [(14400, 14400), (612, 792)])
    pdf(out / "plain.pdf", [(200, 300)])
    pikepdf.open(out / "plain.pdf").save(out / "locked.pdf", encryption=pikepdf.Encryption(user="pw", owner="owner"))
    (out / "plain.pdf").unlink()
    frames = [Image.new("L", (60, 80), v) for v in (255, 200, 255)]
    frames[0].save(out / "three.tiff", save_all=True, append_images=frames[1:])
    (out / "garbage.pdf").write_bytes(b"%PDF-1.4 broken")
    (out / "empty.png").write_bytes(b"")
    (out / "text.txt").write_bytes(b"hello")
    Image.new("L", (40, 50), 255).save(out / "image_named.pdf.png")
    (out / "image_named.pdf.png").rename(out / "image_named.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/data/documents"))
