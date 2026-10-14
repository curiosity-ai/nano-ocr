"""Synthetic office pages for the page-type model: the sparse printed documents and simple forms
that the permissively licensed sources lack, rendered and then degraded like a fax or photocopy.

    python synthesize.py data/ [--printed 600] [--table 400] [--seed 7]

Printed: letters, memos, e-mail printouts, resumes, reports, slides. Table: budgets, forms,
questionnaires, specification sheets. Appends to data/manifest.jsonl (source "synthetic",
licence "generated"). Needs pillow and numpy; uses the fonts under /usr/share/fonts.
"""
import argparse, glob, io, json, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORDS = ("the of and to in a is that for it as was with be by on not he this are or his from at which but have an they "
         "you were her all she there would their we him been has when who will more no if out so said what up its about "
         "into than them can only other new some could time these two may then do first any my now such like our over man "
         "me even most made after also did many before must through back years where much your way well down should because "
         "each just those people how too little state good very make world still own see men work long get here between both "
         "life being under never day same another know while last might us great old year off come since against go came "
         "right used take three report meeting budget project review policy committee results program analysis office "
         "information research company account request service management sales product market development data "
         "schedule proposal agreement contract department division regional quarterly annual summary approval").split()
FIRST = "John Mary Robert Linda James Susan Michael Karen David Nancy William Lisa Richard Betty Thomas Helen Charles Sandra".split()
LAST = "Smith Johnson Williams Brown Jones Miller Davis Wilson Anderson Taylor Thomas Moore Martin Jackson White Harris Clark".split()
MONTHS = "January February March April May June July August September October November December".split()


def fonts():
    found = {}
    for pattern, key in [("liberation/LiberationSerif-Regular", "serif"), ("liberation/LiberationSerif-Bold", "serif-bold"),
                         ("liberation/LiberationSans-Regular", "sans"), ("liberation/LiberationSans-Bold", "sans-bold"),
                         ("liberation/LiberationMono-Regular", "mono"), ("dejavu/DejaVuSerif.", "serif"), ("dejavu/DejaVuSans.", "sans"),
                         ("dejavu/DejaVuSansMono.", "mono"), ("freefont/FreeSerif.", "serif"), ("freefont/FreeSans.", "sans"),
                         ("freefont/FreeMono.", "mono"), ("crosextra/Caladea-Regular", "serif"), ("crosextra/Carlito-Regular", "sans"),
                         ("dejavu/DejaVuSans-Bold.", "sans-bold"), ("freefont/FreeSansBold.", "sans-bold")]:
        for f in glob.glob(f"/usr/share/fonts/**/{pattern}*", recursive=True):
            found.setdefault(key, []).append(f)
    return found


FONTS = fonts()


class Page:
    def __init__(self, rng, w=1700, h=2200):
        self.rng, self.w, self.h = rng, w, h
        self.im = Image.new("L", (w, h), 255)
        self.d = ImageDraw.Draw(self.im)
        self.margin = rng.randint(110, 220)
        self.y = rng.randint(110, 240)
        self.family = rng.choice(["serif", "sans", "mono"] if rng.random() < 0.25 else ["serif", "sans"])
        self.size = rng.randint(26, 36)

    def font(self, size=None, bold=False, family=None):
        fam = family or self.family
        key = fam + "-bold" if bold and fam + "-bold" in FONTS else fam
        return ImageFont.truetype(self.rng.choice(FONTS[key]), size or self.size)

    def words(self, n):
        return " ".join(self.rng.choice(WORDS) for _ in range(n))

    def sentence(self):
        s = self.words(self.rng.randint(6, 18))
        return s[0].upper() + s[1:] + self.rng.choice([".", ".", ".", ",", ";"])

    def text(self, x, s, size=None, bold=False, family=None, color=0):
        f = self.font(size, bold, family)
        self.d.text((x, self.y), s, font=f, fill=color)
        return f

    def line(self, s, size=None, bold=False, x=None, gap=1.35, family=None):
        f = self.text(self.margin if x is None else x, s, size, bold, family)
        self.y += int((size or self.size) * gap)
        return f

    def paragraph(self, width=None, size=None, indent=0, family=None):
        size = size or self.size
        f = self.font(size, family=family)
        width = width or (self.w - 2 * self.margin)
        text = " ".join(self.sentence() for _ in range(self.rng.randint(2, 6)))
        line, first = "", True
        for word in text.split():
            trial = (line + " " + word).strip()
            if f.getlength(trial) + (indent if first else 0) > width:
                self.d.text((self.margin + (indent if first else 0), self.y), line, font=f, fill=0)
                self.y += int(size * 1.35)
                line, first = word, False
            else:
                line = trial
        self.d.text((self.margin + (indent if first else 0), self.y), line, font=f, fill=0)
        self.y += int(size * (1.35 + self.rng.uniform(0.4, 1.2)))

    def name(self):
        return f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"

    def date(self):
        return f"{self.rng.choice(MONTHS)} {self.rng.randint(1, 28)}, {self.rng.randint(1975, 2024)}"

    def full(self):
        return self.y > self.h - self.margin - 120


def letter(p):
    r = p.rng
    if r.random() < 0.7:
        p.line(p.words(2).title() + " " + r.choice(["Inc.", "Corporation", "Company", "Associates", "Group"]), size=p.size + 14, bold=True, x=p.margin if r.random() < 0.5 else p.w // 3)
        p.line(f"{r.randint(10, 9999)} {p.words(1).title()} Street, {p.words(1).title()}", size=p.size - 8)
        p.d.line((p.margin, p.y, p.w - p.margin, p.y), fill=0, width=r.randint(1, 4))
        p.y += 60
    p.line(p.date())
    p.y += 40
    for s in [p.name(), p.words(2).title(), f"{r.randint(10, 999)} {p.words(1).title()} Avenue", p.words(1).title()]:
        p.line(s)
    p.y += 40
    p.line(f"Dear {r.choice(['Mr.', 'Ms.', 'Dr.'])} {r.choice(LAST)}:")
    p.y += 20
    while not p.full() and r.random() < 0.85:
        p.paragraph(indent=r.choice([0, 0, 60]))
    if p.y < p.h - 400:
        p.line(r.choice(["Sincerely,", "Very truly yours,", "Best regards,"]))
        p.y += 120
        p.line(p.name())


def memo(p):
    r = p.rng
    p.line(r.choice(["MEMORANDUM", "M E M O R A N D U M", "INTEROFFICE MEMO", "Memo"]), size=p.size + 16, bold=True, x=p.margin if r.random() < 0.6 else p.w // 3)
    p.y += 30
    tab = p.margin + r.randint(160, 260)
    for label in ["TO:", "FROM:", "DATE:", "SUBJECT:"] + (["CC:"] if r.random() < 0.4 else []):
        p.text(p.margin, label, bold=r.random() < 0.5)
        p.line(p.date() if label == "DATE:" else (p.words(r.randint(2, 6)).title() if label == "SUBJECT:" else p.name()), x=tab)
    p.d.line((p.margin, p.y, p.w - p.margin, p.y), fill=0, width=r.randint(1, 3))
    p.y += 50
    while not p.full() and r.random() < 0.9:
        p.paragraph()


def email(p):
    r = p.rng
    p.size = r.randint(22, 30)
    for label in ["From:", "Sent:", "To:", "Cc:", "Subject:"]:
        if label == "Cc:" and r.random() < 0.5:
            continue
        p.text(p.margin, label, bold=True)
        p.line(p.date() if label == "Sent:" else (p.words(4).title() if label == "Subject:" else f"{p.name()} <{r.choice(LAST).lower()}@{p.words(1)}.com>"), x=p.margin + 170)
    p.y += 40
    while not p.full() and r.random() < 0.85:
        if r.random() < 0.2:
            for _ in range(r.randint(2, 5)):
                p.line("> " + p.words(r.randint(5, 11)), family="mono")
            p.y += 20
        else:
            p.paragraph()


def resume(p):
    r = p.rng
    p.line(p.name(), size=p.size + 20, bold=True, x=p.w // 2 - 250)
    p.line(f"{r.randint(10, 999)} {p.words(1).title()} Road  |  {r.randint(200, 999)}-{r.randint(100, 999)}-{r.randint(1000, 9999)}", size=p.size - 6, x=p.w // 2 - 330)
    p.y += 30
    while not p.full():
        p.line(r.choice(["EXPERIENCE", "EDUCATION", "SKILLS", "PUBLICATIONS", "OBJECTIVE", "REFERENCES"]), bold=True)
        p.d.line((p.margin, p.y - 8, p.w - p.margin, p.y - 8), fill=0, width=2)
        for _ in range(r.randint(1, 3)):
            if p.full():
                break
            p.text(p.w - p.margin - 260, f"{r.randint(1975, 2020)} - {r.randint(1980, 2024)}")
            p.line(p.words(3).title(), bold=True)
            for _ in range(r.randint(1, 4)):
                p.line("•  " + p.words(r.randint(5, 12)), x=p.margin + 30)
        p.y += 30


def report(p):
    r = p.rng
    p.line(p.words(r.randint(3, 7)).title(), size=p.size + 18, bold=True)
    p.y += 30
    cols = r.choice([1, 1, 2])
    width = (p.w - 2 * p.margin - 60 * (cols - 1)) // cols
    top, base = p.y, p.margin
    for c in range(cols):
        p.margin, p.y = base + c * (width + 60), top
        while not p.full():
            if r.random() < 0.2:
                p.line(p.words(r.randint(2, 5)).title(), bold=True)
            p.paragraph(width=width)
    p.margin = base


def slide(p):
    r = p.rng
    p.d.rectangle((p.margin - 40, p.y - 40, p.w - p.margin + 40, p.h - 200), outline=0, width=r.randint(2, 6))
    p.line(p.words(r.randint(2, 5)).title(), size=p.size + 30, bold=True)
    p.y += 50
    for _ in range(r.randint(3, 7)):
        p.line("•  " + p.words(r.randint(3, 8)), size=p.size + 12, x=p.margin + 20, gap=1.8)


def budget(p):
    r = p.rng
    p.line(p.words(3).title() + " " + r.choice(["Budget", "Expenses", "Forecast", "Statement"]), size=p.size + 10, bold=True)
    p.y += 30
    cols = r.randint(3, 7)
    colw = (p.w - 2 * p.margin - 380) // cols
    ruled = r.random() < 0.6
    rows = r.randint(12, 34)
    rh = int(p.size * 1.5)
    top = p.y
    for i in range(rows):
        if p.full():
            break
        p.text(p.margin, p.words(r.randint(1, 3)).title() if i else "", bold=i == 0)
        for c in range(cols):
            s = str(r.randint(1975, 1999) + c) if i == 0 else f"{r.randint(0, 99999):,}" + (f".{r.randint(0, 99):02d}" if r.random() < 0.3 else "")
            x = p.margin + 380 + (c + 1) * colw - int(p.font().getlength(s)) - 10
            p.text(x, s, bold=i == 0)
        p.y += rh
        if ruled:
            p.d.line((p.margin, p.y - 6, p.w - p.margin, p.y - 6), fill=0, width=1)
    if ruled and r.random() < 0.7:
        for c in range(cols + 1):
            x = p.margin + 380 + c * colw
            p.d.line((x, top - 6, x, p.y - 6), fill=0, width=1)


def form(p):
    r = p.rng
    p.line(p.words(3).upper() + " FORM", size=p.size + 8, bold=True)
    p.y += 20
    while not p.full():
        if r.random() < 0.2:
            p.d.rectangle((p.margin, p.y, p.w - p.margin, p.y + int(p.size * 1.6)), fill=0)
            p.text(p.margin + 10, p.words(2).upper(), bold=True, color=255)
            p.y += int(p.size * 2.2)
            continue
        x = p.margin
        for _ in range(r.randint(1, 3)):
            label = p.words(r.randint(1, 3)).title() + ":"
            f = p.font(p.size - 4)
            p.d.text((x, p.y), label, font=f, fill=0)
            w = r.randint(200, 500)
            lx = x + int(f.getlength(label)) + 10
            if r.random() < 0.3:
                p.d.rectangle((lx, p.y - 6, lx + w, p.y + p.size + 6), outline=0, width=2)
            else:
                p.d.line((lx, p.y + p.size, lx + w, p.y + p.size), fill=0, width=2)
            x = lx + w + 40
            if x > p.w - p.margin - 400:
                break
        p.y += int(p.size * 2.0)


def questionnaire(p):
    r = p.rng
    p.line(p.words(2).title() + " Questionnaire", size=p.size + 10, bold=True)
    p.y += 20
    n = 1
    while not p.full():
        p.line(f"{n}. " + p.sentence().replace(".", "?"))
        x = p.margin + 40
        for opt in r.choice([["Yes", "No"], ["Never", "Rarely", "Sometimes", "Often", "Always"], ["1", "2", "3", "4", "5"], [p.words(1).title() for _ in range(3)]]):
            p.d.rectangle((x, p.y + 4, x + p.size, p.y + 4 + p.size), outline=0, width=2)
            p.text(x + p.size + 12, opt)
            x += int(p.font().getlength(opt)) + p.size + 60
        p.y += int(p.size * 2.2)
        n += 1


def specification(p):
    r = p.rng
    p.line("SPECIFICATION " + p.words(2).upper(), size=p.size + 8, bold=True)
    p.y += 20
    tab = p.margin + r.randint(450, 700)
    boxed = r.random() < 0.6
    while not p.full():
        if boxed:
            p.d.rectangle((p.margin, p.y - 8, p.w - p.margin, p.y + int(p.size * 1.5)), outline=0, width=1)
            p.d.line((tab - 20, p.y - 8, tab - 20, p.y + int(p.size * 1.5)), fill=0, width=1)
        p.text(p.margin + 10, p.words(r.randint(1, 4)).title())
        p.text(tab, f"{r.uniform(0, 999):.{r.randint(0, 3)}f} " + r.choice(["mg", "%", "mm", "ppm", "g", "ml", ""]))
        p.y += int(p.size * 1.6) + (0 if boxed else 10)


PRINTED = [letter, letter, memo, memo, email, resume, report, slide]
TABLE = [budget, form, questionnaire, specification]


def degrade(im, rng):
    """Office-copier and fax looks; about a third of the pages stay clean."""
    if rng.random() < 0.33:
        return im
    a = np.asarray(im.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BILINEAR, fillcolor=255), dtype=np.float32)
    h, w = a.shape
    kind = rng.choice(["fax", "copy", "scan"])
    if kind == "fax":
        f = rng.uniform(0.3, 0.5)
        small = Image.fromarray(a.astype(np.uint8)).resize((int(w * f), int(h * f)), Image.Resampling.BILINEAR)
        a = np.asarray(small.resize((w, h), Image.Resampling.NEAREST), dtype=np.float32)
        a = np.where(a > rng.uniform(120, 200), 255.0, 0.0)
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).random(a.shape)
        a[noise < rng.uniform(0.0005, 0.004)] = 0
    elif kind == "copy":
        a = np.clip(a * rng.uniform(0.8, 1.0) - rng.uniform(0, 40), 0, 255)
        yy, xx = np.mgrid[0:h, 0:w]
        vign = 1 - rng.uniform(0.05, 0.25) * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 4
        a = a * vign
        for _ in range(rng.randint(0, 3)):
            x = rng.randint(0, w - 4)
            a[:, x:x + rng.randint(1, 4)] *= rng.uniform(0.3, 0.8)
    else:
        a = np.clip(a + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 14), a.shape) + rng.uniform(-25, 0), 0, 255)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--printed", type=int, default=600)
    ap.add_argument("--table", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    man = open(os.path.join(a.out, "manifest.jsonl"), "a")
    for cls, n, kinds in [("printed", a.printed, PRINTED), ("table", a.table, TABLE)]:
        os.makedirs(os.path.join(a.out, cls), exist_ok=True)
        for i in range(n):
            kind = kinds[i % len(kinds)]
            p = Page(random.Random(rng.random()))
            kind(p)
            im = degrade(p.im, rng)
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            rel = f"{cls}/synthetic_{kind.__name__}_{i:04d}.jpg"
            im.save(os.path.join(a.out, rel), quality=rng.randint(60, 92))
            man.write(json.dumps({"file": rel, "class": cls, "source": "synthetic", "licence": "generated", "split": "val" if rng.random() < 0.12 else "train"}) + "\n")
        print(cls, n, flush=True)


if __name__ == "__main__":
    main()
