"""Held-out evaluation pages for the page-type classifier. Nothing here is used for training, and
these sets may have licences that would not allow it (IAM, RVL-CDIP): they only score the model.

    python build_eval.py eval/

- RVL-CDIP (100 per class, test split): real 1990s scans, mapped onto the four classes.
- IAM handwriting pages composed from IAM test lines (8 lines a page, as a letter would be).
- CORD test receipts and Total-Text test photos; COCO-Text validation photos.
"""
import io, json, os, random, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
API = "https://datasets-server.huggingface.co"
MAN = open(os.path.join(OUT, "manifest.jsonl"), "w")


def get(url, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=180).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)


def rows(ds, split, offset, length):
    q = urllib.parse.quote(ds, safe="")
    return json.loads(get(f"{API}/rows?dataset={q}&config=default&split={split}&offset={offset}&length={length}"))["rows"]


def save(img, rel, cls, source, extra=None):
    im = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))
    im = im.convert("L")
    im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(os.path.join(OUT, rel)), exist_ok=True)
    im.save(os.path.join(OUT, rel), quality=92)
    MAN.write(json.dumps({"file": rel, "class": cls, "source": source, **(extra or {})}) + "\n")
    MAN.flush()


RVL = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific report', 'scientific publication', 'specification',
       'file folder', 'news article', 'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo']
RVL_CLASS = {'letter': 'printed', 'email': 'printed', 'scientific report': 'printed', 'scientific publication': 'printed',
             'news article': 'printed', 'resume': 'printed', 'memo': 'printed', 'presentation': 'printed',
             'form': 'table', 'specification': 'table', 'budget': 'table', 'invoice': 'table', 'questionnaire': 'table',
             'handwritten': 'handwritten', 'advertisement': None, 'file folder': None}

jobs = []
with ThreadPoolExecutor(8) as ex:
    for off in range(0, 800, 100):
        for r in rows("jordyvl/rvl_cdip_100_examples_per_class", "test", off, 100):
            name = RVL[int(r["row"]["label"])]
            jobs.append(ex.submit(lambda r=r, name=name: (get(r["row"]["image"]["src"]), r["row_idx"], name)))
    for j in jobs:
        b, idx, name = j.result()
        save(b, f"rvl/{idx:04d}.jpg", RVL_CLASS[name] or "ignore", "rvl-cdip", {"rvl": name})
print("rvl done", flush=True)

# IAM test lines -> 20 handwritten pages of 8 lines.
lines = []
for off in range(100, 300, 100):
    for r in rows("Teklia/IAM-line", "test", off, 100):
        lines.append(r["row"]["image"]["src"])
with ThreadPoolExecutor(8) as ex:
    imgs = list(ex.map(lambda u: Image.open(io.BytesIO(get(u))).convert("L"), lines[:160]))
for p in range(20):
    page = Image.new("L", (2480, 3508), 255)
    y = 300
    for im in imgs[p * 8:(p + 1) * 8]:
        s = min(1.0, (2480 - 400) / im.width)
        im2 = im.resize((int(im.width * s), int(im.height * s)))
        page.paste(im2, (200, y))
        y += im2.height + 60
    save(page, f"iam/page{p:02d}.jpg", "handwritten", "iam-pages")
print("iam done", flush=True)

for off, n in [(0, 60)]:
    for r in rows("naver-clova-ix/cord-v2", "test", off, n):
        save(get(r["row"]["image"]["src"]), f"cord/{r['row_idx']:04d}.jpg", "photo", "cord-test")
print("cord done", flush=True)

samples = json.loads(get("https://huggingface.co/datasets/Voxel51/Total-Text-Dataset/resolve/main/samples.json"))["samples"]
test = [s for s in samples if "Test" in s["tags"]]
random.Random(3).shuffle(test)
for s in test[:80]:
    save(get("https://huggingface.co/datasets/Voxel51/Total-Text-Dataset/resolve/main/" + s["filepath"]),
         "totaltext/" + os.path.basename(s["filepath"]), "photo", "total-text-test")
print("totaltext done", flush=True)

for r in rows("howard-hou/COCO-Text", "validation", 0, 100):
    toks = r["row"]["ocr_tokens"]
    save(get(r["row"]["image"]["src"]), f"coco/{r['row_idx']:04d}.jpg", "photo", "coco-text-val", {"has_text": len(toks) > 0})
print("coco done", flush=True)

MAN.close()
with open(os.path.join(OUT, "files.txt"), "w") as f:
    f.write("\n".join(json.loads(l)["file"] for l in open(os.path.join(OUT, "manifest.jsonl"))) + "\n")
