"""Builds the page-type training set: printed, table, handwritten, photo.

    python build_dataset.py data/ [sources]

Every source is permissively licensed (see README.md), because the weights ship inside NanoOCR.
Every image is stored grayscale with its long side at most 1024 px (the model sees 512), with one
manifest line per image: file, class, source, licence, split (train, or val for about 12%).
Re-running resumes: images already in the manifest are skipped. Needs pillow; CORD needs pyarrow.
"""
import io, json, os, random, sys, time, zipfile, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

OUT = sys.argv[1]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)
MANIFEST = os.path.join(OUT, "manifest.jsonl")
done = set()
if os.path.exists(MANIFEST):
    done = {json.loads(l)["file"] for l in open(MANIFEST)}
rng = random.Random(1234)
API = "https://datasets-server.huggingface.co"


def get_json(url, tries=6):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)


def rows(ds, split, offset, length, config="default"):
    q = urllib.parse.quote(ds, safe="")
    return get_json(f"{API}/rows?dataset={q}&config={config}&split={split}&offset={offset}&length={length}")["rows"]


def total(ds, split, config="default"):
    q = urllib.parse.quote(ds, safe="")
    return get_json(f"{API}/rows?dataset={q}&config={config}&split={split}&offset=0&length=1")["num_rows_total"]


def save(img_bytes, name, cls, source, licence, split):
    rel = f"{cls}/{name}.jpg"
    if rel in done:
        return
    im = Image.open(io.BytesIO(img_bytes))
    im.draft("L", (1024, 1024))
    im = im.convert("L")
    im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    os.makedirs(os.path.join(OUT, cls), exist_ok=True)
    im.save(os.path.join(OUT, rel), quality=92)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps({"file": rel, "class": cls, "source": source, "licence": licence, "split": split}) + "\n")
    done.add(rel)


def fetch(url, tries=5):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=180).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)


def from_rows(ds, split, n, cls, source, licence, accept=lambda r: True, image_key="image", pool=None, step=100):
    """Samples rows at random offsets, keeps the ones `accept` likes, until n are saved."""
    t = total(ds, split)
    offsets = list(range(0, t, step))
    rng.shuffle(offsets)
    kept = 0
    jobs = []
    with ThreadPoolExecutor(8) as ex:
        for off in offsets:
            if kept >= n:
                break
            for r in rows(ds, split, off, min(step, t - off)):
                if kept >= n:
                    break
                row = r["row"]
                if not accept(row):
                    continue
                img = row[image_key]
                if not isinstance(img, dict) or "src" not in img:
                    continue
                name = f"{source}_{split}_{r['row_idx']}"
                trainval = "val" if rng.random() < 0.12 else "train"
                jobs.append(ex.submit(lambda u=img["src"], nm=name, s=trainval: save(fetch(u), nm, cls, source, licence, s)))
                kept += 1
        for j in jobs:
            try:
                j.result()
            except Exception as e:
                print("skip", e, file=sys.stderr)
    print(f"{source}: {kept}", flush=True)


def doclaynet_share(row, cats):
    w = row["metadata"]["coco_width"] * row["metadata"]["coco_height"]
    return sum(a for a, c in zip(row["area"], row["category_id"]) if c in cats) / w


def want(name):
    return ONLY is None or name in ONLY


# DocLayNet categories: 1 Caption 2 Footnote 3 Formula 4 List-item 5 Page-footer 6 Page-header
# 7 Picture 8 Section-header 9 Table 10 Text 11 Title
if want("doclaynet"):
    from_rows("docling-project/DocLayNet-v1.2", "train", 350, "printed", "doclaynet", "CDLA-Permissive-1.0",
              accept=lambda r: doclaynet_share(r, {9}) < 0.05 and doclaynet_share(r, {7}) < 0.15 and doclaynet_share(r, {4, 10}) > 0.25, step=40)
    from_rows("docling-project/DocLayNet-v1.2", "train", 350, "table", "doclaynet", "CDLA-Permissive-1.0",
              accept=lambda r: doclaynet_share(r, {9}) > 0.35, step=40)
if want("publaynet"):
    from_rows("creative-graphic-design/PubLayNet", "train", 300, "printed", "publaynet", "CDLA-Permissive-1.0",
              accept=lambda r: True)
if want("fatura"):
    from_rows("mathieu1256/FATURA2-invoices", "train", 200, "table", "fatura2", "CC-BY-4.0")
if want("katanaml"):
    from_rows("katanaml-org/invoices-donut-data-v1", "train", 150, "table", "katanaml-invoices", "MIT")
if want("notes"):
    from_rows("NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge", "train", 400, "handwritten", "notes-bank", "Apache-2.0", step=20)
if want("cocotext"):
    from_rows("howard-hou/COCO-Text", "train", 400, "photo", "coco-text", "CC-BY-4.0")
if want("cord"):
    # The dataset server answers 500 for CORD's train split; one parquet shard holds 200 receipts.
    import pyarrow.parquet as pq
    shard = os.path.join(OUT, "_cord0.parquet")
    if not os.path.exists(shard):
        urllib.request.urlretrieve("https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/train-00000-of-00004-b4aaeceff1d90ecb.parquet", shard)
    imgs = pq.read_table(shard, columns=["image"]).column("image").to_pylist()
    idx = list(range(len(imgs)))
    rng.shuffle(idx)
    for i in idx[:150]:
        save(imgs[i]["bytes"], f"cord_train0_{i}", "photo", "cord", "CC-BY-4.0", "val" if rng.random() < 0.12 else "train")
    os.remove(shard)
    print("cord: 150", flush=True)
if want("totaltext"):
    # Total-Text: the Train-tagged images only (the Test ones are the OCR benchmark's).
    samples = json.load(urllib.request.urlopen("https://huggingface.co/datasets/Voxel51/Total-Text-Dataset/resolve/main/samples.json", timeout=300))["samples"]
    train = [s for s in samples if "Train" in s["tags"]]
    rng.shuffle(train)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda s: save(fetch("https://huggingface.co/datasets/Voxel51/Total-Text-Dataset/resolve/main/" + s["filepath"]),
                                   "totaltext_" + os.path.basename(s["filepath"]).split(".")[0], "photo", "total-text", "BSD-3-Clause",
                                   "val" if rng.random() < 0.12 else "train"), train[:200]))
    print("total-text: 200", flush=True)
if want("gnhk"):
    # The archive holds train_data.zip and test_data.zip; only the train pages are used.
    zpath = os.path.join(OUT, "_gnhk.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve("https://huggingface.co/datasets/staghado/GNHK-Dataset/resolve/main/GNHK-dataset.zip", zpath)
    with zipfile.ZipFile(zpath) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read("train_data.zip")))
        names = [n for n in inner.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(names)
        for n in names[:300]:
            save(inner.read(n), "gnhk_" + os.path.basename(n).rsplit(".", 1)[0], "handwritten", "gnhk", "CC-BY-4.0", "val" if rng.random() < 0.12 else "train")
    os.remove(zpath)
    print(f"gnhk: {min(300, len(names))}", flush=True)
