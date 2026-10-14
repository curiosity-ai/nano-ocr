"""Scores `nano-ocr classify` output against the eval manifest: the router's final category per page.

    cd eval && nano-ocr classify [--no-model] $(cat files.txt) > ../classify.tsv
    python score_classify.py classify.tsv eval/

`files.txt` lists the manifest's files, one per line. "right" is the exact category (Mixed counts
as wrong); "engine" is whether the page reached the engine its true class should go to, with the
routing a Curiosity-style setup uses: printed to NanoOCR, table, handwritten and Mixed to
PaddleOCR-VL (the fallback), photo to Florence-2.
"""
import collections, csv, json, os, sys
ENGINE = {"printed": "nano", "table": "vlm", "handwritten": "vlm", "mixed": "vlm", "photo": "florence"}
MAP = {"PrintedText": "printed", "TableOrComplex": "table", "Handwritten": "handwritten", "Photo": "photo", "Mixed": "mixed"}
ev = sys.argv[2]
truth = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(ev, "manifest.jsonl"))}
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
by = collections.defaultdict(list)
allr = []
for r in rows:
    t = truth[r["file"]]
    if t["class"] == "ignore":
        continue
    key = t["source"] if t["source"] != "rvl-cdip" else f"rvl-cdip/{t['class']}"
    pred = MAP[r["category"]]
    by[key].append((t["class"], pred))
    allr.append((t["class"], pred))
print(f"{'set':26} {'n':>4} {'right':>6} {'engine':>6} {'mixed':>6}  predictions")
for k in sorted(by):
    v = by[k]
    print(f"{k:26} {len(v):4} {sum(a == b for a, b in v) / len(v):6.2f} {sum(ENGINE[a] == ENGINE[b] for a, b in v) / len(v):6.2f} {sum(b == 'mixed' for a, b in v) / len(v):6.2f}  {dict(collections.Counter(b for a, b in v))}")
print(f"overall right {sum(a == b for a, b in allr) / len(allr):.3f}, engine {sum(ENGINE[a] == ENGINE[b] for a, b in allr) / len(allr):.3f}, mixed {sum(b == 'mixed' for a, b in allr) / len(allr):.3f}, n={len(allr)}")
hand = [(a, b) for a, b in allr if a == "handwritten"]
print(f"handwritten pages sent to NanoOCR: {sum(ENGINE[b] == 'nano' for a, b in hand)} of {len(hand)}")
