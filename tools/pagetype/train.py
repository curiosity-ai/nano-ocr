"""Fine-tunes MobileNetV3-Small (timm's mobilenetv3_small_100.lamb_in1k, ImageNet weights) as a
four-way page-type classifier - printed, table, handwritten, photo - and exports it to ONNX for
NanoOCR (src/NanoOcr/Models/Assets/pagetype.onnx).

    python train.py --data data/ --out out/ [--epochs 8]

The input is pagetype_input.model_input: the grayscale page, long side scaled to 512 with
INTER_AREA, top-left on white, gray / 255. The exported graph normalizes and ends in a softmax,
so it returns class probabilities in CLASSES order.
"""
import argparse, io, json, os, random, time
import cv2
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image, ImageFilter

from pagetype_input import CLASSES, SIDE, letterbox

MEAN, STD = 0.45, 0.225


def augment(im: Image.Image, cls: str, rng: random.Random) -> Image.Image:
    w, h = im.size
    # A crop that keeps most of the page: what a scan or a photo frames slightly differently.
    if rng.random() < 0.7:
        f = rng.uniform(0.8, 1.0)
        cw, ch = int(w * rng.uniform(f, 1.0)), int(h * rng.uniform(f, 1.0))
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        im = im.crop((x, y, x + cw, y + ch))
    # The engine deskews before classifying, so only a small residual tilt.
    if rng.random() < 0.3:
        im = im.rotate(rng.uniform(-2.5, 2.5), resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255)
    a = np.asarray(im, dtype=np.float32)
    # Contrast and brightness: a grey scan, a dark photo.
    if rng.random() < 0.6:
        c, b = rng.uniform(0.6, 1.3), rng.uniform(-40, 40)
        a = (a - 128) * c + 128 + b
    if cls != "photo" and rng.random() < 0.35:
        # Scanner / fax look: lower resolution, then binarized or noisy.
        f = rng.uniform(0.3, 0.7)
        small = cv2.resize(np.clip(a, 0, 255).astype(np.uint8), (max(8, int(a.shape[1] * f)), max(8, int(a.shape[0] * f))), interpolation=cv2.INTER_AREA)
        a = cv2.resize(small, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if rng.random() < 0.5:
            t = rng.uniform(110, 190)
            a = np.where(a > t, 255.0, 0.0)
    if rng.random() < 0.3:
        a = a + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(3, 18), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.5)))
    if rng.random() < 0.3:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(25, 80))
        im = Image.open(buf).convert("L")
    return im


class Pages(torch.utils.data.Dataset):
    def __init__(self, root, items, train):
        self.root, self.items, self.train = root, items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        im = Image.open(os.path.join(self.root, it["file"])).convert("L")
        if self.train:
            im = augment(im, it["class"], random.Random(hash((i, time.time_ns()))))
        x = letterbox(np.asarray(im)).astype(np.float32) / 255.0
        return torch.from_numpy(x)[None], CLASSES.index(it["class"])


class Exported(nn.Module):
    """The deployed graph: gray in [0,1] (white = 1) -> class probabilities."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net((x - MEAN) / STD), dim=1)


def build(pretrained=True):
    return timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=pretrained, in_chans=1, num_classes=len(CLASSES))


def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(F.softmax(model((x - MEAN) / STD), 1))
            ys.append(y)
    y, p = torch.cat(ys), torch.cat(ps)
    acc = (p.argmax(1) == y).float().mean().item()
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, q in zip(y.tolist(), p.argmax(1).tolist()):
        cm[t, q] += 1
    return acc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)

    items = [json.loads(l) for l in open(os.path.join(args.data, "manifest.jsonl"))]
    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    counts = {c: sum(1 for i in train if i["class"] == c) for c in CLASSES}
    print("train", counts, "val", {c: sum(1 for i in val if i["class"] == c) for c in CLASSES}, flush=True)
    weights = [1.0 / counts[i["class"]] for i in train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    tl = torch.utils.data.DataLoader(Pages(args.data, train, True), batch_size=args.batch, sampler=sampler, num_workers=args.workers, persistent_workers=True)
    vl = torch.utils.data.DataLoader(Pages(args.data, val, False), batch_size=64, num_workers=args.workers)

    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    best = -1
    for epoch in range(args.epochs):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in tl:
            loss = F.cross_entropy(model((x - MEAN) / STD), y, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            seen += len(y)
            loss_sum += loss.item() * len(y)
        acc, cm = evaluate(model, vl)
        print(f"epoch {epoch + 1}: loss {loss_sum / seen:.3f}, val acc {acc:.3f}, {seen / (time.time() - t0):.1f} img/s\n{cm}", flush=True)
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt")))
    model.eval()
    export = Exported(model).eval()
    dummy = torch.full((1, 1, SIDE, SIDE), 1.0)
    path = os.path.join(args.out, "pagetype.onnx")
    torch.onnx.export(export, dummy, path, input_names=["x"], output_names=["probs"], opset_version=17, dynamo=False)
    with open(os.path.join(args.out, "pagetype.json"), "w") as f:
        json.dump({"classes": CLASSES, "side": SIDE, "input": "gray/255, long side scaled with INTER_AREA, top-left on white", "best_val_acc": best}, f, indent=1)
    print("exported", path, "best val acc", best)


if __name__ == "__main__":
    main()
