"""Evaluate a trained model on the test tiles and show where it fails.

    python evaluate.py --bands all
    python evaluate.py --bands rgb

Writes to runs/<bands>/: metrics.json, results.md, confusion_matrix.png,
mistakes.png and training_curves.png. Once both runs exist, it also writes
runs/comparison.md (13 bands against RGB, per class).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data import RGB, EuroSAT, read_tile  # noqa: E402
from model import SmallCNN  # noqa: E402


def predict(model, loader, device="cpu"):
    """Class probabilities and true labels for every tile in the loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def confusion(labels, preds, n_classes):
    """Confusion matrix: rows are true classes, columns are predicted classes."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    return cm


def per_class_scores(cm):
    """Precision, recall and F1 for every class."""
    tp = np.diag(cm).astype(float)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def top_confusions(cm, classes, k=5):
    """The k most frequent (true class, predicted class) mix-ups."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows, cols = np.unravel_index(np.argsort(off, axis=None)[::-1][:k], cm.shape)
    return [{"true": classes[r], "predicted": classes[c], "tiles": int(off[r, c]),
             "share_of_true_class": float(off[r, c] / cm[r].sum())}
            for r, c in zip(rows, cols) if off[r, c] > 0]


def rgb_preview(path):
    """True-colour composite (B04, B03, B02) with a 2-98 % stretch per band."""
    image = read_tile(path, RGB)
    low, high = np.percentile(image, (2, 98), axis=(1, 2), keepdims=True)
    return np.clip((image - low) / np.maximum(high - low, 1e-6), 0, 1).transpose(1, 2, 0)


def plot_confusion(cm, classes, path, title):
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for r in range(len(classes)):
        for c in range(len(classes)):
            if cm[r, c]:
                ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=8,
                        color="white" if share[r, c] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mistakes(paths, labels, probs, classes, path, n=16):
    """The most confident wrong predictions, shown in true colour."""
    preds = probs.argmax(axis=1)
    wrong = np.flatnonzero(preds != labels)
    wrong = wrong[np.argsort(-probs[wrong, preds[wrong]])][:n]
    if len(wrong) == 0:
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(rgb_preview(paths[i]))
        ax.set_title(f"true: {classes[labels[i]]}\npredicted: {classes[preds[i]]} "
                     f"({probs[i, preds[i]]:.0%})", fontsize=8)
    fig.suptitle("Most confident mistakes on the test tiles (true colour)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, label in [("train", "training"), ("val", "validation")]:
        ax_loss.plot(epochs, [h[f"{key}_loss"] for h in history], marker="o", label=label)
        ax_acc.plot(epochs, [h[f"{key}_acc"] for h in history], marker="o", label=label)
    ax_loss.set_ylabel("loss")
    ax_acc.set_ylabel("accuracy")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_results(metrics, path, focus="Forest"):
    lines = [f"# Test results: {metrics['bands']} ({metrics['n_bands']} bands)", "",
             f"Test tiles: {metrics['test_tiles']} | Overall accuracy: {metrics['accuracy']:.1%}", "",
             "| Class | Precision | Recall | F1 | Test tiles |", "|---|---|---|---|---|"]
    for name, s in metrics["per_class"].items():
        lines.append(f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['tiles']} |")
    lines += ["", "## Most frequent mistakes", "",
              "| True class | Predicted as | Tiles | Share of true class |", "|---|---|---|---|"]
    for m in metrics["top_confusions"]:
        lines.append(f"| {m['true']} | {m['predicted']} | {m['tiles']} | {m['share_of_true_class']:.1%} |")
    if focus in metrics["per_class"]:
        def listing(counts):
            return ", ".join(f"{name} {n}" for name, n in counts.items()) or "none"
        f = metrics["focus"]
        lines += ["", f"## {focus}", "",
                  f"- Recall: {metrics['per_class'][focus]['recall']:.1%} of {focus} tiles found",
                  f"- {focus} tiles predicted as something else: {listing(f['missed'])}",
                  f"- Other tiles predicted as {focus}: {listing(f['false_alarms'])}"]
    Path(path).write_text("\n".join(lines) + "\n")


def write_comparison(runs_dir):
    """Per-class F1 for RGB and 13 bands, once both runs have been evaluated."""
    files = {b: runs_dir / b / "metrics.json" for b in ("rgb", "all")}
    if not all(f.exists() for f in files.values()):
        return None
    rgb, ms = (json.loads(files[b].read_text()) for b in ("rgb", "all"))
    lines = ["# 13 bands against RGB", "",
             f"Overall accuracy: RGB {rgb['accuracy']:.1%} | 13 bands {ms['accuracy']:.1%}", "",
             "| Class | F1 RGB | F1 13 bands | Difference |", "|---|---|---|---|"]
    for name in ms["per_class"]:
        a, b = rgb["per_class"][name]["f1"], ms["per_class"][name]["f1"]
        lines.append(f"| {name} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    out = runs_dir / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", choices=["all", "rgb"], default="all")
    parser.add_argument("--data", default="data", help="folder with the EuroSAT class folders")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--focus", default="Forest", help="class to report on in detail")
    return parser.parse_args()


def main():
    args = parse_args()
    run = Path("runs") / args.bands
    checkpoint = torch.load(run / "best.pt", map_location="cpu")
    classes = checkpoint["classes"]
    bands = None if checkpoint["bands"] == "all" else RGB

    # The same test tiles that train.py held out
    split = json.loads((run / "split.json").read_text())
    paths = [Path(args.data) / p for p in split["test"]]
    labels = np.array([classes.index(p.parent.name) for p in paths])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(in_channels=len(checkpoint["mean"]), num_classes=len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    loader = DataLoader(EuroSAT(paths, labels, checkpoint["mean"], checkpoint["std"], bands),
                        batch_size=256, num_workers=args.workers)
    probs, labels = predict(model, loader, device)
    preds = probs.argmax(axis=1)

    cm = confusion(labels, preds, len(classes))
    precision, recall, f1 = per_class_scores(cm)
    metrics = {
        "bands": args.bands,
        "n_bands": len(checkpoint["mean"]),
        "test_tiles": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "tiles": int(cm[i].sum())}
                      for i, name in enumerate(classes)},
        "top_confusions": top_confusions(cm, classes),
    }
    if args.focus in classes:
        k = classes.index(args.focus)
        metrics["focus"] = {
            "missed": {classes[c]: int(cm[k, c]) for c in range(len(classes)) if c != k and cm[k, c]},
            "false_alarms": {classes[r]: int(cm[r, k]) for r in range(len(classes)) if r != k and cm[r, k]},
        }

    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_results(metrics, run / "results.md", args.focus)
    title = f"{metrics['n_bands']} bands, test accuracy {metrics['accuracy']:.1%}"
    plot_confusion(cm, classes, run / "confusion_matrix.png", title)
    plot_mistakes(paths, labels, probs, classes, run / "mistakes.png")
    if (run / "history.json").exists():
        plot_history(json.loads((run / "history.json").read_text()), run / "training_curves.png")

    print((run / "results.md").read_text())
    comparison = write_comparison(Path("runs"))
    if comparison:
        print(comparison.read_text())
    print(f"Figures and tables written to {run}/")


if __name__ == "__main__":
    main()
