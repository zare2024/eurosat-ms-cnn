"""Train the small CNN on EuroSAT.

    python train.py --bands all     # all 13 Sentinel-2 bands
    python train.py --bands rgb     # only red, green and blue, for comparison

Writes the best model (by validation accuracy), the split and the training
history to runs/<bands>/.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import RGB, EuroSAT, band_stats, find_tiles, split_indices
from model import SmallCNN


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    """One pass over the loader. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", choices=["all", "rgb"], default="all")
    parser.add_argument("--data", default="data", help="folder with the EuroSAT class folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2, help="processes that read tiles")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path("runs") / args.bands
    out.mkdir(parents=True, exist_ok=True)

    paths, labels, classes = find_tiles(args.data)
    split = split_indices(labels, seed=args.seed)
    bands = None if args.bands == "all" else RGB
    print(f"{len(paths)} tiles, {len(classes)} classes | "
          + " | ".join(f"{name} {len(idx)}" for name, idx in split.items()))

    # Normalise with statistics of the training tiles only
    mean, std = band_stats([paths[i] for i in split["train"]], bands)

    def loader(name, shuffle=False, augment=False, batch_size=256):
        idx = split[name]
        data = EuroSAT([paths[i] for i in idx], labels[idx], mean, std, bands, augment)
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          num_workers=args.workers, persistent_workers=args.workers > 0)

    train_loader = loader("train", shuffle=True, augment=True, batch_size=args.batch_size)
    val_loader = loader("val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(in_channels=len(mean), num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | {len(mean)} input bands | {n_params:,} parameters")

    # Save the split so evaluate.py tests on exactly the same tiles
    root = Path(args.data)
    (out / "split.json").write_text(json.dumps(
        {name: [str(paths[i].relative_to(root)) for i in idx] for name, idx in split.items()},
        indent=1))

    best_acc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device=device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d}/{args.epochs} | train loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f} | {time.time() - start:.0f} s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "bands": args.bands,
                        "mean": mean.tolist(), "std": std.tolist()}, out / "best.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best validation accuracy {best_acc:.3f} | model saved to {out / 'best.pt'}")


if __name__ == "__main__":
    main()
