import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from config import get_config
from models import STL10CNN
from utils import create_dataloaders, evaluate, predict_all, set_seed, train_one_epoch
from utils.plotting import save_confusion_matrix, save_history_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on STL-10 style folder dataset")
    parser.add_argument("--data_root", type=str, default="HW2/STL10", help="Path to dataset root")
    parser.add_argument("--preset", type=str, default="baseline", choices=["baseline", "aug", "structure", "optimizer"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, default="HW2/outputs")
    return parser.parse_args()


def build_optimizer(config, model):
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(config, optimizer, epochs):
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def save_classification_outputs(y_true, y_pred, class_names, output_dir):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(output_dir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in class_names:
            row = report_dict[cls]
            writer.writerow([cls, row["precision"], row["recall"], row["f1-score"], row["support"]])

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    with open(os.path.join(output_dir, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = get_config(args.preset)

    train_root = os.path.join(args.data_root, "train")
    test_root = os.path.join(args.data_root, "test")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.name}_{timestamp}"
    output_dir = os.path.join(args.output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, class_names = create_dataloaders(
        train_root=train_root,
        test_root=test_root,
        image_size=cfg.image_size,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=cfg.augment,
    )

    model = STL10CNN(
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        activation=cfg.activation,
        pool_type=cfg.pool_type,
        use_batchnorm=cfg.use_batchnorm,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, args.epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "lr": [],
    }

    best_valid_acc = -np.inf
    best_ckpt_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, loaders["valid"], criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        history["lr"].append(lr_now)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} Valid Acc: {valid_acc:.4f} | LR: {lr_now:.6f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "class_names": class_names,
                    "best_valid_acc": best_valid_acc,
                },
                best_ckpt_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "class_names": class_names,
            "best_valid_acc": best_valid_acc,
        },
        os.path.join(output_dir, "last_model.pth"),
    )

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_history_curves(history, output_dir)

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict_all(model, loaders["test"], device)
    summary = save_classification_outputs(y_true, y_pred, class_names, output_dir)

    print("Best validation accuracy:", best_valid_acc)
    print("Test summary:", summary)
    print("Artifacts saved to:", output_dir)


if __name__ == "__main__":
    main()
