import argparse
import json
import os

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from models import STL10CNN
from utils import create_dataloaders, predict_all
from utils.plotting import save_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved STL10 model checkpoint")
    parser.add_argument("--data_root", type=str, default="HW2/STL10")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="HW2/outputs/eval")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]
    class_names = checkpoint["class_names"]

    loaders, dataset_classes = create_dataloaders(
        train_root=os.path.join(args.data_root, "train"),
        test_root=os.path.join(args.data_root, "test"),
        image_size=cfg["image_size"],
        valid_ratio=0.2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        augment=False,
    )

    if class_names != dataset_classes:
        print("Warning: class order in checkpoint differs from dataset classes.")

    model = STL10CNN(
        num_classes=cfg["num_classes"],
        channels=cfg["channels"],
        activation=cfg["activation"],
        pool_type=cfg["pool_type"],
        use_batchnorm=cfg["use_batchnorm"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict_all(model, loaders["test"], device)

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_eval.png"))

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

    with open(os.path.join(args.output_dir, "classification_report_eval.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(os.path.join(args.output_dir, "classification_report_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(args.output_dir, "summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation summary:", summary)
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
