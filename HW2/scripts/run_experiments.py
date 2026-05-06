import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and optimization presets")
    parser.add_argument("--data_root", type=str, default="HW2/STL10")
    parser.add_argument("--output_root", type=str, default="HW2/outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["baseline", "aug", "structure", "optimizer"],
        help="Preset list to run in sequence",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_script = os.path.join(project_root, "HW2", "train.py")

    for preset in args.presets:
        cmd = [
            sys.executable,
            train_script,
            "--data_root",
            args.data_root,
            "--preset",
            preset,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--output_root",
            args.output_root,
        ]

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            raise RuntimeError(f"Experiment failed for preset: {preset}")

    print("All experiments finished.")


if __name__ == "__main__":
    main()
