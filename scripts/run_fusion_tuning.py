import argparse
import csv
import os
import re
import subprocess
import sys


BEST_PATTERN = re.compile(
    r"Run summary: best (?P<metric_name>validation loss|training loss) = "
    r"(?P<metric>\d+\.\d+) at epoch (?P<epoch>\d+)"
)


STAGE1_EXPERIMENTS = [
    {"name": "baseline_ep40", "n_epochs": 40, "batch_size": 128, "lr": 1e-3, "dropout": 0.1, "latent_dim": 128, "n_layers": 1, "weight_decay": 0.0},
    {"name": "lr5e-4_ep40", "n_epochs": 40, "batch_size": 128, "lr": 5e-4, "dropout": 0.1, "latent_dim": 128, "n_layers": 1, "weight_decay": 0.0},
    {"name": "lr2e-3_ep40", "n_epochs": 40, "batch_size": 128, "lr": 2e-3, "dropout": 0.1, "latent_dim": 128, "n_layers": 1, "weight_decay": 0.0},
    {"name": "drop005_ep40", "n_epochs": 40, "batch_size": 128, "lr": 1e-3, "dropout": 0.05, "latent_dim": 128, "n_layers": 1, "weight_decay": 0.0},
    {"name": "drop02_ep40", "n_epochs": 40, "batch_size": 128, "lr": 1e-3, "dropout": 0.2, "latent_dim": 128, "n_layers": 1, "weight_decay": 0.0},
    {"name": "deeper_lat256", "n_epochs": 40, "batch_size": 128, "lr": 1e-3, "dropout": 0.1, "latent_dim": 256, "n_layers": 2, "weight_decay": 0.0},
    {"name": "wd1e-4_ep40", "n_epochs": 40, "batch_size": 128, "lr": 1e-3, "dropout": 0.1, "latent_dim": 128, "n_layers": 1, "weight_decay": 1e-4},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a sequential tuning sweep for Fusion_model.py.")
    parser.add_argument("--preset", default="stage1", choices=["stage1"], help="Experiment preset to run.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--python_cmd", default=sys.executable, help="Python executable used to launch Fusion_model.py.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed passed to each run.")
    parser.add_argument("--val_seed", type=int, default=42, help="Validation split seed passed to each run.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio passed to each run.")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="Early-stop patience shared by runs.")
    parser.add_argument("--min_epochs", type=int, default=15, help="Minimum epochs before early stopping.")
    parser.add_argument("--results_dir", default=os.path.join("results", "tuning"), help="Directory for logs and curves.")
    parser.add_argument("--checkpoints_dir", default=os.path.join("model", "fusion_runs"), help="Directory for per-run checkpoints.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def get_experiments(preset):
    if preset == "stage1":
        return STAGE1_EXPERIMENTS
    raise ValueError(f"Unsupported preset: {preset}")


def parse_summary(log_path):
    best_row = None
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            match = BEST_PATTERN.search(line)
            if match:
                best_row = {
                    "metric_name": match.group("metric_name"),
                    "best_metric": float(match.group("metric")),
                    "best_epoch": int(match.group("epoch")),
                }
    return best_row


def run_experiment(args, experiment, results_dir, checkpoints_dir):
    name = experiment["name"]
    log_path = os.path.abspath(os.path.join(results_dir, f"{name}.log"))
    curve_path = os.path.abspath(os.path.join(results_dir, f"{name}.png"))
    save_path = os.path.abspath(os.path.join(checkpoints_dir, f"{name}.pkl"))

    command = [
        args.python_cmd,
        os.path.abspath("Fusion_model.py"),
        "--batch_size", str(experiment["batch_size"]),
        "--n_epochs", str(experiment["n_epochs"]),
        "--lr", str(experiment["lr"]),
        "--dropout", str(experiment["dropout"]),
        "--latent_dim", str(experiment["latent_dim"]),
        "--n_layers", str(experiment["n_layers"]),
        "--weight_decay", str(experiment["weight_decay"]),
        "--val_ratio", str(args.val_ratio),
        "--val_seed", str(args.val_seed),
        "--seed", str(args.seed),
        "--min_epochs", str(args.min_epochs),
        "--early_stop_patience", str(args.early_stop_patience),
        "--log_file", log_path,
        "--curve_path", curve_path,
        "--save_path", save_path,
    ]

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    printable = " ".join(command)
    print(f"\n=== Running {name} ===")
    print(printable)
    if args.dry_run:
        return {
            "name": name,
            "log_path": log_path,
            "curve_path": curve_path,
            "save_path": save_path,
            "status": "dry_run",
        }

    completed = subprocess.run(command, env=env, check=False)
    summary = parse_summary(log_path)
    return {
        "name": name,
        "log_path": log_path,
        "curve_path": curve_path,
        "save_path": save_path,
        "status": "ok" if completed.returncode == 0 else f"failed({completed.returncode})",
        "best_metric": summary["best_metric"] if summary else None,
        "best_epoch": summary["best_epoch"] if summary else None,
        "metric_name": summary["metric_name"] if summary else None,
    }


def write_summary(rows, results_dir, preset):
    summary_path = os.path.abspath(os.path.join(results_dir, f"{preset}_summary.csv"))
    fieldnames = [
        "name",
        "status",
        "metric_name",
        "best_metric",
        "best_epoch",
        "log_path",
        "curve_path",
        "save_path",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSummary saved to: {summary_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    checkpoints_dir = os.path.abspath(args.checkpoints_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    rows = []
    for experiment in get_experiments(args.preset):
        rows.append(run_experiment(args, experiment, results_dir, checkpoints_dir))

    write_summary(rows, results_dir, args.preset)


if __name__ == "__main__":
    main()
