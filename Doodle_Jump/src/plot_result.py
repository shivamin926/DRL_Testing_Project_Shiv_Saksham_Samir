"""
Plot learning curves and evaluation distributions for Doodle-DRL experiments.
All output PNGs are saved under the 'notebooks/' folder.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utility ----------
def ensure_notebooks_dir():
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def smooth(y, weight=0.9):
    s, last = [], y[0] if len(y) else 0
    for v in y:
        last = weight * last + (1 - weight) * v
        s.append(last)
    return np.array(s)

def _read_sb3_monitor(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, comment="#")
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        try:
            df = pd.read_csv(path, comment="#", header=None)
        except Exception:
            df = pd.DataFrame()
    return df

def load_monitor_csv(path):
    df = _read_sb3_monitor(path)
    if df.empty:
        raise ValueError(f"{path} has no episode rows yet.")
    possible_reward = ["r", "reward", "ep_rew_mean", "episode_reward", 0]
    reward_col = next((c for c in possible_reward if c in df.columns), None)
    if reward_col is None and len(df.columns) >= 1:
        reward_col = df.columns[0]
    df["r"] = pd.to_numeric(df[reward_col], errors="coerce")
    possible_len = ["l", "length", "ep_len_mean", 1]
    len_col = next((c for c in possible_len if c in df.columns), None)
    if len_col is None and len(df.columns) >= 2:
        len_col = df.columns[1]
    if len_col in df.columns:
        df["cum_steps"] = pd.to_numeric(df[len_col], errors="coerce").fillna(0).cumsum()
    else:
        df["cum_steps"] = np.arange(len(df)) * 1000
    df = df.dropna(subset=["r"]).reset_index(drop=True)
    return df

def plot_learning_curves(monitor_paths, labels, out_dir):
    out_path = out_dir / "learning_curves.png"
    plt.figure(figsize=(9, 5))
    for path, lab in zip(monitor_paths, labels):
        df = load_monitor_csv(path)
        x, y = df["cum_steps"].values, df["r"].values
        y_s = smooth(y, 0.92) if len(y) > 3 else y
        plt.plot(x, y_s, label=lab)
    plt.xlabel("Environment steps")
    plt.ylabel("Episode return (smoothed)")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] Saved → {out_path}")
    plt.close()

def load_eval_csv(path):
    df = pd.read_csv(path)
    rename_map = {"return": "return_", "best_height(+)": "best_height"}
    df.rename(columns=rename_map, inplace=True)
    return df

def plot_eval_distributions(eval_paths, labels, out_dir):
    # Return histogram
    ret_path = out_dir / "eval_returns.png"
    plt.figure(figsize=(9, 5))
    for p, lab in zip(eval_paths, labels):
        df = load_eval_csv(p)
        plt.hist(df["return_"].values, bins=20, alpha=0.5, label=lab)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title("Return Distribution (Eval)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ret_path, dpi=150)
    print(f"[plot] Saved → {ret_path}")
    plt.close()

    # Crash rate + height
    names, deaths, heights = [], [], []
    for p, lab in zip(eval_paths, labels):
        df = load_eval_csv(p)
        names.append(lab)
        deaths.append(df["death"].mean() * 100.0)
        heights.append(df["best_height"].mean())

    crash_path = out_dir / "eval_crash.png"
    plt.figure(figsize=(7, 4))
    plt.bar(names, deaths)
    plt.ylabel("Crash rate (%)")
    plt.title("Crash (Death) Rate by Model/Persona")
    plt.tight_layout()
    plt.savefig(crash_path, dpi=150)
    print(f"[plot] Saved → {crash_path}")
    plt.close()

    cov_path = out_dir / "eval_coverage.png"
    plt.figure(figsize=(7, 4))
    plt.bar(names, heights)
    plt.ylabel("Mean best height (px)")
    plt.title("Coverage (Vertical Progress) by Model/Persona")
    plt.tight_layout()
    plt.savefig(cov_path, dpi=150)
    print(f"[plot] Saved → {cov_path}")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitors", nargs="*", default=[], help="Paths to SB3 monitor CSVs")
    ap.add_argument("--evals", nargs="*", default=[], help="Paths to eval CSVs from eval.py")
    ap.add_argument("--labels", nargs="*", default=[], help="Labels for plots")
    args = ap.parse_args()

    out_dir = ensure_notebooks_dir()
    def default_labels(paths):
        return [os.path.splitext(os.path.basename(p))[0] for p in paths]

    if args.monitors:
        labs = args.labels[:len(args.monitors)] if args.labels else default_labels(args.monitors)
        plot_learning_curves(args.monitors, labs, out_dir)

    if args.evals:
        labs = args.labels[:len(args.evals)] if args.labels else default_labels(args.evals)
        plot_eval_distributions(args.evals, labs, out_dir)

if __name__ == "__main__":
    main()
