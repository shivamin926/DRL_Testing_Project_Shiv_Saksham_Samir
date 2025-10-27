import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_completion_rate(path_ppo, path_a2c, output_path="completion_rate.png"):
    ppo_df = pd.read_csv(path_ppo, comment="#", header=None)
    a2c_df = pd.read_csv(path_a2c, comment="#", header=None)

    # Assign column names    
    cols = ["r", "l", "t", "result", "match_ratio", "page"]
    ppo_df.columns = cols
    a2c_df.columns = cols

    # Assign column names    
    for df in [ppo_df, a2c_df]:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df["r"] = pd.to_numeric(df["r"], errors="coerce")

    ppo_df["success"] = (ppo_df["page"].str.strip().str.lower() == "done").astype(int)
    a2c_df["success"] = (a2c_df["page"].str.strip().str.lower() == "done").astype(int)

    # Apply smoothing
    window = 200
    ppo_df["success_rate"] = ppo_df["success"].rolling(window=window, min_periods=1).mean()
    a2c_df["success_rate"] = a2c_df["success"].rolling(window=window, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_df["t"], ppo_df["success_rate"], label=f"PPO ({window}-ep avg)", color="tab:blue", alpha=0.9)
    plt.plot(a2c_df["t"], a2c_df["success_rate"], label=f"A2C ({window}-ep avg)", color="tab:orange", alpha=0.9)
    plt.title("Completion Rate Over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Completion Rate (rolling average)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"plot saved to: {output_path}")

# Paths 
base_dir = Path(__file__).resolve().parent.parent.parent
path_ppo = base_dir / "Agent" / "logs" / "ppo" / "full_webflow" / "pass_merged.csv"
path_a2c = base_dir / "Agent" / "logs" / "a2c" / "full_webflow" / "pass.csv"
output_path = base_dir / "Analysis" / "Images" / "completion_rate_ppo_a2c.png"

plot_completion_rate(path_ppo, path_a2c, output_path)
