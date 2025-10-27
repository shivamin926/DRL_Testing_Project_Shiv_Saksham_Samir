import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_questions_nextpage_rate(fail1_path, pass_path, output_path="questions_nextpage_rate.png"):

    fail1_df = pd.read_csv(fail1_path, comment="#", header=None)
    pass_df = pd.read_csv(pass_path, comment="#", header=None)

    # Assign column names    
    cols = ["r", "l", "t", "page"]
    for df in [fail1_df, pass_df]:
        df.columns = cols
        df["t"] = pd.to_numeric(df["t"], errors="coerce")

    # Finding whether next page is reached
    def add_reach_flag(df):
        df["reached_next"] = df["page"].str.strip().str.lower().isin(
            ["----------review.html"]
        ).astype(int)
        return df

    fail1_df = add_reach_flag(fail1_df)
    pass_df = add_reach_flag(pass_df)

    # Apply smoothing
    window = 200
    fail1_df["reach_rate"] = fail1_df["reached_next"].rolling(window=window, min_periods=1).mean()
    pass_df["reach_rate"] = pass_df["reached_next"].rolling(window=window, min_periods=1).mean()

    # Plot 
    plt.figure(figsize=(10, 6))
    plt.plot(fail1_df["t"], fail1_df["reach_rate"], label=f"Fail1 (before penalties)", color="tab:red", alpha=0.8)
    plt.plot(pass_df["t"], pass_df["reach_rate"], label=f"Pass (after revisit penalty & step cost)", color="tab:green", alpha=0.9)
    plt.title("QuestionsEnv – Learning to Reach Next Page Over Time")
    plt.xlabel("Timesteps")
    plt.ylabel(f"Reach Rate to Review/Done (rolling avg, window={window})")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

# Paths
base_dir = Path(__file__).resolve().parent.parent.parent
fail1_path = base_dir / "Agent" / "logs" / "ppo" / "page3" / "fail1.csv"
pass_path  = base_dir / "Agent" / "logs" / "ppo" / "page3" / "pass.csv"
output_path = base_dir / "Analysis" / "Images" / "questions_nextpage_rate.png"

plot_questions_nextpage_rate(fail1_path, pass_path, output_path)
