import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import OrderedDict

def plot_experience_comparison(fail1_path, fail2_path, pass_path, output_path="experience_outcomes.png"):

    fail1 = pd.read_csv(fail1_path, comment="#", header=None)
    fail2 = pd.read_csv(fail2_path, comment="#", header=None)
    ppass = pd.read_csv(pass_path,  comment="#", header=None)

    cols = ["r", "l", "t", "page"]
    for df in [fail1, fail2, ppass]:
        df.columns = cols

    def normalize(s):
        s = s.str.strip().str.lower()
        mapped = s.map({
            "NONE": "experience.html",
            "questions.html": "questions.html",
            "questions.html--partial": "questions.html-->partially filled",
        }).fillna("experience.html") 
        return mapped
    
    def normalize_fail1(s):
        s = s.str.strip().str.lower()
        mapped = s.map({
            "NONE": "experience.html",
            "questions.html": "questions.html-->Mixed(partial and full)",
        }).fillna("experience.html") 
        return mapped

    fail1["bucket"] = normalize_fail1(fail1["page"])
    fail2["bucket"] = normalize(fail2["page"])
    ppass["bucket"] = normalize(ppass["page"])

    buckets = ["experience.html", "questions.html-->Mixed(partial and full)", "questions.html-->partially filled", "questions.html"]

    def counts(df):
        c = df["bucket"].value_counts()
        return [c.get(b, 0) for b in buckets]

    fail1_counts = counts(fail1)
    fail2_counts = counts(fail2)
    pass_counts  = counts(ppass)

    labels = ["Fail1", "Fail2", "Pass"]
    data = [fail1_counts, fail2_counts, pass_counts]

    # Convert to stacked bars (percent)
    totals = [sum(x) for x in data]
    percents = [[(x[i] / totals[idx] if totals[idx] else 0) for i in range(len(buckets))] for idx, x in enumerate(data)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    bottom = [0]*len(labels)

    colors = OrderedDict([
        ("experience.html", "#2ca02c"),
        ("questions.html-->Mixed(partial and full)", "#d62728"),
        ("questions.html-->partially filled", "#ff7f0e"),
        ("questions.html", "#293ce0"),
    ])

    for i, bucket in enumerate(buckets):
        vals = [p[i] for p in percents]
        ax.bar(x, vals, bottom=bottom, label=bucket, alpha=0.9, color=colors[bucket])
        bottom = [bottom[j] + vals[j] for j in range(len(vals))]

    ax.set_title("ExperienceEnv – How Rewards Impact Reinforcement Learning Outcomes")
    ax.set_ylabel("Proportion of Episodes")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Pages reached")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"plot saved to: {output_path}")

# Paths
base_dir = Path(__file__).resolve().parent.parent.parent
fail1_path = base_dir / "Agent" / "logs" / "ppo" / "page2" / "fail1.csv"
fail2_path = base_dir / "Agent" / "logs" / "ppo" / "page2" / "fail2.csv"
pass_path  = base_dir / "Agent" / "logs" / "ppo" / "page2" / "pass.csv"
output_path = base_dir / "Analysis" / "Images" / "experience_outcomes.png"

plot_experience_comparison(fail1_path, fail2_path, pass_path, output_path)
