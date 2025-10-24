
# 🧠 Doodle-DRL: Automated Testing via Deep Reinforcement Learning

A **Doodle Jump-style environment** built using **PyGame + Gymnasium**, trained with **Stable-Baselines3 (PPO & A2C)**.  
This project demonstrates automated testing of an interactive application through **Deep Reinforcement Learning (DRL)** agents.

The experiments compare **two algorithms (PPO, A2C)** using the **same persona (Survivor)** across **multiple random seeds** to ensure reproducibility and statistical robustness — fulfilling the requirement of **“3+ experiments per app.”**

---



## 📁 Project Structure

```text
Doodle-DRL/
├─ envs/
│   └─ doodle_jump_env.py         # Custom Gymnasium      environment (coins, enemies, pellets)
├─ src/
│   ├─ train.py                  # Train PPO/A2C models
│   ├─ eval.py                   # Evaluate trained models → metrics CSVs
│   ├─ visualize.py              # Record gameplay GIF/MP4s to notebooks/
│   └─ plot_result.py            # Plot learning/eval results → notebooks/
├─ configs/
│   └─ personas.yaml             # Reward persona configurations
├─ models/                       # Trained model .zip files
├─ logs/                         # Training logs, monitor CSVs, TensorBoard data
├─ notebooks/                    # Recorded videos and generated plots
└─ requirements.txt
```

---

## ⚙️ 1. Setup Instructions

### Create Virtual Environment & Install Dependencies

#### 🪟 Windows PowerShell
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 💻 macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


---

## 🧩 2. Training Experiments (2 Algorithms × 1 Persona × Multiple Seeds)

We train PPO and A2C using the Survivor persona with two different random seeds (7 and 21)
for a total of 4 experiments (≥3 required).

```powershell
# PPO + Survivor (seed 7)
python src\train.py --algo ppo --persona survivor --steps 500000 --seed 7  --tag algo_comp_s7

# A2C + Survivor (seed 7)
python src\train.py --algo a2c --persona survivor --steps 500000 --seed 7  --tag algo_comp_s7

# PPO + Survivor (seed 21)
python src\train.py --algo ppo --persona survivor --steps 500000 --seed 21 --tag algo_comp_s21

# A2C + Survivor (seed 21)
python src\train.py --algo a2c --persona survivor --steps 500000 --seed 21 --tag algo_comp_s21
```

✅ Outputs:
```bash
models/ppo_survivor_algo_comp_s7_final.zip
models/a2c_survivor_algo_comp_s7_final.zip
models/ppo_survivor_algo_comp_s21_final.zip
models/a2c_survivor_algo_comp_s21_final.zip
```

---

## 🧪 3. Evaluation (Performance Metrics → logs/)

Each model is evaluated for 20 episodes to calculate average return, best height, and crash rate.

```powershell
python src\eval.py --model_path models\ppo_survivor_algo_comp_s7_final.zip  --persona survivor --episodes 20 --out_csv logs\eval_survivor_ppo_s7.csv
python src\eval.py --model_path models\a2c_survivor_algo_comp_s7_final.zip  --persona survivor --episodes 20 --out_csv logs\eval_survivor_a2c_s7.csv
python src\eval.py --model_path models\ppo_survivor_algo_comp_s21_final.zip --persona survivor --episodes 20 --out_csv logs\eval_survivor_ppo_s21.csv
python src\eval.py --model_path models\a2c_survivor_algo_comp_s21_final.zip --persona survivor --episodes 20 --out_csv logs\eval_survivor_a2c_s21.csv
```

✅ Resulting files:
```bash
logs/eval_survivor_ppo_s7.csv
logs/eval_survivor_a2c_s7.csv
logs/eval_survivor_ppo_s21.csv
logs/eval_survivor_a2c_s21.csv
```

---

## 🎮 4. Visualization (Record Gameplay → notebooks/)

Each command below records a short GIF of the trained agent playing the game.

```powershell
# PPO & A2C (seed 7)
python src\visualize.py --model_path models\ppo_survivor_algo_comp_s7_final.zip  --persona survivor
python src\visualize.py --model_path models\a2c_survivor_algo_comp_s7_final.zip  --persona survivor

# PPO & A2C (seed 21)
python src\visualize.py --model_path models\ppo_survivor_algo_comp_s21_final.zip --persona survivor
python src\visualize.py --model_path models\a2c_survivor_algo_comp_s21_final.zip --persona survivor

```

---

## 📈 5. Generate Plots (All → notebooks/)

Generate learning curves, return distributions, crash rates, and coverage plots
for all experiments using a single command:

```powershell
python src\plot_result.py --monitors logs\ppo_survivor_algo_comp_s7_monitor.csv logs\a2c_survivor_algo_comp_s7_monitor.csv logs\ppo_survivor_algo_comp_s21_monitor.csv logs\a2c_survivor_algo_comp_s21_monitor.csv --evals logs\eval_survivor_ppo_s7.csv logs\eval_survivor_a2c_s7.csv logs\eval_survivor_ppo_s21.csv logs\eval_survivor_a2c_s21.csv --labels "PPO-s7" "A2C-s7" "PPO-s21" "A2C-s21"
```

✅ Generated plots:
```bash
notebooks/learning_curves.png
notebooks/eval_returns.png
notebooks/eval_crash.png
notebooks/eval_coverage.png
```

---

## 📊 6. Experiment Summary

| # | Algorithm | Persona | Seed | Notes      |
|---|-----------|---------|------|------------|
| 1 | PPO       | Survivor| 7    | Baseline   |
| 2 | A2C       | Survivor| 7    | Comparison |
| 3 | PPO       | Survivor| 21   | Replicate  |
| 4 | A2C       | Survivor| 21   | Replicate  |

---

## 🧠 7. Seeds & Reproducibility

Reinforcement learning and procedural environments are inherently stochastic.
Fixing a random seed ensures that experiments are reproducible and results are comparable.

Seed = 7 → Baseline experiments

Seed = 21 → Replicates for variance analysis

Consistent persona = Survivor (controls for reward bias)

In short: same persona + same hyperparameters + different seeds → tests reliability, not luck.

---

## ⚙️ 8. Environment Specification

**Actions (Discrete-4):**

| Action | Meaning           |
|--------|-------------------|
| 0      | Move left         |
| 1      | Move right        |
| 2      | Idle (small penalty) |
| 3      | Shoot pellet upward |

**Observations (Box(13)):**

- Player: position (x, y), velocity (vx, vy)
- Two nearest platforms: relative (x, y)
- Nearest coin: relative (x, y)
- Nearest enemy: relative (x, y)
- On-platform flag (1)

**Rewards (from configs/personas.yaml):**

- Platform landing bonus
- Honest climb reward (capped)
- Coin and enemy-kill rewards
- Idle / platform-camping penalties
- Upward motion + height presence bonus
- Death penalty

**Persona Used:**

survivor — balanced, survival-oriented behavior

---

## 🧾 9. Dependencies

```shell
gymnasium==0.29.1
stable-baselines3==2.3.2
pygame==2.6.1
numpy>=1.24,<3.0
torch>=2.2.0
tensorboard>=2.14.0
pyyaml>=6.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---

## 🧪 10. Full Reproduction Checklist

Run these commands from the project root:

```powershell
# --- TRAIN (4 runs) ---
python src\train.py --algo ppo --persona survivor --steps 500000 --seed 7  --tag algo_comp_s7
python src\train.py --algo a2c --persona survivor --steps 500000 --seed 7  --tag algo_comp_s7
python src\train.py --algo ppo --persona survivor --steps 500000 --seed 21 --tag algo_comp_s21
python src\train.py --algo a2c --persona survivor --steps 500000 --seed 21 --tag algo_comp_s21

# --- EVALUATE (4 runs) ---
python src\eval.py --model_path models\ppo_survivor_algo_comp_s7_final.zip  --persona survivor --episodes 20 --out_csv logs\eval_survivor_ppo_s7.csv
python src\eval.py --model_path models\a2c_survivor_algo_comp_s7_final.zip  --persona survivor --episodes 20 --out_csv logs\eval_survivor_a2c_s7.csv
python src\eval.py --model_path models\ppo_survivor_algo_comp_s21_final.zip --persona survivor --episodes 20 --out_csv logs\eval_survivor_ppo_s21.csv
python src\eval.py --model_path models\a2c_survivor_algo_comp_s21_final.zip --persona survivor --episodes 20 --out_csv logs\eval_survivor_a2c_s21.csv

# --- VISUALIZE (Gameplay) ---
python src\visualize.py --model_path models\ppo_survivor_algo_comp_s7_final.zip  --persona survivor
python src\visualize.py --model_path models\a2c_survivor_algo_comp_s7_final.zip  --persona survivor
python src\visualize.py --model_path models\ppo_survivor_algo_comp_s21_final.zip --persona survivor
python src\visualize.py --model_path models\a2c_survivor_algo_comp_s21_final.zip --persona survivor

# --- PLOTS (Learning + Eval) ---
python src\plot_result.py --monitors logs\ppo_survivor_algo_comp_s7_monitor.csv logs\a2c_survivor_algo_comp_s7_monitor.csv logs\ppo_survivor_algo_comp_s21_monitor.csv logs\a2c_survivor_algo_comp_s21_monitor.csv --evals logs\eval_survivor_ppo_s7.csv logs\eval_survivor_a2c_s7.csv logs\eval_survivor_ppo_s21.csv logs\eval_survivor_a2c_s21.csv --labels "PPO-s7" "A2C-s7" "PPO-s21" "A2C-s21"
```

✅ Output Summary

Models → models/

Logs + CSVs → logs/

Plots → notebooks/
---

## 🎬 Example Video

<p align="center">
	<video width="480" controls>
		<source src="notebooks/Video.mp4" type="video/mp4">
		Your browser does not support the video tag. 
		<a href="notebooks/Video.mp4">Download Video.mp4</a>
	</video>
</p>

If the video does not display, you can download it here: [▶️ Video.mp4](notebooks/Video.mp4)

---

Author: Shiv Amin, Saksham Tejpal, Samir Choudary
Course: Topics in Computer Science — DRL for Automated Testing
Date: October 2025
