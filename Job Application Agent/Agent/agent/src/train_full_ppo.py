# file: train_full.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from agent.envs.webflow_env import WebFlowEnv  # unified env

# ---------- Paths ----------
LOG_DIR = "Temp/logs/ppo/full_webflow"
MODEL_DIR = "Temp/models/ppo_full"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env():
    def _init():
        base_env = WebFlowEnv()
        monitored_env = Monitor(
            base_env,
            filename=os.path.join(LOG_DIR, "full_metrics.csv"),
            info_keywords=("result", "match_ratio", "page")
        )
        return monitored_env
    return _init


if __name__ == "__main__":
    # ---------- Environment ----------
    env = DummyVecEnv([make_env()])

    # ---------- Model ----------
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        device="cpu"
    )

    # ---------- Save model every 5 000 steps ----------
    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,                     # every 5 000 timesteps
        save_path=MODEL_DIR,
        name_prefix="ppo_webflow",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ---------- Train ----------
    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_callback,
    )

    # ---------- Final save ----------
    model.save(os.path.join(MODEL_DIR, "/ppo_webflow_final"))
    env.close()
