import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from agent.envs.webflow_env import WebFlowEnv

LOG_DIR = "logs/a2c/full_webflow"
MODEL_DIR = "models/a2c_full"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env():
    def _init():
        base_env = WebFlowEnv()
        monitored_env = Monitor(
            base_env,
            filename=os.path.join(LOG_DIR, "a2c_full.csv"),
            info_keywords=("result", "match_ratio", "page")
        )
        return monitored_env
    return _init

env = DummyVecEnv([make_env()])

model = A2C(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=7e-4,
    n_steps=512,
    gamma=0.99,
    ent_coef=0.05,
    vf_coef=0.25,
    max_grad_norm=0.5,
    device="cpu"
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=MODEL_DIR,
    name_prefix="a2c_webflow",
)

model.learn(total_timesteps=100_000, callback=checkpoint_callback)
model.save(os.path.join(MODEL_DIR, "a2c_webflow_final"))
env.close()
