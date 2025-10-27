from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from agent.envs.review_env import ReviewEnv
from agent.envs.index_env import IndexEnv
from agent.envs.experience_env import ExperienceEnv
from agent.envs.questions_env import QuestionsEnv
from Agent.agent.handler.data_loader import ApplicantManager
# from agent.envs.webform_env import WebAppEnv

# Initialize environment
env = ReviewEnv()

env = Monitor(env, filename="./logs/ppo/", info_keywords=("page",))
# Initialize model
model = PPO(
    policy="MlpPolicy",   # multilayer perceptron
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=10,
    batch_size=64,
    gamma=0.99,
    device="cpu"
)

# Train
model.learn(total_timesteps=100)

# Save
model.save("models/ppo_webapp_review")
env.close()
