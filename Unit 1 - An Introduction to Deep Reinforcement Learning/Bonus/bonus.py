#%%
import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

#%%
gym_name = "MountainCar-v0"
model_name = f"OverFitted-{gym_name}-MlpPolicy"


#%%
env = make_vec_env(f"{gym_name}", n_envs=16)
observation = env.reset()

print(f"Observation Space Shape {env.observation_space.shape}")
print(f"Sample observation {env.observation_space.sample()}")
print(f"Action Space Shape {env.action_space.n}")
print(f"Action Space Sample {env.action_space.sample()}")

#%%
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1536,
    batch_size=64,
    n_epochs=16,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)

model.learn(total_timesteps=int(3e6))
model.save(model_name)

#%%
eval_env = gym.make(f"{gym_name}")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"""
mean reward: {mean_reward:.2f}
std reward: {std_reward:.2f}
""")
