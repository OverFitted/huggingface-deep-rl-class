# An Introduction to Deep Reinforcement Learning

## [Module github](https://github.com/huggingface/deep-rl-class/tree/main/unit1#unit-1-introduction-to-deep-reinforcement-learning)

---

## Model data

library_name: stable-baselines3

tags:

- LunarLander-v2
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3

model-index:

- name: PPO
- results:
  - metrics:
    - type: mean_reward
    - value: 286.33 +/- 8.54
    - name: mean_reward
  - task:
    - type: reinforcement-learning
    - name: reinforcement-learning
  - dataset:
    - name: LunarLander-v2
    - type: LunarLander-v2

---

## **PPO** Agent playing **LunarLander-v2**

  This is a trained model of a **PPO** agent playing **LunarLander-v2** using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).

## Usage (with Stable-baselines3)

```python
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('LunarLander-v2', n_envs=16)
model = PPO.load("OverFitted-MlpPolicy-test")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

## Preview

<figure class="video_container">
    <video controls="true" allowfullscreen="true" poster="media/replay.png">
        <source src="media/replay.mp4" type="video/mp4">
    </video>
</figure>
