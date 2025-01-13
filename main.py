import gymnasium as gym
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage, ListStorage
import torch
from tensordict import TensorDict

# We define the maximum size of the buffer
size = 100

if __name__ == '__main__':
    env = gym.make("LunarLander-v3", continuous = True, render_mode = 'human')
    observation, info = env.reset()

    buffer = ReplayBuffer(storage = LazyTensorStorage(size))

    episode_over = False
    while not episode_over:

        action = env.action_space.sample()  # agent policy that uses the observation and info
        next_observation, reward, terminated, truncated, info = env.step(action)

        buffer.add({
            's': torch.tensor(observation), 
            'a': torch.tensor(action),
            'r': torch.tensor(reward),
            't': torch.tensor(terminated),
            's_': torch.tensor(next_observation)})

        observation = next_observation

        episode_over = terminated or truncated

    env.close()

    batch = buffer.sample(10)

    s = batch['s']
    a = batch['a']

    print(s.shape)
    print(a)
    print(a.shape)