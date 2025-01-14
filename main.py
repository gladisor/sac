from copy import deepcopy

import gymnasium as gym
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage, ListStorage
import torch
from torch import nn, Tensor

# We define the maximum size of the buffer
size = 1000

def mlp(in_size: int, h_size: int, h_layers: int, out_size: int):
    layers = [
        nn.Linear(in_size, h_size),
        nn.ReLU()
    ]

    for _ in range(h_layers):
        layers += [
            nn.Linear(h_size, h_size),
            nn.ReLU()
        ]
    
    layers.append(nn.Linear(h_size, out_size))

    return nn.Sequential(*layers)

class SAC:
    def __init__(
            self, 
            observation_size: int,
            action_size: int,
            h_size: int,
            buffer_size: int):
        
        self.policy = mlp(observation_size, h_size, 2, 2 * action_size)
        self.Q1 = mlp(observation_size + action_size, h_size, 3, 1)
        self.Q2 = mlp(observation_size + action_size, h_size, 3, 1)
        self.Q1_target = deepcopy(self.Q1)
        self.Q2_target = deepcopy(self.Q2)

        self.buffer = ReplayBuffer(storage = LazyTensorStorage(buffer_size))

    def select_action(self, observation: Tensor):
        mu, logvar = self.policy(observation).chunk(2, dim = 0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        a = mu + std * eps
        print(a)
        return
    
    def entropy(self, observation: Tensor, action: Tensor):
        mu, logvar = self.policy(observation).chunk(2, dim = 0)
        std = torch.exp(0.5 * logvar)
        entropy = 0.5 * (1 + torch.log(2 * torch.pi)) + torch.log(std)
        print(entropy)


if __name__ == '__main__':

    render_mode = None #'human'
    env = gym.make("LunarLander-v3", continuous = True, render_mode = render_mode)


    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    h_size = 128

    sac = SAC(
        observation_size = env.observation_space.shape[0],
        action_size = env.action_space.shape[0],
        h_size = 128,
        buffer_size = 1000
    )

    observation, info = env.reset()

    observation = torch.tensor(observation)
    a = sac.select_action(observation)

    print(a)


    # for episode in range(10):

    #     observation, info = env.reset()
    #     episode_over = False

    #     while not episode_over:

    #         action = env.action_space.sample()  # agent policy that uses the observation and info
    #         next_observation, reward, terminated, truncated, info = env.step(action)
    #         episode_over = terminated or truncated

    #         buffer.add({
    #             's': torch.tensor(observation), 
    #             'a': torch.tensor(action),
    #             'r': torch.tensor(reward),
    #             't': torch.tensor(episode_over),
    #             's_': torch.tensor(next_observation)})

    #         observation = next_observation

    # env.close()

    # batch = buffer.sample(64)

    # s = batch['s']
    # a = batch['a']
    # r = batch['r']
    # t = batch['t']
    # s_ = batch['s_']

    # print(len(buffer))