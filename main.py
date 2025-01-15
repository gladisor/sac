from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Box

from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
import torch
from torch import nn, Tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_mlp(in_size: int, h_size: int, h_layers: int, out_size: int):
    '''
    Simple function which builds a multi-layered-perceptron neural network with an arbitrary number of hidden layers with a fixed size.
    '''
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

from torch.distributions import Normal, MultivariateNormal

class SAC(nn.Module):
    def __init__(
            self, 
            observation_size: int,
            action_space: Box,
            h_size: int,
            buffer_size: int,
            gamma: float = 0.99,
            alpha: float = 0.2,
            tau: float = 0.005,
            batch_size: int = 64,
        ):

        super().__init__()

        ## extracting information from action_space
        action_size = action_space.shape[0]
        self.register_buffer('low', torch.tensor(action_space.low))
        self.register_buffer('high', torch.tensor(action_space.high))

        ## sac hyperparams
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        ## build networks
        self.policy = build_mlp(observation_size, h_size, 2, 2 * action_size)
        self.Q1 = build_mlp(observation_size + action_size, h_size, 3, 1)
        self.Q2 = build_mlp(observation_size + action_size, h_size, 3, 1)
        self.Q1_target = deepcopy(self.Q1)
        self.Q2_target = deepcopy(self.Q2)

        ## set up optimizers
        self.policy_opt = torch.optim.Adam(self.policy.parameters())
        self.Q1_opt = torch.optim.Adam(self.Q1.parameters())
        self.Q2_opt = torch.optim.Adam(self.Q2.parameters())

        ## initialize buffer with capacity
        self.buffer = ReplayBuffer(storage = LazyTensorStorage(buffer_size))

    @property
    def device(self):
        return self.low.device

    def select_action(self, s: Tensor, prob: bool = False):
        '''
        Takes in a single observation or a batch and selects action(s)
        '''
        mu, logvar = self.policy(s).chunk(2, dim = -1)
        logvar = torch.clamp(logvar, min=-20, max=2) ## prevents over and underflow 

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        u = mu + std * eps

        delta = (self.high - self.low)
        a = (torch.tanh(u) + 1.0) / 2.0 * delta + self.low

        if prob:

            dist = Normal(mu, std)
            ## probability under a normal distribution with diagonal covariance of the unbounded u
            log_prob_u = dist.log_prob(u).sum(dim = -1, keepdims = True)
            '''
            Need to correct the tanh squishing operation with the change of variables trick:

            pi is our policy distribution and N is our unbounded normal distribution.

            pi(a|s) = N(u|s) * |det(da/du)|^-1

            Apply log to both sides:
            ln(pi(a|s)) = ln(N(u|s)) - ln(|det(da/du)|)
            '''
            # log_det_jac = torch.sum( torch.log( torch.abs( (1.0 - torch.tanh(u).pow(2)) * delta / 2.0 ) ), dim = -1, keepdim = True)

            tanh_derivative = 1.0 - torch.tanh(u).pow(2)
            tanh_derivative = torch.clamp(tanh_derivative, min = 1e-6)
            log_det_jac = torch.sum(torch.log(tanh_derivative * delta / 2.0), dim=-1, keepdim=True)

            log_prob = log_prob_u - log_det_jac
            return a, log_prob
        else:
            return a

    def update(self):

        batch = self.buffer.sample(self.batch_size)

        s   = batch['s'].to(self.device)
        a   = batch['a'].to(self.device)
        r   = batch['r'].to(self.device)
        t   = batch['t'].to(self.device)
        s_  = batch['s_'].to(self.device)

        with torch.no_grad():
            ## select an action in the next state
            a_, log_prob = self.select_action(s_, prob = True)

            ## compute the value of this action using the least optimistic critic
            s_a_ = torch.cat([s_, a_], dim = 1)
            next_q = torch.minimum(self.Q1_target(s_a_), self.Q2_target(s_a_))

            ## target should be the current reward plus the discounted soft q estimate (future value and policy entropy)
            y = r + (~t) * self.gamma * (next_q - self.alpha * log_prob)

        ## state action pair from data
        sa = torch.cat([s, a], dim = 1)

        ## update state-action value functions
        q1_loss = torch.mean(torch.square(self.Q1(sa) - y))
        self.Q1_opt.zero_grad()
        q1_loss.backward()
        self.Q1_opt.step()

        q2_loss = torch.mean(torch.square(self.Q2(sa) - y))
        self.Q2_opt.zero_grad()
        q2_loss.backward()
        self.Q2_opt.step()

        a, log_prob = self.select_action(s, prob = True)

        ## state action pair with fresh action sampled in the update loop
        sa = torch.cat([s, a], dim = 1)
        q_value = torch.minimum(self.Q1(sa), self.Q2(sa))
        policy_loss = (self.alpha * log_prob - q_value).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # Soft update target networks
        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'q1_loss': q1_loss.item(), 'q2_loss': q2_loss.item(), 'policy_loss': policy_loss.item(), 'a_min': a.min().item(), 'a_max': a.max().item(), 'a_mean': a.mean().item()}

if __name__ == '__main__':

    device = torch.device(0)

    # render_mode = 'human'
    render_mode = None
    env = gym.make('LunarLander-v3', continuous = True, render_mode = render_mode)
    # env = gym.make('Pendulum-v1', render_mode = render_mode)
    # env = gym.make('HalfCheetah-v5', render_mode = render_mode)

    state_size = env.observation_space.shape[0]
    h_size = 128

    sac = SAC(
        observation_size = env.observation_space.shape[0],
        action_space = env.action_space,
        h_size = 128,
        buffer_size = 10000
    ).to(device)



    # observation, info = env.reset()
    # observation = torch.tensor(observation, device = device, dtype = torch.float32)
    # print(observation)
    # sac.select_action(observation)

    hist = []
    reward_hist = []

    for episode in range(100):

        observation, info = env.reset()
        episode_over = False
        episode_reward = 0.0

        while not episode_over:

            observation = torch.tensor(observation, dtype = torch.float32)

            with torch.no_grad():
                action = sac.select_action(
                    observation.to(device)
                ).cpu().numpy()

            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_over = terminated or truncated
            episode_reward += reward

            sac.buffer.add({
                's': observation,
                'a': torch.tensor(action),
                'r': torch.tensor([reward]),
                't': torch.tensor([episode_over]),
                's_': torch.tensor(next_observation, dtype = torch.float32)})

            observation = next_observation

            if len(sac.buffer) > sac.batch_size * 2:
                info = sac.update()
                hist.append(info)
            
        print(f'Episode: {episode}, Total Reward: {episode_reward}')
        reward_hist.append(episode_reward)

    hist = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 4, figsize = (5, 15))
    ax[0].set_title('Log Critic Loss')
    ax[0].plot(np.log(hist['q1_loss']))
    ax[0].plot(np.log(hist['q2_loss']))

    ax[1].set_title('Reward Per Episode')
    ax[1].plot(reward_hist)

    ax[2].set_title('Policy Loss')
    ax[2].plot(hist['policy_loss'])
    
    ax[3].set_title('Action Stats')
    ax[3].plot(hist['a_min'])
    ax[3].plot(hist['a_mean'])
    ax[3].plot(hist['a_max'])
    plt.show()




    env = gym.make('LunarLander-v3', continuous = True, render_mode = 'human')
    # env = gym.make('Pendulum-v1', render_mode = 'human')
    # env = gym.make('HalfCheetah-v5', render_mode = 'human')

    for episode in range(3):
        observation, info = env.reset()
        episode_over = False
        episode_reward = 0.0

        while not episode_over:

            observation = torch.tensor(observation, dtype = torch.float32)

            with torch.no_grad():
                action = sac.select_action(
                    observation.to(device)
                ).cpu().numpy()

            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_over = terminated or truncated
            episode_reward += reward
            observation = next_observation
            
        print(f'Episode: {episode}, Total Reward: {episode_reward}')