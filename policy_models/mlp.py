import numpy as np
import scipy.signal
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLP_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        self.pi = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation,
            output_activation=nn.Softmax,
        )
    
    def forward(self, obs, batch_mode=False):
        obs = torch.from_numpy(np.array(obs)).float()
        obs = Variable(obs)
        obs = obs.view(obs.shape[0], -1)

        return self.pi(obs)

class Q_Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        obs = Variable(torch.from_numpy(np.array(obs)).float())
        obs = obs.view(obs.shape[0], -1)
        act = Variable(torch.from_numpy(np.array(act)))
        act = F.one_hot(act.to(torch.int64), self.act_dim).float()
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class V_Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        obs = Variable(torch.from_numpy(np.array(obs)).float())
        obs = obs.view(obs.shape[0], -1)
        v = self.v(obs)
        return torch.squeeze(v, -1)


class ActorCriticMLP(nn.Module):
    def __init__(
        self, obs_dim, act_dim, soft_critic=False, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__()

        # build policy and value functions
        self.PI = MLP_Actor(obs_dim, act_dim,hidden_sizes, activation)
        
        self.soft_critic = soft_critic
        if soft_critic:
            self.Q1 = Q_Critic(obs_dim, act_dim, hidden_sizes, activation)
            self.Q2 = Q_Critic(obs_dim, act_dim, hidden_sizes, activation)
        else:
            self.V = V_Critic(obs_dim, hidden_sizes, activation)


    def forward(self, obs, batch_mode=False):
        obs = torch.from_numpy(np.array(obs)).float()
        if not batch_mode:
            obs = obs.unsqueeze(0)
        obs = Variable(obs)
        obs = obs.view(obs.shape[0], -1)

        if self.soft_critic:
            return self.PI(obs), 0

        return self.PI(obs), self.V(obs)

