from typing import Union

from policy_models.mlp import ActorCriticMLP
from .neural_agent import NeuralAgent
from common import Action
from Environment import KarelEnv

import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import pickle



class ActorCritic(NeuralAgent):
    def __init__(
        self,
        policy: Union[ActorCriticMLP, str],
        env,
        GAMMA=0.99,
        learning_rate=0.0003,
        max_eps_len=100,
        max_episodes=100000,
    ):
        super().__init__(
            policy,
            env,
            GAMMA=GAMMA,
            learning_rate=learning_rate,
            max_eps_len=max_eps_len,
            max_episodes=max_episodes,
        )

    def compute_actor_loss(self, data):
        s, a, r, probs = data["S"], data["A"], data["R"], data["PI"]
        q = self.policy.Qs[0](s, a)
        probs = torch.stack(probs)
        #probs = self.policy(s, batch_mode=True)
        log_pi = torch.log(probs + 1e-13)
        Gs = self.compute_returns(r)
        #q2_pi = ac.q2(s, a)
        #q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        alpha = 0.5
        loss_pi = alpha *( -log_pi*(Gs - q)).mean()

        return loss_pi
    
    def compute_critic_loss(self, data):
        s, a, r, s_next, d = data['S'], data['A'], data['R'], data['S_next'], data['D']

        q = self.policy.Qs[0](s,a)
        alpha = 0.5
        Gs = self.compute_returns(r)
        

        # MSE loss against Bellman backup
        loss_q = alpha*((Gs - q).pow(2)).mean()

        return loss_q

    def compute_returns(self, rewards):
        G = 0.0
        Gs = np.zeros_like(rewards, dtype=float)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.GAMMA * G
            Gs[t] = G
        return torch.FloatTensor(Gs)

    def update_agent(self):
        actor_loss = self.compute_actor_loss(self.epidata)
        critic_loss = self.compute_critic_loss(self.epidata)

        ac_loss = actor_loss + critic_loss  # + 0.001 * self.entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()
        #print('well well')
