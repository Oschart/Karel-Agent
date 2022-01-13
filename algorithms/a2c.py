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
        probs, G, v = data["PI"], data["G"], data["V"]
        probs = torch.stack(probs)
        log_pi = torch.log(probs + 1e-13)

        # Entropy-regularized policy loss
        alpha = 0.5
        loss_pi = alpha * (-log_pi*(G - v)).mean()

        return loss_pi

    def compute_critic_loss(self, data):
        G, v = data['G'], data['V']

        alpha = 0.5
        # MSE loss against Bellman backup
        loss_q = alpha*((G - v).pow(2)).mean()

        return loss_q

    def update_agent(self):
        actor_loss = self.compute_actor_loss(self.epidata)
        critic_loss = self.compute_critic_loss(self.epidata)

        ac_loss = actor_loss + critic_loss  # + 0.001 * self.entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()
