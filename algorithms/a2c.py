from typing import Union

from policy_models.mlp import ActorCriticMLP
from .neural_agent import NeuralAgent
from common import Action
from environment import KarelEnv

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
    name = 'actor_critic'
    def __init__(
        self,
        policy: Union[ActorCriticMLP, str],
        env,
        GAMMA=0.99,
        learning_rate=3e-4,
        alpha=0.5,
        max_episodes=100000,
        max_eps_len=100,
        num_actions=6,
        learn_by_demo=True,
        early_stop=None,
        variant_name='v0',
        load_pretrained=False
    ):
        super().__init__(
            policy,
            env,
            GAMMA=GAMMA,
            learning_rate=learning_rate,
            alpha=alpha,
            max_episodes=max_episodes,
            max_eps_len=max_eps_len,
            num_actions=num_actions,
            learn_by_demo=learn_by_demo,
            early_stop=early_stop,
            variant_name=variant_name,
            load_pretrained=load_pretrained
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)


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
