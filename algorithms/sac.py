import itertools
import pickle
import sys
from copy import deepcopy
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common import Action
from environment import KarelEnv
from policy_models.mlp import ActorCriticMLP
from torch.autograd import Variable

from .neural_agent import NeuralAgent


class SoftActorCritic(NeuralAgent):
    name = 'sac'
    def __init__(
        self,
        policy: Union[ActorCriticMLP, str],
        env,
        GAMMA=0.99,
        learning_rate=3e-4,
        alpha=0.5,
        clip_range=None,
        max_episodes=100000,
        max_eps_len=100,
        num_actions=6,
        learn_by_demo=True,
        early_stop=30,
        variant_name='v0',
        load_pretrained=False,
        verbose=False
    ):
        super().__init__(
            policy,
            env,
            GAMMA=GAMMA,
            learning_rate=learning_rate,
            alpha=alpha,
            clip_range=clip_range,
            max_episodes=max_episodes,
            max_eps_len=max_eps_len,
            num_actions=num_actions,
            learn_by_demo=learn_by_demo,
            early_stop=early_stop,
            variant_name=variant_name,
            load_pretrained=load_pretrained,
            verbose=verbose
        )
        self.policy_targ = deepcopy(policy)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.policy_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(policy.Q1.parameters(), policy.Q2.parameters())
        self.pi_optimizer = optim.Adam(policy.PI.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_params, lr=learning_rate)

        


    def compute_actor_loss(self, data):
        s, a, probs = data['S'], data['A'], data["PI"]
        probs = torch.stack(probs)
        log_pi = -torch.log(probs + 1e-6)

        q1_pi = self.policy.Q1(s,a)
        q2_pi = self.policy.Q2(s,a)

        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * log_pi - q_pi).mean()

        return loss_pi

    def compute_critic_loss(self, data):
        s, a, r, s_next,  d = data['S'], data['A'], data['R'], data['S_next'], data['D']

        q1 = self.policy.Q1(s,a)
        q2 = self.policy.Q2(s,a)

        alpha = 0.5
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            action_dist = self.policy.PI(s_next, batch_mode=True)

            dist_stats = action_dist.max(dim=1)
            a2 = dist_stats.indices
            a2_prob = dist_stats.values
            logp_a2 =  torch.log(a2_prob + 1e-13)

            # Target Q-values
            q1_pi_targ = self.policy_targ.Q1(s_next, a2)
            q2_pi_targ = self.policy_targ.Q2(s_next, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.GAMMA * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        return loss_q

    def update_agent(self):
        critic_loss = self.compute_critic_loss(self.epidata)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False
        
        actor_loss = self.compute_actor_loss(self.epidata)
        
        self.pi_optimizer.zero_grad()
        actor_loss.backward()
        self.pi_optimizer.step()

        # Finally, update target networks by polyak averaging.
        polyak=0.995
        with torch.no_grad():
            for p, p_targ in zip(self.policy.parameters(), self.policy_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        
        return actor_loss.item()
    
    def act(self, state, return_dist=False):
        action_dist, _ = self.policy(state)
        action = action_dist.argmax()
        if return_dist:
            return action.item(), action_dist.squeeze()
        else:
            return action.item()
    
    def judge(self, state, action=None):
        q1 = self.policy.Q1(state,action)
        q2 = self.policy.Q2(state,action)
        q_min = torch.min(q1, q2)
        return q_min.item()
