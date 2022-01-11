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



class SoftActorCritic(NeuralAgent):
    def __init__(
        self,
        policy,
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

    def process_episode(self, state, optimal_seq):
        self.log_probs = []
        self.critic_values = []
        self.rewards = []
        for t in range(self.max_eps_len):
            value, policy_dist = self.policy(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(self.num_actions, p=np.squeeze(dist))
            optimal_action = Action.from_str[optimal_seq[t]]
            log_prob = torch.log(policy_dist.squeeze(0)[optimal_action] + 1e-10)
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = self.env.step(optimal_action)

            self.rewards.append(reward)
            self.critic_values.append(value)
            self.log_probs.append(log_prob)
            self.entropy_term += entropy
            state = new_state

            if done:
                break
        return t

    def update_policy(self):
        # compute Q values
        Qval = 0.0
        Qvals = np.zeros_like(self.critic_values)
        for t in reversed(range(len(self.rewards))):
            Qval = self.rewards[t] + self.GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        critic_values = torch.FloatTensor(self.critic_values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(self.log_probs)

        advantage = Qvals - critic_values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss  # + 0.001 * self.entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()
