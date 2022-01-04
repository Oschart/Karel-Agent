from parse_utils import vectorize_obs
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


class PolicyGradientAgent:
    def __init__(
        self,
        policy,
        env,
        GAMMA=0.99,
        learning_rate=3e-4,
        max_eps_len=100,
        max_episodes=100000,
        num_actions=6,
    ):
        self.policy = policy
        self.env = env
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.max_eps_len = max_eps_len
        self.max_episodes = max_episodes
        self.num_actions = num_actions

    def train(self, tasks, expert_outputs=None):

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        all_lengths = []
        average_lengths = []
        all_rewards = []
        self.entropy_term = 0

        for episode in range(self.max_episodes):

            task_state = tasks[episode % len(tasks)]
            optimal_seq = expert_outputs[episode % len(expert_outputs)]["sequence"]
            task_state = self.env.reset(task_state)

            self.policy.train()
            t = self.process_episode(task_state, optimal_seq)
            self.update_policy()

            all_rewards.append(np.sum(self.rewards))
            all_lengths.append(t)
            average_lengths.append(np.mean(all_lengths[-10:]))
            if episode % 100 == 0:
                print(
                    "episode: {}, reward: {}, total length: {}, average length: {} \n".format(
                        episode, np.sum(self.rewards), t, average_lengths[-1]
                    )
                )

        return self.policy

    def solve(self, tasks, opt_seqs, H=100):
        solved = 0
        extra_steps = 0
        for i in range(len(tasks)):
            task_state = tasks[i]
            opt_seq = opt_seqs[i]["sequence"]

            state = self.env.reset(task_state)
            for t in range(H):
                _, policy_dist = self.policy(state)
                dist = policy_dist.detach().numpy()
                action = dist.argmax()

                state, reward, done, _ = self.env.step(action)
                if done:
                    break

            solved += reward > 0
            extra_steps += t - len(opt_seq) + 1 if reward > 0 else 0

        accr = solved / len(tasks)
        avg_extra_steps = extra_steps / len(tasks)

        print(
            f"Attempted {len(tasks)} tasks, correctly solved {solved}. Accuracy={accr*100:.2f}%, avg extra steps={avg_extra_steps:.2f}"
        )

    def process_episode(self, state, optimal_seq):
        raise NotImplementedError()

    def update_policy(self):
        raise NotImplementedError()
