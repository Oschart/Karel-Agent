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


class NeuralAgent:
    def __init__(
        self,
        policy,
        env,
        GAMMA=0.99,
        learning_rate=3e-4,
        max_eps_len=100,
        max_episodes=100000,
        num_actions=6,
        learn_by_demo=True,
    ):
        self.policy = policy
        self.env = env
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.max_eps_len = max_eps_len
        self.max_episodes = max_episodes
        self.num_actions = num_actions
        self.learn_by_demo = learn_by_demo

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
            t = self.generate_ep_rollout(task_state, optimal_seq)

            self.update_agent()

            all_rewards.append(np.sum(self.epidata["R"]))
            all_lengths.append(t)
            average_lengths.append(np.mean(all_lengths[-10:]))
            if episode % 100 == 0:
                print(
                    "episode: {}, reward: {}, total length: {}, average length: {} \n".format(
                        episode, np.sum(self.epidata["R"]), t, average_lengths[-1]
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
                action_dist, _ = self.policy(state)
                dist = action_dist.detach().numpy()
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

    def reset_rollout_buffer(self):
        self.epidata = {"S": [], "V": [], "A": [], "R": [], "G": [], "PI": [], "D": []}

    def update_rollout_buffer(self, state, state_value, action, reward, act_prob, done):
        self.epidata["S"].append(state)
        self.epidata["V"].append(state_value)
        self.epidata["A"].append(action)
        self.epidata["R"].append(reward)
        self.epidata["PI"].append(act_prob)
        self.epidata["D"].append(done)

    def wrap_up_episode(self):
        self.epidata["G"] = self.compute_returns(self.epidata["R"], self.epidata["S"])
        self.epidata["V"] = torch.FloatTensor(self.epidata["V"])


    def generate_ep_rollout(self, state, optimal_seq):
        self.reset_rollout_buffer()
        for t in range(self.max_eps_len):
            action_dist, state_value = self.policy(state)
            dist = action_dist.detach().numpy()

            if self.learn_by_demo:
                # Use expert optimal action
                action = Action.from_str[optimal_seq[t]]
            else:
                # Use own agent's action
                action = np.random.choice(self.num_actions, p=np.squeeze(dist))

            act_prob = action_dist.squeeze(0)[action]
            new_state, reward, done, _ = self.env.step(action)

            self.update_rollout_buffer(state, state_value, action, reward, act_prob, done)
            state = new_state

            if done:
                break
        self.wrap_up_episode()
        return t

    def compute_returns(self, rewards, states):
        G = 0.0
        Gs = np.zeros_like(rewards, dtype=float)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.GAMMA * G
            Gs[t] = G
        return torch.FloatTensor(Gs)

    def update_agent(self):
        raise NotImplementedError()
