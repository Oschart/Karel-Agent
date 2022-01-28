from parse_utils import vectorize_obs
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


class NeuralAgent:
    def __init__(
        self,
        policy,
        env,
        GAMMA=0.99,
        learning_rate=3e-4,
        alpha=0.5,
        clip_range=(-1, 1),
        max_episodes=100000,
        max_eps_len=30,
        num_actions=6,
        learn_by_demo=True,
        early_stop=30,
        variant_name='v0',
        load_pretrained=False,
        verbose=False
    ):
        self.policy = policy
        self.env = env
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.clip_range = clip_range
        self.max_episodes = max_episodes
        self.max_eps_len = max_eps_len
        self.num_actions = num_actions
        self.learn_by_demo = learn_by_demo
        self.early_stop = early_stop
        self.variant_name = variant_name
        self.verbose = verbose

        if load_pretrained:
            self.load_policy()

    def train(self, tasks, expert_traj=None, data_val=None):
        print("=============================================================")
        print(f"Started training {self.name} (variant: {self.variant_name})")
        print("=============================================================")
        all_lengths = []
        average_lengths = []
        all_rewards = []
        self.entropy_term = 0
        best_accr = 0.0
        best_avg_extra_steps = 1e10
        worse_count = 0

        stats = {'loss': [], 'accr': []}

        for episode in range(self.max_episodes):

            task_state = tasks[episode % len(tasks)]
            if expert_traj:
                optimal_seq = expert_traj[episode % len(expert_traj)]["sequence"]
            else:
                optimal_seq = None
            task_state = self.env.reset(task_state)

            self.policy.train()
            t = self.generate_ep_rollout(task_state, optimal_seq)

            loss = self.update_agent()

            stats['loss'].append(loss)

            #all_rewards.append(np.sum(self.epidata["R"]))#
            all_lengths.append(t)
            average_lengths.append(np.mean(all_lengths[-10:]))
            if episode % 500 == 0:
                accr, avg_extra_steps = self.evaluate(data_val[0], data_val[1])
                stats['accr'].append(accr)
                if accr > best_accr or (accr == best_accr and avg_extra_steps < best_avg_extra_steps):
                    best_accr = accr
                    best_avg_extra_steps = avg_extra_steps
                    worse_count = 0
                    self.save_policy()
                else:
                    worse_count += 1

                if self.verbose:
                    print(f"Episode {episode}: validation accuracy={accr*100:.2f}%, \t avg_extra_steps={avg_extra_steps:.2f}")
                    if self.early_stop and worse_count >= self.early_stop:
                        print(
                            f"Early stopping triggered; validation accuracy hasn't increased for {worse_count} epsiodes!")
                        break

        print("=============================================================")
        print(f"Finished training {self.name} (variant: {self.variant_name})")
        print("=============================================================")
        self.load_policy()
        return stats

    def save_policy(self):
        save_path = f"pretrained/{self.name}_{self.variant_name}.pth"
        torch.save(self.policy.state_dict(), save_path)

    def load_policy(self):
        load_path = f"pretrained/{self.name}_{self.variant_name}.pth"
        self.policy.load_state_dict(torch.load(load_path))

    def evaluate(self, tasks, opt_seqs, H=50, verbose=False):
        solved = 0
        solved_opt = 0
        extra_steps = 0
        self.policy.eval()
        for i in range(len(tasks)):
            task_state = tasks[i]
            opt_seq = opt_seqs[i]["sequence"]

            state = self.env.reset(task_state)
            for t in range(H):
                action = self.act(state)

                state, reward, done, _ = self.env.step(action)
                if done:
                    break

            solved += reward > 0
            solved_opt += (reward > 0 and t + 1 == len(opt_seq))
            extra_steps += t - len(opt_seq) + 1 if reward > 0 else 0

        accr = solved / len(tasks)
        opt_accr = solved_opt/len(tasks)
        avg_extra_steps = extra_steps / len(tasks)

        if verbose:
            print(f"Attempted {len(tasks)} tasks, correctly solved {solved}. Accuracy(solved)={accr*100:.2f}%, Accuracy(optimally solved)={opt_accr*100:.2f}%, avg extra steps={avg_extra_steps:.2f}")
        return accr, avg_extra_steps

    def reset_rollout_buffer(self):
        self.epidata = {"S": [], "V": [], "A": [], "R": [],
                        "S_next": [], "G": [], "PI": [], "D": []}

    def update_rollout_buffer(self, state, state_value, action, reward, state_next, act_prob, done):
        self.epidata["S"].append(state)
        self.epidata["V"].append(state_value)
        self.epidata["A"].append(action)
        self.epidata["R"].append(reward)
        self.epidata["S_next"].append(state_next)
        self.epidata["PI"].append(act_prob)
        self.epidata["D"].append(done)

    def wrap_up_episode(self):
        self.epidata["S"] = torch.from_numpy(np.array(self.epidata["S"]))
        self.epidata["S_next"] = torch.from_numpy(
            np.array(self.epidata["S_next"]))

        self.epidata["G"] = self.compute_returns(
            self.epidata["R"], self.epidata["S"])
        self.epidata["R"] = torch.FloatTensor(self.epidata["R"])

        if None not in self.epidata["V"]:
            self.epidata["V"] = torch.cat(self.epidata["V"])
        self.epidata["D"] = torch.IntTensor(self.epidata["D"])

    def generate_ep_rollout(self, state, optimal_seq=None):
        self.reset_rollout_buffer()
        for t in range(self.max_eps_len):
            action_dist, state_value = self.policy(state)
            #dist = action_dist.cpu().detach().numpy()

            if self.learn_by_demo and optimal_seq:
                # Use expert optimal action
                action = Action.from_str[optimal_seq[t]]
            else:
                # Use own agent's action
                dist = action_dist.cpu().detach().numpy()
                action = np.random.choice(self.num_actions, p=np.squeeze(dist))

            act_prob = action_dist[0][action]
            new_state, reward, done, _ = self.env.step(action)

            self.update_rollout_buffer(
                state, state_value, action, reward, new_state, act_prob, done)
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

    def act(self, state):
        raise NotImplementedError()

    def judge(self, state, action=None):
        raise NotImplementedError()
