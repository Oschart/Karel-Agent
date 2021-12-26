from ActorCritic import ActorCritic
from parse_utils import featurize_task
from common import Action
from Environment import Environment

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

class PolicyGradientAgent():
    def __init__(self, level='', pretrained=False):
        self.level = level
        if pretrained:
            self.actor_critic = pickle.load(open(f'pretrained/actor_critic_{self.level}.pkl', "rb"))

    def train(self, tasks, seqs, env=None, hidden_size=256, learning_rate=3e-4, GAMMA =0.99, num_steps=100, max_episodes=100000):
        num_inputs = 11*16
        num_outputs = 6

        env = Environment()
        
        actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
        ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for episode in range(max_episodes):
            log_probs = []
            values = []
            rewards = []

            task = tasks[episode%len(tasks)]
            optimal_seq = seqs[episode%len(seqs)]['sequence']
            state = task
            env.init(state)
            
            for t in range(len(optimal_seq)):
                state_vect = featurize_task(state)
                value, policy_dist = actor_critic.forward(state_vect)
                value = value.detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 

                action = np.random.choice(num_outputs, p=np.squeeze(dist))
                optimal_action = Action.from_str[optimal_seq[t]]
                log_prob = torch.log(policy_dist.squeeze(0)[optimal_action] + 1e-10)
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward = env.step(optimal_action)
                done = new_state == 'terminal'

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                
                if done or t == num_steps-1:
                    if reward < 1:
                        print('OH')
                    #Qval, _ = actor_critic.forward(featurize_task(new_state))
                    #Qval = Qval.detach().numpy()[0,0]
                    Qval = 0.0
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(t)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    if episode % 100 == 0:                    
                        print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), t, average_lengths[-1]))
                    break
            
            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval
    
            #update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss# + 0.001 * entropy_term

            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()

            
        
        self.actor_critic = actor_critic
        pickle.dump(actor_critic, open(f'pretrained/actor_critic_{self.level}.pkl', "wb"))
        return actor_critic
    
    def solve(self, tasks, opt_seqs, H=100):
        solved = 0
        extra_steps = 0
        env = Environment()
        for i in range(len(tasks)):
            task = tasks[i]
            opt_seq = opt_seqs[i]['sequence']
            
            state = task
            env.init(state)
            for t in range(H):
                state_vect = featurize_task(state)
                value, policy_dist = self.actor_critic.forward(state_vect)
                value = value.detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 
                action = dist.argmax()

                state, reward = env.step(action)
                if state == 'terminal':
                    break
            
            solved += (reward > 0)
            extra_steps += t - len(opt_seq) + 1 if reward > 0 else 0
        
        accr = solved/len(tasks)
        avg_extra_steps = extra_steps/len(tasks)

        print(f'Attempted {len(tasks)} tasks, correctly solved {solved}. Accuracy={accr*100:.2f}%, avg extra steps={avg_extra_steps:.2f}')


