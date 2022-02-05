from common import Action
from algorithms import NeuralAgent
from environment import KarelEnv
import numpy as np
from copy import deepcopy

class KarelAgent():
    def __init__(self, base_agent, env):
        self.base_agent = base_agent
        self.env = env
        self.init_buffers() 

    def init_buffers(self):
        # Experience buffer
        self.exp_buffer = []
        self.act_buffer = []
    
    def check_cycle(self, state):
        if state in self.exp_buffer:
            return True

        self.exp_buffer.append(deepcopy(state))
        return False

    def break_cycle(self, state):
        all_actions = sorted(Action.from_str.values())
        all_rewards = []
        for action in all_actions:
            _, probed_reward, probed_done, probed_info = self.env.probe(action)
            all_rewards.append(probed_reward)
        break_action = np.argmax(all_rewards)
        return break_action

    def solve(self, tasks, opt_seqs=None, H=50):
        solved = 0
        solved_opt = 0
        extra_steps = 0

        self.base_agent.policy.eval()
        for i in range(len(tasks)):
            task_state = tasks[i]

            if opt_seqs:
                opt_seq = opt_seqs[i]["sequence"]
            
            cmd_seq = []
            state = self.env.reset(task_state)
            self.init_buffers()
            for t in range(H):
                is_cycle = self.check_cycle(self.env.get_full_state())
                if is_cycle:
                    action = self.break_cycle(state)
                else:
                    action = self.base_agent.act(state)

                cmd_seq.append(Action.to_str[action])

                _, probed_reward, probed_done, probed_info = self.env.probe(action)
                if probed_info["crashed"]:
                    action = self.break_cycle(state)
                    #print("crashed")

                state, reward, done, info = self.env.step(action)

                if done:
                    break


            solved += info["solved"]
            if opt_seqs:
                solved_opt += (info["solved"] and t + 1 <= len(opt_seq))
                extra_steps += t - len(opt_seq) + 1 if info["solved"] else 0
        
        accr = solved / len(tasks)

        if opt_seqs:
            opt_accr = solved_opt/len(tasks)
            avg_extra_steps = extra_steps / len(tasks)
            opt_stats_str = f", Accuracy(optimally solved)={opt_accr*100:.2f}%, avg extra steps={avg_extra_steps:.2f}"
        else:
            opt_stats_str = ""

        print(f"Attempted {len(tasks)} tasks, correctly solved {solved}. Accuracy(solved)={accr*100:.2f}%{opt_stats_str}")
