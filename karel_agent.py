from common import Action
from algorithms import NeuralAgent
from environment import KarelEnv
import numpy as np
from copy import deepcopy
import json

class KarelAgent():
	def __init__(self, base_agent, env, cycle_detection_enabled=False, env_probe_enabled=True):
		self.base_agent = base_agent
		self.env = env
		self.cycle_detection_enabled = cycle_detection_enabled
		self.env_probe_enabled = env_probe_enabled
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

	def act_on_reward(self, state):
		all_actions = sorted(Action.from_str.values())
		all_rewards = []
		for action in all_actions:
			_, probed_reward, probed_done, probed_info = self.env.probe(action)
			all_rewards.append(probed_reward)
		break_action = np.argmax(all_rewards)
		return break_action


	def solve(self, tasks, opt_seqs=None, H=50, output_dir=None, task_ids=None):
		solved = 0
		solved_opt = 0
		extra_steps = 0
		all_seqs = []
		self.base_agent.policy.eval()
		for i in range(len(tasks)):
			task_state = tasks[i]

			if opt_seqs:
				opt_seq = opt_seqs[i]["sequence"]
			
			cmd_seq = []
			state = self.env.reset(task_state)
			self.init_buffers()
			for t in range(H):
				if self.cycle_detection_enabled:
					is_cycle = self.check_cycle(self.env.get_full_state())
					if is_cycle:
						action = self.act_on_reward(state)
					else:
						action = self.base_agent.act(state)
				else:
					action = self.base_agent.act(state)
					

				cmd_seq.append(Action.to_str[action])

				if self.env_probe_enabled:
					_, probed_reward, probed_done, probed_info = self.env.probe(action)
					if probed_info["crashed"]:
						action = self.act_on_reward(state)

				state, reward, done, info = self.env.step(action)

				if done:
					break
			
			all_seqs.append(cmd_seq)
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

		if output_dir:
			self.write_sequences(all_seqs, output_dir, task_ids)
		else:
			self.print_sequences(all_seqs, task_ids)

		print('='*120)
		print(f"Attempted {len(tasks)} tasks, correctly solved {solved} of them. Accuracy(solved)={accr*100:.2f}%{opt_stats_str}")

	def print_sequences(self, sequences, task_ids=None):
		if task_ids:
			id_seq_pairs = zip(task_ids, sequences)
		else:
			id_seq_pairs = enumerate(sequences)

		for i, seq in id_seq_pairs:
			seq_str = ', '.join(seq)
			print('-'*120)
			print(f"Task {i} command sequence: {seq_str}")

	def write_sequences(self, sequences, output_path, task_ids=None):
		if task_ids:
			id_seq_pairs = zip(task_ids, sequences)
		else:
			id_seq_pairs = enumerate(sequences)

		for i, seq in id_seq_pairs:
			seq_dict = {"sequence": seq}
			save_path = f"{output_path}/{i}_seq.json"
			json.dump(seq_dict, open(save_path, 'w'))