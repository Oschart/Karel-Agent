# Imports
import os
import torch
from algorithms import ActorCritic, SoftActorCritic
from environment import KarelEnv
from karel_agent import KarelAgent
from parse_utils import parse_dataset, parse_test_dataset
from policy_models import ActorCriticMLP
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
#pyo.init_notebook_mode()
from plot_utils import plot_lines


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch_gen = torch.manual_seed(1998)

X_train, y_train = parse_dataset(levels=["easy", "medium", "hard"], mode='train', compact=True)
X_test, y_test = parse_dataset(levels=["hard"], mode='val', compact=True)

state_size = 16*5 + 8
env_binary = KarelEnv(is_compact=True, reward_func='binary')
env_complex = KarelEnv(is_compact=True, reward_func='complex')

train_config = dict(max_episodes=250000, early_stop=80, load_pretrained=False, verbose=True)

a2c_policy = ActorCriticMLP(state_size, 6, soft_critic=False, hidden_sizes=(256, 256))
a2c_agent = ActorCritic(a2c_policy, env=env_binary, learn_by_demo=True, **train_config, variant_name='basic_256x256')

stats = a2c_agent.train(X_train, expert_traj=y_train, data_val=(X_test, y_test), reshuffle=False, proliferate=False)

print("Base A2C Performance:")
a2c_agent.evaluate(X_test, y_test, verbose=True)

print("Wrapped A2C Performance:")
karel_agent = KarelAgent(a2c_agent, env_complex)
karel_agent.solve(X_test)