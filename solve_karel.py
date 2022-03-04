import os
import torch
from algorithms import ActorCritic, SoftActorCritic
from environment import KarelEnv
from karel_agent import KarelAgent
from parse_utils import parse_dataset, parse_test_dataset, parse_dataset_by_dir
from policy_models import ActorCriticMLP
from argparse import ArgumentParser
import json 
torch_gen = torch.manual_seed(1998)

parser = ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--task-path", dest="task_path",
                    help="Path to Karel task", metavar="path/to/karel/task.json")
group.add_argument("-d", "--dataset-path", dest="dataset_dir",
                    help="Path to Karel dataset folder", metavar="path/to/karel/dataset")

args = parser.parse_args()

if args.task_path:
    X_test = [json.load(open(args.task_path))]
    task_ids = None
    output_dir = None
else:
    X_test, task_ids = parse_dataset_by_dir(args.dataset_dir)
    output_dir = 'testseqs'

state_size = 16*5 + 8
env_complex = KarelEnv(is_compact=True, reward_func='complex')

config = dict(load_pretrained=True, verbose=True)

a2c_policy = ActorCriticMLP(state_size, 6, soft_critic=False, hidden_sizes=(256, 256))
a2c_agent = ActorCritic(a2c_policy, env=env_complex, **config, variant_name='300K_reshuffle_reset')


karel_agent = KarelAgent(a2c_agent, env_complex, cycle_detection_enabled=True, env_probe_enabled=True)
karel_agent.solve(X_test, output_dir=output_dir, task_ids=task_ids)
