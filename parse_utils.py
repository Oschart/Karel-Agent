import os
import json
import numpy as np

def parse_dataset(data_dir='datasets/data_easy', mode='train', return_vect=False):
    tasks_dir = f'{data_dir}/{mode}/task'
    seqs_dir = f'{data_dir}/{mode}/seq'
    task_fnames = os.listdir(tasks_dir)
    seq_fnames = os.listdir(seqs_dir)

    tasks = [json.load(open(f'{tasks_dir}/{fname}')) for fname in task_fnames]
    seqs = [json.load(open(f'{seqs_dir}/{fname}')) for fname in seq_fnames]

    if return_vect:
        X = [featurize_task(task) for task in tasks]
        y = [featurize_seq(seq) for seq in seqs]
        tasks = {"raw": tasks, "vect": X}
        seqs = {"raw": seqs, "vect": y}
    
    return tasks, seqs



def featurize_task(task):
    n, m = task['gridsz_num_rows'], task['gridsz_num_cols']
    dir2idx = {
        'north': 1,
        'east': 2,
        'south': 3,
        'west': 4
    }
    feat_v = np.zeros((n,m,1 + 2*5))
    ar1 = task['pregrid_agent_row']
    ac1 = task['pregrid_agent_col']        
    ad1 = dir2idx[task['pregrid_agent_dir']]

    feat_v[ar1, ac1, ad1] = 1     
    
    ar2 = task['postgrid_agent_row']
    ac2 = task['postgrid_agent_col']   
    ad2 = dir2idx[task['postgrid_agent_dir']]

    feat_v[ar2, ac2, ad2 + 5] = 1

    # Represent walls
    for w_pos in task['walls']:
        feat_v[w_pos[0], w_pos[1], 0] = 1
    
    # Represent markers
    # pass
    
    return feat_v


def featurize_seq(seq):
    cmds = ['move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker', 'finish']
    cmd2idx = {cmd: i for i, cmd in enumerate(cmds)}
    feat_v = [cmd2idx[cmd] for cmd in seq['sequence']]
    
    return feat_v

