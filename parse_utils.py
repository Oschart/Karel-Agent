#%%
import os
import json
import numpy as np
from common import Action, Direction

def parse_dataset(data_dir='datasets/data_easy', mode='train', return_vect=False):
    tasks_dir = f'{data_dir}/{mode}/task'
    seqs_dir = f'{data_dir}/{mode}/seq'
    task_fnames = sorted(os.listdir(tasks_dir), key= lambda s: int(s.split('_')[0]))
    seq_fnames = sorted(os.listdir(seqs_dir), key= lambda s: int(s.split('_')[0]))

    tasks = [json.load(open(f'{tasks_dir}/{fname}')) for fname in task_fnames]
    seqs = [json.load(open(f'{seqs_dir}/{fname}')) for fname in seq_fnames]

    if return_vect:
        X = [vectorize_obs(task) for task in tasks]
        y = [vectorize_seq(seq) for seq in seqs]
        tasks = {"raw": tasks, "vect": X}
        seqs = {"raw": seqs, "vect": y}
    
    return tasks, seqs



def vectorize_obs(task):
    n, m = task['gridsz_num_rows'], task['gridsz_num_cols']
    feat_v = np.zeros((n,m,1 + 2*5))

    ar1 = task['pregrid_agent_row']
    ac1 = task['pregrid_agent_col']        
    ad1 = Direction.from_str[task['pregrid_agent_dir']]

    ar2 = task['postgrid_agent_row']
    ac2 = task['postgrid_agent_col']   
    ad2 = Direction.from_str[task['postgrid_agent_dir']]

    feat_v[ar1, ac1, ad1 + 2] = 1     
    feat_v[ar2, ac2, ad2 + 7] = 1

    # Represent walls
    for w_pos in task['walls']:
        feat_v[w_pos[0], w_pos[1], 0] = 1
    
    # Represent markers
    for m_pos in task['pregrid_markers']:
        feat_v[m_pos[0], m_pos[1], 1] = 1
    
    for m_pos in task['postgrid_markers']:
        feat_v[m_pos[0], m_pos[1], 6] = 1
    
    return feat_v


def vectorize_seq(seq):
    feat_v = [Action.from_str[cmd] for cmd in seq['sequence']]
    return feat_v



'''
X_easy, _ = parse_dataset(data_dir='datasets/data_easy')
X_medium, _ = parse_dataset(data_dir='datasets/data_medium')
X, _ = parse_dataset(data_dir='datasets/data')


#%%

for taskm in X_medium:
    subs = False
    for taskh in X:
        if taskm == taskh:
            print('OVERLAP')
            break



#%%
X_set = set(map(str,X))
X_easy_set = set(map(str,X_easy))
X_medium_set = set(map(str,X_medium))

#%%
med_inter_hard = X_medium_set.intersection(X_set)
easy_inter_hard = X_easy_set.intersection(X_set)
print(len(easy_inter_hard)/len(X_easy_set))
print(len(med_inter_hard)/len(X_medium_set))
# %%
for sth in X_easy_set:
    print(type(sth))
    break
# %%
print(len(easy_inter_hard))
# %%
'''
