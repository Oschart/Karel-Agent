#%%
import os
import json
import numpy as np
from common import Action, Direction
from sklearn.utils import shuffle
import pickle as pkl

data_level2dir = {"easy": 'data_easy', "medium": 'data_medium', "hard": 'data'}

def parse_dataset(levels=["easy"], mode="train", sort_by_hardness=False, compact=True):
    size_mode = 'compact' if compact else 'verbose'
    pickled_path = f'datasets/preprocessed_{mode}_{"_".join(levels)}_{size_mode}.pkl'
    if os.path.isfile(pickled_path):
        all_tasks, all_seqs = pkl.load(open(pickled_path, 'rb'))
        return all_tasks, all_seqs
    
    all_tasks, all_seqs = [], []
    for level in levels:
        data_dir = f"datasets/{data_level2dir[level]}"
        tasks_dir = f"{data_dir}/{mode}/task"
        seqs_dir = f"{data_dir}/{mode}/seq"
        task_fnames = sorted(os.listdir(tasks_dir), key=lambda s: int(s.split("_")[0]))
        seq_fnames = sorted(os.listdir(seqs_dir), key=lambda s: int(s.split("_")[0]))

        tasks = [json.load(open(f"{tasks_dir}/{fname}")) for fname in task_fnames]
        seqs = [json.load(open(f"{seqs_dir}/{fname}")) for fname in seq_fnames]

        all_tasks.extend(tasks)
        all_seqs.extend(seqs)

    if sort_by_hardness:
        all_tasks, all_seqs = (
            list(t) for t in zip(*sorted(zip(all_tasks, all_seqs), key=compute_hardness))
        )
    else:
        all_tasks, all_seqs = shuffle(all_tasks, all_seqs, random_state=73)

    if pickled_path:
        pkl.dump((all_tasks, all_seqs), open(pickled_path, 'wb'))
    return all_tasks, all_seqs


def parse_dataset_by_dir(tasks_dir):
    all_tasks = []
    task_fnames = sorted(os.listdir(tasks_dir), key=lambda s: int(s.split("_")[0]))
    task_ids = list(map(lambda s: s.split("_")[0], task_fnames))
    tasks = [json.load(open(f"{tasks_dir}/{fname}")) for fname in task_fnames]
    all_tasks.extend(tasks)
    return all_tasks, task_ids


def parse_test_dataset(levels=["hard"], compact=True):
    size_mode = 'compact' if compact else 'verbose'
    mode = 'test_without_seq'
    pickled_path = f'datasets/preprocessed_{mode}_{"_".join(levels)}_{size_mode}.pkl'
    if os.path.isfile(pickled_path):
        all_tasks = pkl.load(open(pickled_path, 'rb'))
        return all_tasks
    
    all_tasks = []
    for level in levels:
        data_dir = f"datasets/{data_level2dir[level]}"
        tasks_dir = f"{data_dir}/{mode}/task"
        task_fnames = sorted(os.listdir(tasks_dir), key=lambda s: int(s.split("_")[0]))
        tasks = [json.load(open(f"{tasks_dir}/{fname}")) for fname in task_fnames]
        all_tasks.extend(tasks)

    if pickled_path:
        pkl.dump(all_tasks, open(pickled_path, 'wb'))
    return all_tasks

def compute_hardness(task_seq):
    seq = task_seq[1]["sequence"]
    task_len = len(seq)
    pickMarkers = seq.count("pickMarker")
    putMarkers = seq.count("putMarker")
    return task_len + 1.5 * putMarkers + 2.0 * pickMarkers


def vectorize_obs(task, is_compact=True):
    n, m = task["gridsz_num_rows"], task["gridsz_num_cols"]

    ar1 = task["pregrid_agent_row"]
    ac1 = task["pregrid_agent_col"]
    ad1 = Direction.from_str[task["pregrid_agent_dir"]]

    ar2 = task["postgrid_agent_row"]
    ac2 = task["postgrid_agent_col"]
    ad2 = Direction.from_str[task["postgrid_agent_dir"]]


    if is_compact:
        feat_v = np.zeros((n, m, 1 + 2 * 2))
        feat_rot = np.zeros((8))

        feat_v[ar1, ac1, 2] = 1
        feat_v[ar2, ac2, 4] = 1

        # Represent walls
        for w_pos in task["walls"]:
            feat_v[w_pos[0], w_pos[1], 0] = 1

        # Represent markers
        for m_pos in task["pregrid_markers"]:
            feat_v[m_pos[0], m_pos[1], 1] = 1

        for m_pos in task["postgrid_markers"]:
            feat_v[m_pos[0], m_pos[1], 3] = 1
        
        rot_bits = np.zeros((8))
        rot_bits[ad1] = 1
        rot_bits[ad2 + 4] = 1

        feat_v = feat_v.flatten()
        feat_v = np.concatenate((feat_v, rot_bits))

    else:
        feat_v = np.zeros((n, m, 1 + 2 * 5))

        feat_v[ar1, ac1, ad1 + 2] = 1
        feat_v[ar2, ac2, ad2 + 7] = 1

        # Represent walls
        for w_pos in task["walls"]:
            feat_v[w_pos[0], w_pos[1], 0] = 1

        # Represent markers
        for m_pos in task["pregrid_markers"]:
            feat_v[m_pos[0], m_pos[1], 1] = 1

        for m_pos in task["postgrid_markers"]:
            feat_v[m_pos[0], m_pos[1], 6] = 1

    return feat_v


def vectorize_seq(seq):
    feat_v = [Action.from_str[cmd] for cmd in seq["sequence"]]
    return feat_v

