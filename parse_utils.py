#%%
import os
import json
import numpy as np
from common import Action, Direction
from sklearn.utils import shuffle

data_level2dir = {"easy": 'data_easy', "medium": 'data_medium', "hard": 'data'}

def parse_dataset(levels=["easy"], mode="train", sort_by_hardness=False):
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

    return all_tasks, all_seqs


def compute_hardness(task_seq):
    seq = task_seq[1]["sequence"]
    task_len = len(seq)
    pickMarkers = seq.count("pickMarker")
    putMarkers = seq.count("putMarker")
    return task_len + 1.5 * putMarkers + 2.0 * pickMarkers


def vectorize_obs(task):
    n, m = task["gridsz_num_rows"], task["gridsz_num_cols"]
    feat_v = np.zeros((n, m, 1 + 2 * 5))

    ar1 = task["pregrid_agent_row"]
    ac1 = task["pregrid_agent_col"]
    ad1 = Direction.from_str[task["pregrid_agent_dir"]]

    ar2 = task["postgrid_agent_row"]
    ac2 = task["postgrid_agent_col"]
    ad2 = Direction.from_str[task["postgrid_agent_dir"]]

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

