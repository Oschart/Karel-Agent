from common import Action
import copy
import gym
from gym import spaces
import numpy as np
from random import choice
from copy import deepcopy


from parse_utils import vectorize_obs


class KarelEnv(gym.Env):
    N_ACTIONS = 6
    # Direction encoding
    dir_to_dxy = {"north": (-1, 0), "east": (0, 1),
                  "south": (1, 0), "west": (0, -1)}
    dir_ord = ["north", "east", "south", "west"]

    def __init__(self, task_space=None, is_compact=True, reward_func='binary'):
        super(KarelEnv, self).__init__()

        if reward_func == 'binary':
            self.R = self.R_binary
        else:
            self.R = self.R_complex

        self.task_space = task_space
        self.is_compact = is_compact
        self.debug = False
        self.probe_mode = False

        self.obs_shape = (4, 4, 11)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.uint8
        )

        self.action_handlers = {
            Action.move: self.move,
            Action.turnLeft: lambda src_state: self.turn(-1, src_state),
            Action.turnRight: lambda src_state: self.turn(1, src_state),
            Action.pickMarker: self.pickMarker,
            Action.putMarker: self.putMarker,
            Action.finish: self.finish,
        }
        # self.reset()

    def reset(self, init_state=None):
        if init_state is None:
            init_state = choice(self.task_space)

        self.init(init_state)
        return vectorize_obs(init_state, self.is_compact)

    def init(self, task):
        self.task = copy.deepcopy(task)
        self.is_terminal = False

        self.task["pregrid_markers"] = set(
            map(tuple, self.task["pregrid_markers"]))
        self.task["postgrid_markers"] = set(
            map(tuple, self.task["postgrid_markers"]))

        # Active (i.e., changing) state, note that this is not the total state
        self.state = {
            "agent_r": self.task["pregrid_agent_row"],
            "agent_c": self.task["pregrid_agent_col"],
            "agent_d": self.task["pregrid_agent_dir"],
            "markers": self.task["pregrid_markers"],
        }

        # Active target state, note that this is not the total state
        self.target_state = {
            "agent_r": self.task["postgrid_agent_row"],
            "agent_c": self.task["postgrid_agent_col"],
            "agent_d": self.task["postgrid_agent_dir"],
            "markers": self.task["postgrid_markers"],
        }

    def get_full_state(self, state=None):
        if state is None:
            state = self.state

        if state == "terminal":
            return "terminal"

        task_state = copy.deepcopy(self.task)
        task_state["pregrid_agent_row"] = state["agent_r"]
        task_state["pregrid_agent_col"] = state["agent_c"]
        task_state["pregrid_agent_dir"] = state["agent_d"]
        task_state["pregrid_markers"] = state["markers"]
        return task_state

    def generate_rollout(self, PI, H):
        EP = []
        for i in range(H):
            s = self.get_full_state()
            if s == "terminal":
                break

            a = PI(s)
            r = self.R(self.state, a)
            EP.append((s, a, r))
            self.step(a)

        return EP

    def probe(self, action):
        state_copy = deepcopy(self.state)
        self.next_state, self.is_terminal = self.action_handlers[action](state_copy)
        r = self.R(self.state, action)
        is_solved = self.state == self.target_state and action == Action.finish
        has_crashed = self.is_terminal and not is_solved

        next_obs = vectorize_obs(self.get_full_state(), self.is_compact)
        return next_obs, r, self.is_terminal, {"solved": is_solved, "crashed": has_crashed}

    def step(self, action):
        state_copy = deepcopy(self.state)
        self.next_state, self.is_terminal = self.action_handlers[action](state_copy)
        r = self.R(self.state, action)
        is_solved = self.state == self.target_state and action == Action.finish
        has_crashed = self.is_terminal and not is_solved
        self.state = self.next_state

        next_obs = vectorize_obs(self.get_full_state(), self.is_compact)
        return next_obs, r, self.is_terminal, {"solved": is_solved, "crashed": has_crashed}

    def R_binary(self, s, a):
        if s == self.target_state and a == Action.finish:
            return 1
        else:
            return 0

    def R_complex(self, s, a):
        if self.debug:
            next_state, is_terminal = self.action_handlers[a](src_state=deepcopy(s))
        else:
            next_state, is_terminal = self.next_state, self.is_terminal

        if s == self.target_state and a == Action.finish:   # Task solved
            return 20
        elif is_terminal:  # Crash
            return -10
        elif a == Action.move:
            vd1 = s["agent_r"]-self.task["postgrid_agent_row"]
            hd1 = s["agent_c"]-self.task["postgrid_agent_col"]
            d1 =  abs(hd1) + abs(vd1)
            
            vd2 = next_state["agent_r"]-self.task["postgrid_agent_row"]
            hd2 = next_state["agent_c"]-self.task["postgrid_agent_col"]
            d2 =  abs(hd2) + abs(vd2)
            
            if s['markers'] == self.task["postgrid_markers"]:
                if d2 < d1:
                    return 1
                elif d2 > d1:
                    h_ort, v_ort = hd1//max(abs(hd1),1), vd1//max(abs(vd1),1)
                    h_blocked, v_blocked = False, False
                    for step in range(1, 5):
                        h_blocked |= [s["agent_r"], s["agent_c"]+h_ort*step] in self.task["walls"]
                        v_blocked |= [s["agent_r"]+v_ort*step, s["agent_c"]] in self.task["walls"]
                    if not (h_blocked and v_blocked):
                        return -1
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
        elif a == Action.putMarker:
            loc = (s["agent_r"], s["agent_c"])
            if loc in self.task["postgrid_markers"] and loc not in s["markers"]:
                return 3
            else:
                return -3
        elif a == Action.pickMarker:
            loc = (s["agent_r"], s["agent_c"])
            if loc not in self.task["postgrid_markers"] and loc in s["markers"]:
                return 3
            else:
                return -3
        else:
            return 0



    def move(self, src_state):
        agent_r, agent_c = src_state["agent_r"], src_state["agent_c"]
        agent_d = src_state["agent_d"]

        dxy = self.dir_to_dxy[agent_d]
        next_pos = [agent_r + dxy[0], agent_c + dxy[1]]

        out_of_bounds = (
            next_pos[0] >= self.task["gridsz_num_rows"]
            or next_pos[1] >= self.task["gridsz_num_cols"]
            or next_pos[0] < 0
            or next_pos[1] < 0
        )
        wall_hit = next_pos in self.task["walls"]

        if out_of_bounds or wall_hit:
            return src_state, True
        
        src_state["agent_r"], src_state["agent_c"] = next_pos[0], next_pos[1]

        return src_state, False

    def turn(self, clk_ort, src_state):
        agent_r, agent_c = src_state["agent_r"], src_state["agent_c"]
        agent_d = src_state["agent_d"]

        dir_idx = self.dir_ord.index(agent_d)
        next_idx = (dir_idx + clk_ort + 4) % 4
        new_dir = self.dir_ord[next_idx]

        src_state["agent_d"] = new_dir
        return src_state, False

    def pickMarker(self, src_state):
        agent_r, agent_c = src_state["agent_r"], src_state["agent_c"]
        if (agent_r, agent_c) not in src_state["markers"]:
            return src_state, True

        src_state["markers"].remove((agent_r, agent_c))
        return src_state, False


    def putMarker(self, src_state):
        agent_r, agent_c = src_state["agent_r"], src_state["agent_c"]
        if (agent_r, agent_c) in src_state["markers"]:
            return src_state, True
        
        src_state["markers"].add((agent_r, agent_c))
        return src_state, False

    def finish(self, src_state):
        return src_state, True
