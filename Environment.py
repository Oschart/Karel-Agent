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
    dir_to_dxy = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
    dir_ord = ["north", "east", "south", "west"]

    def __init__(self, task_space=None, is_compact=True):
        super(KarelEnv, self).__init__()

        self.task_space = task_space
        self.is_compact = is_compact
        self.probe_mode = False

        self.obs_shape = (4, 4, 11)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.uint8
        )

        self.action_handlers = {
            Action.move: self.move,
            Action.turnLeft: lambda: self.turn(-1),
            Action.turnRight: lambda: self.turn(1),
            Action.pickMarker: self.pickMarker,
            Action.putMarker: self.putMarker,
            Action.finish: self.finish,
        }
        #self.reset()

    def reset(self, init_state=None):
        if init_state is None:
            init_state = choice(self.task_space)
            
        self.init(init_state)
        return vectorize_obs(init_state, self.is_compact)

    def init(self, task):
        self.task = copy.deepcopy(task)
        self.is_terminal = False

        self.task["pregrid_markers"] = set(map(tuple, self.task["pregrid_markers"]))
        self.task["postgrid_markers"] = set(map(tuple, self.task["postgrid_markers"]))

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
        r = self.R(self.state, action)
        is_terminal_backup = self.is_terminal
        self.probe_mode = True
        next_state = self.action_handlers[action]()
        is_mock_terminal = self.is_terminal
        self.probe_mode = False
        self.is_terminal = is_terminal_backup

        next_obs = vectorize_obs(self.get_full_state(state=self.mock_state), self.is_compact)

        return next_obs, r, is_mock_terminal, {}

    def step(self, action):
        r = self.R(self.state, action)
        self.action_handlers[action]()

        next_obs = vectorize_obs(self.get_full_state(), self.is_compact)

        return next_obs, r, self.is_terminal, {}

    def R(self, s, a):
        if s == self.target_state and a == Action.finish:
            return 1
        else:
            return 0

    def move(self):
        agent_r, agent_c = self.state["agent_r"], self.state["agent_c"]
        agent_d = self.state["agent_d"]

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
            self.is_terminal = True
            return

        if self.probe_mode:
            self.mock_state = deepcopy(self.state)
            self.mock_state["agent_r"], self.mock_state["agent_c"] = next_pos[0], next_pos[1]
        else:
            self.state["agent_r"], self.state["agent_c"] = next_pos[0], next_pos[1]


    def turn(self, clk_ort):
        agent_r, agent_c = self.state["agent_r"], self.state["agent_c"]
        agent_d = self.state["agent_d"]

        dir_idx = self.dir_ord.index(agent_d)
        next_idx = (dir_idx + clk_ort + 4) % 4
        new_dir = self.dir_ord[next_idx]

        if self.probe_mode:
            self.mock_state = deepcopy(self.state)
            self.mock_state["agent_d"] = new_dir
        else:
            self.state["agent_d"] = new_dir

    def pickMarker(self):
        agent_r, agent_c = self.state["agent_r"], self.state["agent_c"]
        if (agent_r, agent_c) not in self.state["markers"]:
            self.is_terminal = True
            return 

        if self.probe_mode:
            self.mock_state = deepcopy(self.state)
            self.mock_state["markers"].remove((agent_r, agent_c))
        else:
            self.state["markers"].remove((agent_r, agent_c))


    def putMarker(self):
        agent_r, agent_c = self.state["agent_r"], self.state["agent_c"]
        if (agent_r, agent_c) in self.state["markers"]:
            self.is_terminal = True
            return

        if self.probe_mode:
            self.mock_state = deepcopy(self.state)
            self.mock_state["markers"].add((agent_r, agent_c))
        else:
            self.state["markers"].add((agent_r, agent_c))


    def finish(self):
        self.is_terminal = True

