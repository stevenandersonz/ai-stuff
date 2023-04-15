import gymnasium as gym
import numpy as np
from collections import namedtuple
import torch
import torch.optim as optim
import math
from model import AlphaZeroNet 
from copy import deepcopy

ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "terminated"))

class WithSnapshot(gym.Wrapper):
    '''Wraps the env and allows copying its state, so it can be simulated and explore at each timestep'''
    def __init__(self, env):
        super().__init__(env)
    def get_snapshot(self):
        return deepcopy(self.env)
    def load_snapshot(self, snapshot):
        self.env = deepcopy(snapshot)
    def get_result(self, snapshot, action):
        self.load_snapshot(snapshot)
        observation, reward, terminated, truncated, info = env.step(action)
        next_snapshot = self.get_snapshot()
        return ActionResult(next_snapshot, observation, reward, terminated)

env = WithSnapshot(gym.make('CartPole-v1'))
initial_obs, _ = env.reset()
initial_env = env.get_snapshot()
n_actions = env.action_space.n
state_size = env.observation_space.shape[0]

print(f"# actions: {n_actions} - state size {state_size}")

m = AlphaZeroNet(state_size, n_actions)

class Node():
    def __init__(self, prior, state=None, reward=0, terminated=False, parent=None, snapshot=None):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.parent = parent
        self.state = state
        self.terminated = terminated
        self.reward = reward
        self.snapshot = snapshot

    def expanded(self):
        return len(self.children)>0
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action 
def ucb_score(child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(child.parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score
def select_child (node):
    while  node.expanded():
       node = max(node.children.values(), key=ucb_score)
    return node

def print_tree(x, hist=[]):
    print("%4d %-16s %8.4f %4s %s" % (x.visit_count, str(hist), x.value_sum, x.prior, x.terminated))
    for key,c in x.children.items():
        print_tree(c,hist+[key])

def expand(node, action_prob):
    for a, prob in enumerate(action_prob):
        snap, o, reward, terminated= env.get_result(node.snapshot, a)
        new_node = Node(prob, o, reward, terminated, node, snap)
        node.children[a] = new_node

def backpropagate(node, value):
    node.visit_count += 1
    node.value_sum += value
    if node.parent:
        backpropagate(node.parent, node.value_sum)

def mcts(n_sim):
    root = Node(0, None)
    root.state = env.state
    root.snapshot = env.get_snapshot()
    x = torch.tensor(root.state, dtype=torch.float32).unsqueeze(0)
    action_prob, _ = m.predict(x)
    expand(root, action_prob)
    for _ in range(n_sim):
        node = select_child(root)
        if not node.terminated:
            x = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
            action_prob, value = m.predict(x)
            expand(node, action_prob)
            backpropagate(node, value)
        else:
            backpropagate(node, 0)
    return root

root = mcts(5)
print_tree(root)
