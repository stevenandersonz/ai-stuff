import math
import random
import copy
import gymnasium as gym
from collections import namedtuple

C = 1.4
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "terminated","truncated","info"))

class WithSnapshot(gym.Wrapper):
    '''Wraps the env and allows copying its state, so it can be simulated and explore at each timestep'''
    def __init__(self, env):
        super().__init__(env)
    def get_snapshot(self):
        return copy.deepcopy(self.env)
    def load_snapshot(self, snapshot):
        self.env = copy.deepcopy(snapshot)
    def get_result(self, snapshot, action):
        self.load_snapshot(snapshot)
        observation, reward, terminated, truncated, info = env.step(action)
        next_snapshot = self.get_snapshot()
        return ActionResult(next_snapshot, observation, reward, terminated,truncated, info)

def ucb_score(node, scale=10):
    if node.visits == 0 :
        return float('inf')
    exploitation_term = node.qvalue_sum / node.visits
    exploration_term = math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation_term + C * exploration_term

class Node: 

    def __init__(self, parent, action):
        self.action = action
        self.children = []
        self.parent = parent
        self.visits = 0
        self.qvalue_sum=0
        if parent:
            res = env.get_result(parent.snapshot, action)
            self.snapshot, self.observation, self.reward, self.terminated, self.truncated, _ = res

    def select(self):
        scores = [ucb_score(child) for child in self.children] 
        best_score = max(scores)
        max_indices = [i for i in range(len(scores)) if scores[i] == best_score]
        return self.children[random.choice(max_indices)]

    def expand(self):
        assert not self.terminated or not self.truncated, "Can't expand from terminal state"
        for action in range(env.action_space.n):
            self.children.append(Node(self, action))

    def update(self, child_qvalue):
        self.visits +=1
        my_qvalue = self.reward + child_qvalue
        self.qvalue_sum += my_qvalue
        if self.parent:
            self.parent.update(my_qvalue)

    def rollout(self, t_max=10*2):
        rollout_reward =  0
        env.load_snapshot(self.snapshot)
        for _ in range(t_max) :
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rollout_reward += reward
            if terminated or truncated:
                return self.reward
        return rollout_reward
    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child
    
def mcts(root, max_iter=1):
    for _ in range(max_iter):
        node = root
        while node.children:
            node = node.select()
        if node.terminated:
            node.update(0)
        else:
            if not node.visits:
                node.expand()
            result = node.rollout() 
            node.update(result)

    
class Root(Node):
    def __init__(self, snapshot, observation):
        """
        creates special node that acts like tree root
        :snapshot: snapshot (from env.get_snapshot) to start planning from
        :observation: last environment observation
        """

        self.parent = self.action = None
        self.children = []  # set of child nodes

        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.reward = 0
        self.visits = 0 
        self.terminated = False
        self.qvalue_sum = 0

    @staticmethod
    def from_node(node):
        """initializes node as root"""
        root = Root(node.snapshot, node.observation)
        # copy data
        root.visits = node.visits
        root.qvalue_sum = node.qvalue_sum
        root.children = node.children
        root.terminated = node.terminated
        return root

env = WithSnapshot(gym.make('CartPole-v1'))
root_observation, _ = env.reset()
s0 = env.get_snapshot()
root = Root(s0,root_observation)
total_rewards = 0

def print_tree(node, depth=0):
    if node is None:
        return
    indent = ' ' * depth
    print(f"{indent} v:{node.visits} w:{node.qvalue_sum} t:{node.terminated} a:{node.action} o:{node.observation}")
    for child in node.children:
        print_tree(child, depth + 2)

mcts(root, 2000)

env.load_snapshot(s0)
while True:
    env.unwrapped.render_mode = 'human'
    best_node = max(root.children, key=lambda node: node.qvalue_sum)
    observation, reward, terminated, truncated, info = env.step(best_node.action)
    total_rewards += reward
    if terminated or truncated:
        print(f"total rewards: {total_rewards}")
        break
    for child in root.children:
        if child != best_node:
            child.safe_delete()
     # declare best child a new root
    root = Root.from_node(best_node)

    # you may want to expand tree here
    if not root.children:
        mcts(root, 2000)
        env.load_snapshot(s0)
    # <YOUR CODE>
env.close()
