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
        if self.unwrapped.render_mode=="human":
            self.unwrapped.surf = None
            self.unwrapped.screen = None
            self.unwrapped.clock = None
        return copy.deepcopy(self.env)
    def load_snapshot(self, snapshot):
        self.env = copy.deepcopy(snapshot)
    def get_result(self, snapshot, action):
        self.load_snapshot(snapshot)
        observation, reward, terminated, truncated, info = env.step(action)
        next_snapshot = self.get_snapshot()
        return ActionResult(next_snapshot, observation, reward, terminated,truncated, info)
def uct_score(node, C):
    if node.visits == 0 :
        return float('inf')
    exploitation_term = node.qvalue_sum / node.visits
    exploration_term = math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation_term + C * exploration_term

def ucb1_score(node, scale=10):
    if node.visits == 0 :
        return float('inf')
    exploitation_term = node.qvalue_sum / node.visits
    return exploitation_term + scale * math.sqrt(2 * math.log(node.parent.visits) / node.visits)

class Node: 

    def __init__(self, parent, action, snapshot=None):
        self.action = action
        self.children = []
        self.parent = parent
        self.visits = 0
        self.qvalue_sum=0
        self.terminated = False
        self.snapshot = None
        self.truncated = None
        self.reward = 0
        self.observation = None
        if parent:
            assert not self.parent.terminated, "calling step on terminated node"
            res = env.get_result(self.parent.snapshot, action)
            self.snapshot, self.observation, self.reward, self.terminated, self.truncated, _ = res
        else:
            env.load_snapshot(snapshot)
            self.snapshot = snapshot

    def select(self):
        scores = [ucb1_score(child, C) for child in self.children] 
        best_score = max(scores)
        max_indices = [i for i in range(len(scores)) if scores[i] == best_score]
        return self.children[random.choice(max_indices)]

    def expand(self):
        for action in range(env.action_space.n):
            self.children.append(Node(self, action))

    def update(self, child_qvalue):
        self.visits +=1
        my_qvalue = self.reward + child_qvalue
        self.qvalue_sum += my_qvalue
        if self.parent:
            self.parent.update(my_qvalue)

    def rollout(self, t_max=20):
        if self.terminated:
            return 0
        rollout_reward =  0
        env.load_snapshot(self.snapshot)
        for _ in range(t_max) :
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rollout_reward += reward
            if terminated:
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
        if node.visits > 0 and not node.terminated:
            node.expand()
            node = node.children[0] if not node.children[0].terminated else node.children[1]
        node.update(node.rollout())

    
env = WithSnapshot(gym.make('CartPole-v1'))
root_observation, _ = env.reset()
s0 = env.get_snapshot()
root = Node(None,None, s0)
total_rewards = 0

def print_tree(node, depth=0):
    if node is None:
        return
    indent = ' ' * depth
    print(f"{indent} id:{str(hash(node))[-6:]} v:{node.visits} w:{node.qvalue_sum} t:{node.terminated} a:{node.action} o:{node.observation}")
    for child in node.children:
        print_tree(child, depth + 2)

max_iter = 10000 
mcts(root, max_iter)
env.load_snapshot(s0)
env.unwrapped.render_mode = "human"
screen = None
surf = None
while True:
    best_node = max(root.children, key=lambda node: node.qvalue_sum)
    observation, reward, terminated, truncated, info = env.step(best_node.action)
    total_rewards += reward
    if terminated:
        print(f"total rewards: {total_rewards}")
        break
    
    #set the new root as best_node
    #set parent == none so the reference goes away and gets garbage collected
    #TODO: need to delete the all nodes of root except for best_node
    root = best_node
    root.parent = None
    root.action = None
    if not root.children:
        # Since root has not children we can build the tree again with mcts
        # get_snapshot() deep copy env but it can't copy pygame Objects
        # thats why i saved these properties so I can load them after
        screen = env.unwrapped.screen
        surf = env.unwrapped.surf
        clock = env.unwrapped.clock
        snap = env.get_snapshot()
        mcts(root, 100)
        env.load_snapshot(snap)
        env.unwrapped.screen = screen
        env.unwrapped.surf = surf
        env.unwrapped.clock = clock

env.close()
