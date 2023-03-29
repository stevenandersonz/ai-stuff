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

def ucb_score(node):
    if node.visits == 0 or not node.parent:
        return float('inf')
    exploitation_term = node.reward / node.visits
    exploration_term = math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation_term + C * exploration_term

class Node: 

    def __init__(self, parent, action):
        self.action = action
        self.children = []
        self.parent = parent
        self.visits = 0
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

    def update(self, reward):
        self.visits +=1
        self.reward += reward 
        if self.parent:
            self.parent.update(reward)

    def rollout(self, t_max=10*2):
        rollout_reward =  0
        env.load_snapshot(self.snapshot)
        for _ in range(t_max) :
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rollout_reward += reward
            if terminated or truncated:
                return self.reward
        return rollout_reward
    
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

    


env = WithSnapshot(gym.make('CartPole-v1'))
root_observation, _ = env.reset()
s0 = env.get_snapshot()
root = Node(None,None)
root.observation =root_observation
root.snapshot = s0
root.reward = 0
root.terminated = False

move = mcts(root, 1000)

env.load_snapshot(s0)
while True:
    bestNode = max(root.children, key=lambda node: node.visits)
    observation, reward, terminated, truncated, info = env.step(bestNode.action)
    if terminated or truncated:
        break

env.close()
