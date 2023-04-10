import math
import random
import numpy as np
import torch

C_PUCT = 1.4

def puct(node, exploration_weight):
    """
    Calculates the PUCT score for a given node in the tree.
    :param node: The node for which to calculate the PUCT score.
    :param exploration_weight: A hyperparameter that controls the amount of exploration.
    :return: The PUCT score for the node.
    """
    if node.N == 0:
        return float("inf")
    exploration_term = exploration_weight * math.sqrt(math.log(node.parent.N) / node.N)
    return  node.Q + exploration_term


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, outcome):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, outcome)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, outcomes = zip(*batch)
        return np.array(states), np.array(actions), np.array(outcomes)

class Node:
    def __init__(self, parent, action, observation=None, terminated=False, snapshot=None):
        '''
        N -> Visits Count 
        W -> Total Action Value
        Q -> Mean Action Value
        P -> Prior Prob of selecting edge
        '''
        self.parent = parent
        self.children = []
        self.snapshot = snapshot 
        self.action = action
        self.terminated = terminated
        self.observation = observation
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = []
    def select(self, m):
        scores = [puct(child, C_PUCT) for child in self.children]
        best_score = max(scores)
        best_idxs = [idx for idx in range(len(scores)) if scores[idx] == best_score]
        return self.children[random.choice(best_idxs)] 
    def expand(self, env):
        for a in range(env.action_space.n): 
            snap, o, terminated= env.get_result(self.snapshot, a)
            node = Node(self, a,  o, terminated, snap)
            self.children.append(node)
    def update(self, v):
        self.Q = (self.N * self.Q + v) / (self.N + 1)
        self.N += 1
        if self.parent:
            self.parent.update(self.Q)

# (p,v) = net(s)
# p -> vector of move probabilities. Represents the prob of selecting each move a, p_a = Pr(a|s)
# value v is a scalar evaluation, estimating the probability of current player winning from position s



        
@torch.no_grad()
def mcts(root, m, env, n_simulations, device="cuda"):
    m.eval()
    for _ in range(n_simulations):
        node = root
        while node.children:
            node = node.select(m)
        if node.terminated:
            node.update(0)
        else:
            if not node.N:
                node.expand(env)
            rewards = 0
            env.load_snapshot(node.snapshot)
            for _ in range(20): 
                x = torch.tensor(node.observation).view(-1, len(node.observation)).to(device)
                policy, _= m(x)
                action = torch.argmax(policy).item()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    rewards=reward
                    break
                rewards +=reward
            env.load_snapshot(node.snapshot)
            node.update(rewards)
    m.train()