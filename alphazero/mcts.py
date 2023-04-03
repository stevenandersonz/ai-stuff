import math
import random
import torch

C_PUCT = 1.4

def puct(node):
    '''
    This search control strategy initially prefers actions with high prior probability 
    and low visit count, but asympotically prefers actions with high action value
    '''
    return node.Q + C_PUCT * node.P * (math.sqrt(node.parent.N) / (1 + node.N))

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
        self.P = 0 
    def select(self):
        scores = [puct(child) for child in self.children]
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
def mcts(root, m, env, n_simulations):
    m.eval()
    for _ in range(n_simulations):
        node = root
        while node.children:
            node = node.select()
        if node.terminated:
            node.update(0)
        else:
            if not node.N:
                node.expand(env)
            p,v = m(torch.tensor(node.observation).view(-1, len(node.observation)))
            node.update(v.view(-1)[0])
    m.train()