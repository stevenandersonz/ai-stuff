import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import namedtuple
import copy
import math
import random

C_PUCT = 1.4 # Constant determining the level of exploration
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "terminated"))

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
        return ActionResult(next_snapshot, observation, terminated)

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
    def expand(self):
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
def softmax(x, temperature=1.0):
    """Softmax function with a temperature feature that allows one to reshape the output probability distribution.
    High temperature implies a distribution tending towards uniform distribution
    Low temperature implies a distribution tending toward a one hot vector, or Dirac distribution
    A temperature of 1.0 yields a classical softmax function
    :param x: (ndarray of floats) input vector
    :param temperature: (float) allows to smooth or sharpen
    :return: (ndarray of floats) A probability distribution
    """
    if temperature < 0.1:
        raise ValueError('Temperature parameter should not be less than 0.1')
    t = 1/temperature
    return (torch.exp(x)**t/torch.sum(torch.exp(x)**t, dim=-1, keepdim=True))

class AlphaZeroNet(nn.Module):
    def __init__(self, n_obs, num_actions):
        super(AlphaZeroNet, self).__init__()
        self.linear1 = nn.Linear(n_obs, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head1 = nn.Linear(256, 64)
        self.value_head2 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # Policy head
        p = softmax(self.policy_head(x))
        # Value head
        v = F.relu(self.value_head1(x))
        v = self.value_head2(v)
        return p, v
        
@torch.no_grad()
def mcts(root, n_simulations):
    net.eval()
    for _ in range(n_simulations):
        node = root
        while node.children:
            node = node.select()
        if node.terminated:
            node.update(0)
        else:
            if not node.N:
                node.expand()
            p,v = net(torch.tensor(node.observation).view(-1, len(node.observation)))
            node.update(v)
    net.train()

def get_action_prob (temperature=1):
    action_visits = [child.N for child in root.children]
    action_prob = torch.zeros((2,))
    if temperature == 0:
        best_action_idx = np.argmax(action_visits)  
        action_prob[best_action_idx] = 1
        return action_prob, root.children[best_action_idx]
    
    counts = [v ** (1. / temp) for v in action_visits]
    counts_sum = float(sum(counts))
    action_prob = [v/counts_sum for v in counts]

    return action_prob 

plays = []
env = WithSnapshot(gym.make('CartPole-v1'))
root_observation, _ = env.reset()
net = AlphaZeroNet(len(root_observation), num_actions=env.action_space.n)
s0 = env.get_snapshot()
root = Node(None, None, observation=root_observation, terminated=False, snapshot=s0)
total_rewards = 0
n_episodes = 100
n_sim = 25
n_iter = 100
eval_iter = 20
v_resign = None
temp = 1
epochs=10
batch_size=4

def run_episodes(root):
    for _ in range(n_episodes):
        mcts(root, n_sim)
        env.load_snapshot(s0)
        while True:
            action_prob = get_action_prob(temperature=1)
            action = np.random.choice(len(action_prob), p=action_prob)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
            plays.append([observation, action_prob, reward])

def run_test(root):
    models = { 'prev':0, 'new':0}
    rewards = 0
    for model in ['prev','new']:
        temp = root
        print(f"running {model} model")
        net.load_state_dict(torch.load(f"./temp/{model}.pth"))
        mcts(temp, 100)
        env.load_snapshot(s0)
        while True:
            action_prob, node = get_action_prob(temperature=0)
            action = np.random.choice(len(action_prob), p=action_prob)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            rewards += reward
            temp = node 
            if not temp.children:
                snap = env.get_snapshot()
                mcts(temp, 100)
                print("rebuilding tree")
                env.load_snapshot(snap)

        models[model] = rewards
        print(f"reward for {model} model is {rewards}")
        rewards = 0
    return models 

def loss_pi(targets, outputs):
    return  -torch.sum(targets * torch.log(outputs)) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

def get_batch():
    sample_ids = np.random.randint(len(plays), size=batch_size)
    obs, pis, vs = list(zip(*[plays[i] for i in sample_ids]))
    obs = torch.tensor(np.array(obs).astype(np.float32))
    target_pis = torch.tensor(np.array(pis).astype(np.float32))
    target_vs = torch.tensor(np.array(vs).astype(np.float32))
    return obs, target_pis, target_vs
    
@torch.no_grad()
def estimate_loss():
    net.eval()
    losses = torch.zeros(eval_iter)
    for i in range(eval_iter):
        x, target_pis, target_vs = get_batch()
        pi, v = net(x)
        l_pi = loss_pi(target_pis, pi)
        l_vs = loss_v(target_vs, v)
        total_loss = l_pi + l_vs 
        losses[i] = total_loss.item()
    net.train()
    return losses.mean()

optimizer = optim.AdamW(net.parameters(), lr=0.001)
#torch.save(net.state_dict(), "./temp/prev.pth")

for iter in range(n_iter):
    # play
    run_episodes(root)
    # training
    for epoch in range(epochs):
        obs, target_pis, target_vs = get_batch()
        out_pi, out_v = net(obs)
        l_pi = loss_pi(target_pis, out_pi)
        l_vs = loss_v(target_vs, out_v)
        total_loss = l_pi + l_vs
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if iter % 10 == 0:
        losses = estimate_loss()
        print(f"iter: {iter} loss: {losses}")

#torch.save(net.state_dict(), "./temp/new.pth")
#stats = run_test(root)
#print(f"rewards for prev model: {stats['prev']} - new model: {stats['new']}")
    

env.close()






