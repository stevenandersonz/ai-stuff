import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from model import AlphaZeroNet
from collections import namedtuple
import copy
from mcts import mcts, Node

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
initial_state, _ = env.reset()
n_actions = env.action_space.n
m = AlphaZeroNet(len(initial_state), num_actions=n_actions)
s0 = env.get_snapshot()
root = Node(None, None, observation=initial_state, terminated=False, snapshot=s0)
total_rewards = 0
n_episodes = 100
n_sim = 25
n_iter = 50
eval_iter = 20
v_resign = None
temp = 1
epochs=10
batch_size=4

def run_episodes(root):
    for _ in range(n_episodes):
        mcts(root, m,env, n_sim)
        env.load_snapshot(s0)
        while True:
            action_prob = get_action_prob(temperature=1)
            action = np.random.choice(len(action_prob), p=action_prob)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
            plays.append([observation, action_prob, reward])

def print_tree(node, depth=0):
    if node is None:
        return
    indent = ' ' * depth
    print(f"{indent}  v:{node.N} w:{node.Q:.4f} t:{node.terminated} a:{node.action} o:{node.observation}")
    for child in node.children:
        print_tree(child, depth + 2)

def run_test(root,model):
    rewards = 0
    env.load_snapshot(s0)
    m.load_state_dict(torch.load(f"./temp/{model}.pth"))
    mcts(root,m, env,10)
    #print_tree(root)
    env.load_snapshot(s0)
    while True:
        action_prob, node = get_action_prob(temperature=0)
        action = np.random.choice(len(action_prob), p=action_prob)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        rewards += reward
        root = node 
        if not root.children:
            snap = env.get_snapshot()
            mcts(root,m,env, 100)
            print("rebuilding tree")
            env.load_snapshot(snap)

    print(f"reward for {model} model is {rewards}")

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
    m.eval()
    losses = torch.zeros(eval_iter)
    for i in range(eval_iter):
        x, target_pis, target_vs = get_batch()
        pi, v = m(x)
        l_pi = loss_pi(target_pis, pi)
        l_vs = loss_v(target_vs, v)
        total_loss = l_pi + l_vs 
        losses[i] = total_loss.item()
    m.train()
    return losses.mean()

optimizer = optim.AdamW(m.parameters(), lr=0.001)
torch.save(m.state_dict(), "./temp/prev.pth")

for iter in range(n_iter):
    # play
    run_episodes(root)
    # training
    for epoch in range(epochs):
        obs, target_pis, target_vs = get_batch()
        out_pi, out_v = m(obs)
        l_pi = loss_pi(target_pis, out_pi)
        l_vs = loss_v(target_vs, out_v)
        total_loss = l_pi + l_vs
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if iter % 10 == 0:
        losses = estimate_loss()
        print(f"iter: {iter} loss: {losses}")

torch.save(m.state_dict(), "./temp/new.pth")
#run_test(root,"new")
    

env.close()






