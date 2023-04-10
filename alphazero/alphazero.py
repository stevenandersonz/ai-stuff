import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from model import AlphaZeroNet
from collections import namedtuple
import copy
from mcts import mcts, Node, ReplayBuffer

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

def get_action_prob(root, temperature=1):
    action_visits = [child.N for child in root.children]
    action_prob = torch.zeros(len(action_visits))
    
    if temperature == 0:
        best_action_idx = torch.argmax(torch.tensor(action_visits))
        action_prob[best_action_idx] = 1
        return action_prob, root.children[best_action_idx]
    
    counts = torch.tensor(action_visits) ** (1. / temperature)
    counts_sum = torch.sum(counts)
    action_prob = counts / counts_sum
    
    return np.array(action_prob)

def print_tree(node, depth=0):
    if node is None:
        return
    indent = ' ' * depth
    print(f"{indent}  v:{node.N} w:{node.Q:.4f} t:{node.terminated} a:{node.action} o:{node.observation}")
    for child in node.children:
        print_tree(child, depth + 2)

def loss_pi(targets, outputs):
    return  -torch.sum(targets * torch.log(outputs)) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

def get_batch(device):
    obs, pis, vs = replay_buffer.sample(batch_size) 
    obs = torch.tensor(obs)
    target_pis = torch.tensor(pis)
    target_vs = torch.tensor(vs)
    obs, target_pis, target_vs = obs.to(device), target_pis.to(device), target_vs.to(device)
    return obs, target_pis, target_vs
    
@torch.no_grad()
def estimate_loss(m):
    m.eval()
    losses = torch.zeros(eval_iter)
    for i in range(eval_iter):
        x, target_pis, target_vs = get_batch(device=device)
        pi, v = m(x)
        l_pi = loss_pi(target_pis, pi)
        l_vs = loss_v(target_vs, v)
        regularization_loss = 0.1 * sum(torch.norm(p, 2) ** 2 for p in m.parameters())
        total_loss = l_pi + l_vs + regularization_loss
        losses[i] = total_loss.item()
    avg_loss = losses.mean()
    m.train()
    return avg_loss 


def run_episode(m, training=True, debug=False):
    rewards=0
    root = Node(None, None, observation=initial_state, terminated=False, snapshot=s0)
    # for each episode lets load the game at the initial state
    # note: maybe each episode would require to start over
    env.load_snapshot(s0)
    while True:
        # tree without node (possible actions)
        # run mcts search to fill tree with actions as long as the game hasn't reach the end
        if not root.children:
            # copy current state as it will change when search is done
            state_before_search = env.get_snapshot()
            mcts(root, m,env, n_sim)
            env.load_snapshot(state_before_search)
            if debug:
                print_tree(root)
            
        action_prob = get_action_prob(root, temperature=1)
        action = np.random.choice(len(action_prob), p=action_prob)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards +=reward
        if terminated:
            break
        if training:
            replay_buffer.add(observation, action_prob, reward)
    return rewards

def train (m):
   for epoch in range(epochs):
        obs, target_pis, target_vs = get_batch(device)
        out_pi, out_v = m(obs)
        l_pi = loss_pi(target_pis, out_pi)
        l_vs = loss_v(target_vs, out_v)
        regularization_loss = 0.1 * sum(torch.norm(p, 2) ** 2 for p in m.parameters())
        total_loss = l_pi + l_vs + regularization_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

def pit (new_model, prev_model):

    rewards = {}
    env.load_snapshot(s0)
    rewards["prev_model"] = run_episode(prev_model, False)
    rewards["new_model"] = run_episode(new_model, False)

    # Collects total rewards from both models, then computes how much better is the new model based on total rewards
    #print(f"new model: {rewards['new_model']} old model {rewards['prev_model']}")
    diff =  rewards["new_model"] - rewards["prev_model"] 
    percent_better = (diff / rewards["prev_model"])    
    return percent_better

replay_buffer = ReplayBuffer(10000)
env = WithSnapshot(gym.make('CartPole-v1'))
n_actions = env.action_space.n

total_rewards = 0
n_episodes = 100
n_sim = 5
n_iter = 100
eval_iter = 100
temp = 1
epochs= 10
batch_size=256
m = AlphaZeroNet(4, num_actions=n_actions)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m.to(device)
optimizer = optim.AdamW(m.parameters(), lr=0.001)
initial_state, _ = env.reset()
s0 = env.get_snapshot()

def print_tree(x, hist=[]):
  if x.N != 0:
    print("%4d %-16s %8.4f %4s" % (x.N, str(hist), x.Q, x.P))
  for i,c in enumerate(x.children):
    print_tree(c,hist+[i])

#un_episode(m, False, True)

for i in range(n_iter):
    for e in range(n_episodes):
        run_episode(m)
    m.save("prev_m")
    train(m)

    if i%10==0:
        loss = estimate_loss(m)
        print(f"iter {i} - loss {loss}")
    prev_m = m.load("prev_m")
    frac_win = pit(m, prev_m)
    if frac_win < 0.6:
        #print("keeping old model")
        m = prev_m
    else:
        print(f"iter {i} - updates model")

env.close()