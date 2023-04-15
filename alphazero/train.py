import gymnasium as gym
import numpy as np
from collections import namedtuple
import torch
import torch.optim as optim
import math
from model import AlphaZeroNet 
from copy import deepcopy
from random import shuffle

ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "terminated", "truncated"))

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
        return ActionResult(next_snapshot, observation, terminated, truncated)

n_iters = 100
epochs = 10
n_episode = 20
n_sim = 100
batch_size = 64
env = WithSnapshot(gym.make('CartPole-v1'))
initial_obs, _ = env.reset()
initial_env = env.get_snapshot()
n_actions = env.action_space.n
state_size = env.observation_space.shape[0]

print(f"# actions: {n_actions} - state size {state_size}")

m = AlphaZeroNet(state_size, n_actions)
m.to("cuda")

class Node():
    def __init__(self, prior, state=None, terminated=False, truncated=False, parent=None, snapshot=None, action=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.truncated= truncated
        self.parent = parent
        self.state = state
        self.terminated = terminated
        self.action=action
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
        value_score = child.value()
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
        snapshot, o,terminated, truncated = env.get_result(node.snapshot, a)
        new_node = Node(prob, o, terminated, truncated, node, snapshot, a)
        node.children[a] = new_node

def backpropagate(node, value):
    node.visit_count += 1
    node.value_sum += value
    if node.parent:
        backpropagate(node.parent, node.value_sum)

def mcts():
    root = Node(0, None)
    root.state = env.state
    root.snapshot = env.get_snapshot()
    x = torch.tensor(root.state, dtype=torch.float32).unsqueeze(0).to("cuda")
    action_prob, _ = m.predict(x)
    expand(root, action_prob)
    for _ in range(n_sim):
        node = select_child(root)
        if not node.terminated and not node.truncated:
            x = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0).to("cuda")
            action_prob, value = m.predict(x)
            expand(node, action_prob)
            backpropagate(node, value)
        else: 
            backpropagate(node, 0)
    env.load_snapshot(root.snapshot)
    return root


def run_episode():
    train_examples = []
    terminated = False
    truncated = False
    env.reset()
    while not terminated and not truncated: 
        root = mcts()
        action_prob = [0 for _ in range(n_actions)]
        for a, c in root.children.items():
            action_prob[a] = c.visit_count 
        action_prob = action_prob / np.sum(action_prob)
        action = root.select_action(temperature=0)
        obs, reward, terminated, truncated, _ = env.step(action)
        train_examples.append((root.state, action_prob, reward))

    return train_examples

def learn():
    m.save('cartpolev1-prev')
    print("rewards before training %d" % run_model())
    for i in range(1, n_iters + 1):

        print("{}/{}".format(i, n_iters))

        train_examples = []

        for eps in range(n_episode):
            obs ,_ = env.reset()
            iteration_train_examples = run_episode()
            train_examples.extend(iteration_train_examples)

        shuffle(train_examples)
        train(train_examples)
        percent_better = pit()
        if percent_better > 0.55:
            print("updating model...")
            m.save('cartpolev1-prev')


def train(examples):
    optimizer = optim.Adam(m.parameters(), lr=5e-4)
    pi_losses = []
    v_losses = []

    for epoch in range(epochs):
        m.train()

        batch_idx = 0

        while batch_idx < int(len(examples) / batch_size):
            sample_ids = np.random.randint(len(examples), size=batch_size)
            boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            boards = torch.tensor(np.array(boards).astype(np.float32))
            target_pis = torch.tensor(np.array(pis))
            target_vs = torch.tensor(np.array(vs).astype(np.float32))

            # predict
            boards = boards.contiguous().cuda()
            target_pis = target_pis.contiguous().cuda()
            target_vs = target_vs.contiguous().cuda()

            # compute output
            out_pi, out_v = m(boards)
            l_pi = loss_pi(target_pis, out_pi)
            l_v = loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

            pi_losses.append(float(l_pi))
            v_losses.append(float(l_v))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_idx += 1

        print(f"pi loss {np.mean(pi_losses)} - v loss {np.mean(v_losses)}")


def loss_pi(targets, outputs):
    loss = -(targets * torch.log(outputs)).sum(dim=1)
    return loss.mean()

def loss_v( targets, outputs):
    loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
    return loss

def run_model():
    initial_state, _ = env.reset()
    terminated = False
    total_reward = 0
    while not terminated: 
        root = mcts()
        env.load_snapshot(root.snapshot)
        action_prob = [0 for _ in range(n_actions)]
        for a, c in root.children.items():
            action_prob[a] = c.visit_count 
        action_prob = action_prob / np.sum(action_prob)
        action = root.select_action(temperature=0)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward

def pit ():
    rewards = {}
    m.save("cartpolev1-new")
    m.load("cartpolev1-prev")
    rewards["prev_model"] = run_model()
    m.load("cartpolev1-new")
    rewards["new_model"] = run_model()

    # Collects total rewards from both models, then computes how much better is the new model based on total rewards
    #print(f"new model: {rewards['new_model']} old model {rewards['prev_model']}")
    diff =  rewards["new_model"] - rewards["prev_model"] 
    percent_better = (diff / rewards["prev_model"])    
    print(f"before {rewards['prev_model']} - new {rewards['new_model']}  new is %{percent_better*100:.2f}")
    return percent_better

#learn()
m.load('cartpolev1-prev')
print(run_model())

# import time

# # start_time = time.time()
# # root = mcts()
# # end_time = time.time()
# # total_time = end_time - start_time
# # print("Total execution time: ", total_time, "seconds")

# start_time = time.time()
# for _ in range(100):
#     run_episode()
# end_time = time.time()
# total_time = end_time - start_time
# print("Total execution time: ", total_time, "seconds")


