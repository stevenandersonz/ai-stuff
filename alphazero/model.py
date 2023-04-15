import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_with_temperature(x, temperature=1.0):
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
        self.linear1 = nn.Linear(n_obs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(.2) 
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head1 = nn.Linear(256, 64)
        self.value_head2 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        # Policy head
        p = F.softmax(self.policy_head(x), dim=1)
        # Value head
        v = F.relu(self.value_head1(x))
        v = self.value_head2(v)
        return p, v
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(x)
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    def save(self, name):
        torch.save(self.state_dict(), f"./temp/{name}.pth")
    def load(self, name):
        m = AlphaZeroNet(4, 2)
        m.load_state_dict(torch.load(f"./temp/{name}.pth"))
        m.to("cuda")
        return m