import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 10, flow="target_to_source")
        self.conv_succ1 = GCNConv(10, 10, flow="target_to_source")
        self.conv_succ3 = GCNConv(10, 10, flow="target_to_source")
        self.conv_pred1 = GCNConv(10, 10)

        self.conv_probs = GCNConv(10, 1, flow="target_to_source")
        self.do_nothing = Linear(10, 1)
        self.value = Linear(10, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ2(x, edge_index)
        x = F.relu(x)
        x = self.conv_pred1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ3(x, edge_index)
        x = F.relu(x)

        probs = self.conv_output(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs, prob_nothing), dim=0)

        return probs, v
