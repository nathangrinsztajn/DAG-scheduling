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
        self.conv_succ2 = GCNConv(10, 10, flow="target_to_source")
        self.conv_succ3 = GCNConv(10, 10, flow="target_to_source")
        self.conv_pred1 = GCNConv(10, 10)

        self.conv_probs = GCNConv(10, 1, flow="target_to_source")
        self.do_nothing = Linear(10, 1)
        self.value = Linear(10, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ2(x, edge_index)
        x = F.relu(x)
        x = self.conv_pred1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ3(x, edge_index)
        x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 64, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_succ3 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_pred1 = GCNConv(64, 64)

        self.conv_probs = GCNConv(64, 1, flow="target_to_source")
        self.do_nothing = Linear(64, 1)
        self.value = Linear(64, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        # x = self.conv_succ2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_pred1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_succ3(x, edge_index)
        # x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v
