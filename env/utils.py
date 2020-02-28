from torch_geometric.data import Data

import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np


durations_cpu = [24, 52, 57, 95, 0]
durations_gpu = [11, 8.3, 2.5, 3.3, 0]
durations_gpu2 = [12, 1, 3, 2, 0]
simple_durations = [1, 3, 3, 6, 0]

colors = {0: [0, 0, 0], 1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
          4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
          9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
          14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144],
          19: [0, 0, 117]}
color_normalized = {i: list(np.array(colors[i])/255) for i in colors}


class Task():

    def __init__(self, barcode):
        """
        task_type 0: POTRF 1:SYRK 2:TRSM 3: GEMMS
        """

        self.type = barcode[0]
        self.duration_cpu = durations_cpu[self.type]
        self.duration_gpu = durations_gpu[self.type]
        self.durations = [durations_cpu[self.type], durations_gpu[self.type]]
        self.barcode = barcode


class TaskGraph(Data):

    def __init__(self, x, edge_index, task_list):
        Data.__init__(self, x, edge_index)
        self.task_list = task_list

    def render(self, root=None):
        graph = self.data
        task_list = [t.barcode for t in self.task_list[graph.x]]

        pos = graphviz_layout(graph, prog='dot', root=root)
        # pos = graphviz_layout(G, prog='tree')
        node_color = [color_normalized[task[0]] for task in task_list]
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)

    def remove_nodes(self, node_list):
        mask_node = torch.logical_not(isin(self.x, node_list))
        self.x = self.x[mask_node]
        mask_edge = isin(self.edge_index[:, 0], node_list) or isin(self.edge_index[:, 1], node_list)
        self.edge_index = self.edge_index[torch.logical_not(mask_edge)]


class Node():
    def __init__(self, type):
        self.type = type


class Cluster():
    def __init__(self, node_types, communication_cost):
        """
        :param node_types:
        :param communication_cost: [(u,v,w) with w weight]
        """
        self.node_types = node_types
        self.node_state = np.zeros(len(node_types))
        self.communication_cost = communication_cost


    def render(self):
        edges_list = [(u, v, {"cost": w}) for (u, v, w) in self.communication_cost]
        colors = ["k" if node_type == 0 else "red" for node_type in self.node_types]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)


def succASAP(task, n):
    tasktype = task.type
    i = task.barcode[1]
    j = task.barcode[2]
    k = task.barcode[3]
    listsucc = []
    if tasktype == 0:
        if i < n:
            for j in range(i + 1, n + 1, 1):
                y = (2, i, j, 0)
                listsucc.append(Task(y))
        else:
            y = (4, 0, 0, 0)
            listsucc.append(Task(y))

    if tasktype == 1:
        if j < i - 1:
            y = (1, i, j + 1, 0)
            listsucc.append(Task(y))
        else:
            y = (0, i, 0, 0)
            listsucc.append(Task(y))

    if tasktype == 2:
        if i <= n - 1:
            for k in range(i + 1, j):
                y = (3, k, j, i)
                listsucc.append(Task(y))
            for k in range(j + 1, n + 1):
                y = (3, j, k, i)
                listsucc.append(Task(y))
            y = (1, j, i, 0)
            listsucc.append(Task(y))

    if tasktype == 3:
        if k < i - 1:
            y = (3, i, j, k + 1)
            listsucc.append(Task(y))
        else:
            y = (2, i, j, 0)
            listsucc.append(Task(y))

    return listsucc


def _add_task(dic_already_seen, list_to_process, task):
    if task.barcode in dic_already_seen:
        pass
    else:
        dic_already_seen[task.barcode] = len(dic_already_seen)
        list_to_process.append(task)


def _add_node(dic_already_seen, list_to_process, node):
    if node in dic_already_seen:
        pass
    else:
        dic_already_seen[node] = True
        list_to_process.append(node)


def compute_graph(n):
    root_nodes = []
    TaskList = {}
    EdgeList = []

    root_nodes.append(Task((0, 1, 0, 0)))
    TaskList[(0, 1, 0, 0)] = 0

    while len(root_nodes) > 0:
        task = root_nodes.pop()
        list_succ = succASAP(task, n)
        for t_succ in list_succ:
            _add_task(TaskList, root_nodes, t_succ)
            EdgeList.append((TaskList[task.barcode], TaskList[t_succ.barcode]))

    # embeddings
    embeddings = [k for k in TaskList]

    data = Data(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous())

    task_array = []
    for (k, v) in TaskList.items():
        task_array.append(Task(k))
    return TaskGraph(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous(), task_list=task_array)
    # return data, task_array


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def compute_sub_graph(data, root_nodes, window):
    """
    :param data: the whole graph
    :param root_nodes: list of node numbers
    :param window: the max distance to go down from the root nodes
    :return: the sub graph with nodes at distance less than h from root_nodes
    """

    already_seen = torch.zeros(data.num_nodes, dtype=torch.bool)
    already_seen[root_nodes] = 1
    edge_list = torch.tensor([[], []], dtype=torch.long)

    i = 0
    while len(root_nodes) > 0 and i < window:
        mask = isin(data.edge_index[0], root_nodes)
        list_succ = data.edge_index[1][mask]
        list_pred = data.edge_index[0][mask]

        edge_list = torch.cat((edge_list, torch.stack((list_pred, list_succ))), dim=1)

        list_succ = torch.unique(list_succ)

        list_succ = list_succ[already_seen[list_succ] == 0]
        already_seen[list_succ] = 1
        root_nodes = list_succ
        i += 1
    return Data(torch.nonzero(already_seen), edge_list)


def taskGraph2SLC(taskGraph, save_path):
    with open(save_path,"w") as file:
        file.write(str(len(taskGraph.task_list)))
        file.write('\n')
        for node, task in enumerate(taskGraph.task_list):
            line1 = str(node + 1) + " " + str(simple_durations[task.type]) + " 1"
            file.write(line1)
            file.write('\n')

            line2 = ""
            for n in taskGraph.edge_index[1][taskGraph.edge_index[0] == node]:
                line2 += str(n.item() + 1) + " 0 "
            line2 += "-1"
            file.write(line2)
            file.write('\n')
        # file.write("-1")

