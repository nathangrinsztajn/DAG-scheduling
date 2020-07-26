import torch
from env import CholeskyTaskGraph
import networkx as nx
from torch_geometric.utils.convert import to_networkx

import pydot
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np

# model = torch.load("/home/ngrinsztajn/HPC/runs/Apr20_05-12-44_chifflot-4.lille.grid5000.fr/model.pth")
# model = torch.load("/home/ngrinsztajn/HPC/runs/Apr20_01-30-52_chifflot-4.lille.grid5000.fr/model.pth")
# model = torch.load('/home/nathan/PycharmProjects/HPC/runs/Jun26_00-59-47_nathan-Latitude-7490/model.pth') # n=4




n = 8
G = 3
S = 2
T = 8
C = 11
P = 4

durations = [C, S, T, G]


def duration(x):
    tasktype = x[0]
    if tasktype == 0:
        duration = C
    if tasktype == 1:
        duration = S
    if tasktype == 2:
        duration = T
    if tasktype == 3:
        duration = G
    return duration


def start_time(time, makespan):
    return makespan - time


def get_data(Processed, time):
    processed = {}
    for k, v in Processed.items():
        processed[k] = [int(v[0]), int(time - v[1])]
    Processed = processed

    # makespan should be dicrete and durations should be discretized
    makespan = Processed[(0, 1, 0, 0)][1]
    current_times = [[makespan] * P]
    data = np.ones((P, makespan)) * (-1)
    compl_data = [[] for _ in range(P)]
    # data = [[]*P]
    for x, sched in Processed.items():
        tasktype = x[0]
        pr = sched[0]
        s_time = start_time(sched[1], makespan)
        e_time = s_time + duration(x)
        data[pr, s_time:e_time] = tasktype
        # print('process', pr)
        if tasktype == 0:
            compl_data[pr].insert(0, (x[1]))
        elif tasktype == 1:
            compl_data[pr].insert(0, (x[1], x[2]))
        elif tasktype == 2:
            compl_data[pr].insert(0, (x[1], x[2]))
        else:
            compl_data[pr].insert(0, (x[1], x[2], x[3]))
        # print('here', len(compl_data[0]), len(compl_data[1]), len(compl_data[2]), len(compl_data[3]))

    return data, compl_data


def visualize_schedule(env, figsize=(80, 30), fig_file=None, flip=True):
    Processed = env.processed
    time = env.time
    data, compl_data = get_data(Processed, time)
    if flip:
        data = data[-1::-1, :]
        compl_data = compl_data[-1::-1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_aspect(1)

    def avg(a, b):
        return (a + b) / 2.0

    for y, row in enumerate(data):
        # for x, col in enumerate(row):
        x = 0
        i = 0
        indices_in_row = compl_data[y]
        while x < len(row):
            col = row[x]
            if col != -1:
                shift = durations[int(col)]
                indices = indices_in_row[i]
            else:
                x = x + 1
                continue
            x1 = [x, x + shift]
            y1 = np.array([y, y])
            y2 = y1 + 1
            if col == 0:
                plt.fill_between(x1, y1, y2=y2, facecolor='green', edgecolor='Black')
                plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), 'C({})'.format(indices),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=30)

            if col == 1:
                plt.fill_between(x1, y1, y2=y2, facecolor='red', edgecolor='Black')
                plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "S{}".format(indices),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=30)
            if col == 2:
                plt.fill_between(x1, y1, y2=y2, facecolor='orange', edgecolor='Black')
                plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "T{}".format(indices),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=30)
            if col == 3:
                plt.fill_between(x1, y1, y2=y2, facecolor='yellow', edgecolor='Black')
                plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "G{}".format(indices),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=30)
            x = x + shift
            i = i + 1

    plt.ylim(P, 0)
    plt.xlim(-1e-3, data.shape[1] + 1e-3)
    plt.xticks(fontsize=50)
    if fig_file != None:
        plt.savefig(fig_file)
    return

# # ns = list(range(1, 15))
# # ps = list(range(1, 10))
# ps = [4]
# rewards = []
# times = []
# critical_path = []
# total_work_normalized = []
#
# for p in ps:
#     env = CholeskyTaskGraph(8, p, 1)
#     print(len(env.task_data.x))
#     observation = env.reset()
#     done = False
#
#     while not done:
#         policy, value = model(observation)
#         # action_raw = torch.multinomial(policy, 1).detach().cpu().numpy()
#         action_raw = policy.argmax().detach().cpu().numpy()
#         ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
#         # action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0][0]
#         action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]
#         observation, reward, done, info = env.step(action)
#     print(reward)
#     print(env.time)
#     rewards.append(reward)
#     times.append(env.time)
#     critical_path.append(env.critic_path_duration)
#     total_work_normalized.append(env.total_work_normalized)
# visualize_schedule(env.processed, figsize=(200, 100),fig_file = '/home/ngrinsztajn/HPC/img/sched_{}.pdf'.format(int(env.time)))

# ns = list(range(1, 15))
# ps = list(range(1, 10))

# temps = [1, 10, 20]
# n_sample = 25
# times = np.zeros((len(temps), n_sample))
# for i, temp in enumerate(temps):
#     for j in range(n_sample):
#         env = CholeskyTaskGraph(8, 4, 1)
#         print(len(env.task_data.x))
#         observation = env.reset()
#         done = False
#
#         while not done:
#             policy, value = model(observation)
#             new_policy = policy ** temp / ((policy ** temp).sum())
#             action_raw = torch.multinomial(new_policy, 1).detach().cpu().numpy()[0]
#     #         action_raw = policy.argmax().detach().cpu().numpy()
#             ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
#             # action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0][0]
#             action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]
#             observation, reward, done, info = env.step(action)
#         print(reward)
#         print(env.time)
#         times[i, j] = env.time

# plt.title('time distribution')
# for i, temp in enumerate(temps):
#     sns.distplot(times[i, :], label='temperature: {}'.format(temp))
# # plt.vlines(times[7], 0, 0.07, label='agent', colors='r')
# # plt.ylim(0, 0.07)
# plt.xlabel('time')
# plt.legend()
# plt.savefig('/home/ngrinsztajn/HPC/img/distrib_agent_163.pdf')

# def decode(env, model, greedy=False, sampling=False, temperature=None, n_traj_BS=None):
#     if n_traj_BS is not None:
#         # implement Beam-Search decoding
#
#         trajs = np.
#         current_probs = np.zeros(n_traj_BS)
#         observation = env.reset()