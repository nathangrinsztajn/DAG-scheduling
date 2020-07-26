import torch
from env import CholeskyTaskGraph
import networkx as nx
from torch_geometric.utils.convert import to_networkx

import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np

env = CholeskyTaskGraph(8, 4, 1)

def print_available_actions(available_actions_list):
    """
    Prints all available actions nicely formatted..
    :return:
    """

    display_actions = '\n'.join(available_actions_list)
    print()
    print('Action out of Range!')
    print('Available Actions:\n{}'.format(display_actions))
    print()


for i_episode in range(1):
    print('Starting new game!')
    observation = env.reset()
    available_actions = env.ready_tasks
    t = 0

    while True:
        t += 1
        env.render()

        print('Available tasks: ', env.ready_tasks + [-1])
        action = input('Select action: ')
        try:
            action = int(action)

            if not action in env.ready_tasks + [-1]:
                raise ValueError

        except ValueError:
            print_available_actions(env.ready_tasks + [-1])
            continue

        observation, reward, done, info = env.step(action, render_before=True)

        # if save_images:
        #     # img = Image.fromarray(env.render(mode="return"), 'RGB')
        #     # img.save(os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t)))
        #     img = env.render(mode="return")
        #     fig = plt.imshow(img, vmin=0, vmax=255, interpolation='none')
        #     fig.axes.get_xaxis().set_visible(False)
        #     fig.axes.get_yaxis().set_visible(False)
        #     plt.savefig(os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t)))


        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    # if generate_gifs:
    #     print('')
    #     import imageio
    #
    #     with imageio.get_writer(os.path.join('images', 'round_{}.gif'.format(i_episode)), mode='I', fps=1) as writer:
    #
    #             for t in range(n_steps):
    #                 try:
    #
    #                     filename = os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t))
    #                     image = imageio.imread(filename)
    #                     writer.append_data(image)
    #
    #                 except:
    #                     pass

env.close()
time.sleep(10)