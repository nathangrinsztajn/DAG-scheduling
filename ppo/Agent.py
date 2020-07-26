from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd

from math import sqrt
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

import random, os.path, math, glob, csv, base64, itertools, sys
import gym
from gym.wrappers import Monitor
from pprint import pprint

import os

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Agent(ABC):

    def __init__(self, config, env, actor_critic, writer=None):
        self.config = config
        self.actor_critic = actor_critic
        self.writer = writer

    @abstractmethod
    def update(self, rollouts):
        pass

    def training_batch(self, epochs, batch_size):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []
        best_reward_mean = -1000

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                policy, value = self.network(torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0))
                values[i] = value.detach().cpu().numpy()
                actions[i] = torch.multinomial(policy, 1).detach().cpu().numpy()
                observation, rewards[i], dones[i], _ = self.env.step(actions[i])

                if dones[i]:
                    observation = self.env.reset()

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.network(torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0))[
                    1].detach().cpu().numpy()[0][0]

            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # TO DO: use rewards for train rewards

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages, epoch=epoch)

            # Test it every 100 epochs
            if epoch % self.config['evaluate_every'] == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(
                    f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')
                if self.writer:
                    self.writer.add_scalar('mean_reward', round(rewards_test[-1].mean(), 2), epoch)

                if rewards_test[-1].mean() >= best_reward_mean:
                    best_reward_mean = rewards_test[-1].mean()
                    str_file = str(self.writer.get_logdir()).split('/')[1]
                    torch.save(self.network.state_dict(), os.path.join(str(self.writer.get_logdir()),
                                                                       'model.pth'))

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs - 1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()

        # Plotting
        r = pd.DataFrame(
            (itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))),
            columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');

        print(f'The trainnig was done over a total of {episode_count} episodes')

    def training_batch2(self, epochs, batch_size):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []
        best_reward_mean = -1000

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                policy, value = self.network(torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0))
                values[i] = value.detach().cpu().numpy()
                actions[i] = torch.multinomial(policy, 1).detach().cpu().numpy()
                observation, rewards[i], dones[i], _ = self.env.step(actions[i])

                if dones[i]:
                    observation = self.env.reset()

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.network(torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0))[
                    1].detach().cpu().numpy()[0][0]

            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # TO DO: use rewards for train rewards

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages, epoch=epoch)

            # Test it every 100 epochs
            if epoch % self.config['evaluate_every'] == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(
                    f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')
                if self.writer:
                    self.writer.add_scalar('mean_reward', round(rewards_test[-1].mean(), 2), epoch)

                if rewards_test[-1].mean() >= best_reward_mean:
                    best_reward_mean = rewards_test[-1].mean()
                    str_file = str(self.writer.get_logdir()).split('/')[1]
                    torch.save(self.network.state_dict(), os.path.join(str(self.writer.get_logdir()),
                                                                       'model.pth'))

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs - 1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()

        # Plotting
        r = pd.DataFrame(
            (itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))),
            columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');

        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, advantages, epoch=None):
        actions = F.one_hot(torch.tensor(actions, device=device), self.env.action_space.n)
        returns = torch.tensor(returns[:, None], dtype=torch.float, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=device)
        observations = torch.tensor(observations, dtype=torch.float, device=device)

        # reset
        self.network_optimizer.zero_grad()
        policies, values = self.network(observations)

        # MSE for the values
        loss_value = 1 * F.mse_loss(values, returns)
        if self.writer:
            self.writer.add_scalar('critic_loss', loss_value.data.item(), epoch)

        # Actor loss
        loss_policy = ((actions.float() * policies.log()).sum(-1) * advantages).mean()
        loss_entropy = - (policies * policies.log()).sum(-1).mean()
        loss_actor = - loss_policy - self.entropy_cost * loss_entropy
        if self.writer:
            self.writer.add_scalar('actor_loss', loss_actor.data.item(), epoch)

        total_loss = self.config["loss_ratio"] * loss_value + loss_actor
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.network_optimizer.step()
        return loss_value, loss_actor

    def evaluate(self, render=False):
        env = self.monitor_env if render else self.env
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
        reward_episode = 0
        done = False

        while not done:
            policy, _ = self.network(observation)
            action = torch.multinomial(policy, 1)
            observation, reward, done, info = env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
            reward_episode += reward

        env.close()
        return reward_episode

# ToDo : load training batch during GPU training
