import numpy as np
import torch

from .vectorized_env import *

def evaluate(actor_critic, env, config_enhanced, device):

    eval_envs = VectorEnv(make_envs(env, config_enhanced["world_settings"]), config_enhanced["num_processes_eval"])
    eval_envs.reset()

    eval_episode_rewards = []
    reward_dic = {}

    obs = eval_envs.reset()
    obs = torch.tensor(obs, device=device, dtype=torch.float32)

    while len(reward_dic) < len(eval_envs.envs):
        with torch.no_grad():
            _, action, _ = actor_critic.act(obs, deterministic=True)
        actions = action.squeeze(-1).detach().cpu().numpy()
        # Observe reward and next obs
        obs, reward, done, infos = eval_envs.step(actions)
        obs = torch.tensor(obs, device=device, dtype=torch.float32)

        for k, info in enumerate(infos):
            if 'episode' in info.keys():
                if k not in reward_dic:
                    reward_dic[k] = info['episode']['r']
                    eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()
    mean_reward = np.mean(eval_episode_rewards)

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), mean_reward))
    return mean_reward
