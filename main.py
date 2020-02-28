import os
import time
from collections import deque
from pprint import pprint
from tqdm import tqdm

from PPO.vectorized_env import VectorEnv, make_envs
from PPO.storage import RolloutStorage
from PPO import PPO

from PPO import utils
from PPO.policy import Policy

from PPO.evaluation import evaluate

from model import *
from env import CholeskyTaskGraph
from config import config_enhanced

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def main():
    print("Current config_enhanced is:")
    pprint(config_enhanced)
    writer.add_text("config", str(config_enhanced))

    torch.manual_seed(config_enhanced['seed'])
    torch.cuda.manual_seed_all(config_enhanced['seed'])

    use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available() and config_enhanced['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # torch.set_num_threads(1)
    if use_cuda:
        device = torch.device('cuda')
        print("using GPU")
    else:
        device = torch.device('cpu')
        print("using CPU")

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = torch.nn.DataParallel(model)

    env = CholeskyTaskGraph(**config_enhanced['world_settings'])
    envs = VectorEnv(env, config_enhanced["num_processes"])
    envs.reset()

    model = Net(**config_enhanced["network_parameters"])
    if config_enhanced["model_path"]:
        model.load_state_dict(torch.load(config_enhanced['model_path']))

    actor_critic = Policy(model, envs.action_space, config_enhanced)
    actor_critic = actor_critic.to(device)

    PPO_settings = config_enhanced['PPO_settings']
    agent = PPO(
        actor_critic,
        **PPO_settings)

    rollouts = RolloutStorage(config_enhanced['nbatch'], config_enhanced['num_processes'],
                              envs.envs[0].observation_space.shape, envs.action_space)

    obs = envs.reset()
    obs = torch.tensor(obs, device=device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        config_enhanced['num_env_steps']) // config_enhanced['nbatch'] // config_enhanced['num_processes']
    for j in range(num_updates):

        if config_enhanced['use_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, config_enhanced['network']['lr'])

        for step in tqdm(range(config_enhanced['nbatch'])):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step])
            actions = action.squeeze(-1).detach().cpu().numpy()

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(actions)
            obs = torch.tensor(obs, device=device)
            reward = torch.tensor(reward, device=device).unsqueeze(-1)
            done = torch.tensor(done, device=device)

            n_step = (j * config_enhanced['nbatch'] + step) * config_enhanced['num_processes']
            for info in infos:
                if 'episode' in info.keys():
                    reward_episode = info['episode']['r']
                    episode_rewards.append(reward_episode)
                    writer.add_scalar('reward', reward_episode, n_step)
                    writer.add_scalar('solved', int(info['episode']['length'] == envs.envs[0].max_steps))

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, config_enhanced["use_gae"], config_enhanced["gamma"],
                                 config_enhanced['gae_lambda'], config_enhanced['use_proper_time_limits'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        writer.add_scalar('value loss', value_loss, n_step)
        writer.add_scalar('action loss', action_loss, n_step)
        writer.add_scalar('dist_entropy', dist_entropy, n_step)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % config_enhanced['save_interval'] == 0
                or j == num_updates - 1):
            save_path = str(writer.get_logdir())
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(actor_critic, os.path.join(save_path, "model.pth"))

        if j % config_enhanced['log_interval'] == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, n_step,
                            int(n_step / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

        if (config_enhanced['evaluate_every'] is not None and len(episode_rewards) > 1
                and j % config_enhanced['evaluate_every'] == 0):
            eval_reward = evaluate(actor_critic, boxworld, config_enhanced, device)
            writer.add_scalar("eval reward", eval_reward, n_step)


if __name__ == "__main__":
    main()

