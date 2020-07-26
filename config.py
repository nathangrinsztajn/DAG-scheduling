# import os
import multiprocessing
import numpy as np
#
# # Computing env
seed = 42
# cuda_deterministic = False
num_cores = multiprocessing.cpu_count()-1
print("using {} CPUs".format(num_cores))
num_processes = "num_cpu"
num_processes_eval = "num_cpu"
#

#
# # PPO
# clip_param = 0.2  # 0.1
# ppo_epoch = 5
# mini_batch_size = 16
# num_mini_batch = num_processes // mini_batch_size
# optimizer = "sgd"
optimizer = None

value_loss_coef = 0.5
entropy_coef = 0.005# 0.005
lr = 10 ** -2
eps = 10 ** -1
max_grad_norm = 10
acktr = False
alpha = 0.9
# use_clipped_value_loss = True
#
# # Training
# use_linear_lr_decay = True
gamma = 0.99
# use_gae = True
# gae_lambda = 0.95
# use_proper_time_limits = False
num_env_steps = 10 ** 9
trajectory_length = 40 # 256

#
# # logs
model_path = None
# model_path = "/home/ngrinsztajn/HPC/runs/Apr20_01-29-59_chifflot-4.lille.grid5000.fr/model.pth"
# model_path = "/home/nathan/PycharmProjects/HPC/runs/Apr20_01-11-05_nathan-Latitude-7490/model.pth"
evaluate_every = 10
save_interval = 10
log_interval = 10 ** 2 #4
#
#

A2C_settings = {
    "value_loss_coef": value_loss_coef,
    "entropy_coef": entropy_coef,
    "lr": lr,
    "eps": eps,
    "max_grad_norm": max_grad_norm,
    "acktr": acktr,
    "alpha": alpha
}

# PPO_settings = {
#     "clip_param": clip_param,
#     "ppo_epoch": ppo_epoch,
#     "num_mini_batch": num_mini_batch,
#     "value_loss_coef": value_loss_coef,
#     "entropy_coef": entropy_coef,
#     "lr": lr,
#     "eps": eps,
#     "max_grad_norm": max_grad_norm,
#     "use_clipped_value_loss": use_clipped_value_loss
# }
#
# config_enhanced = {
#     'cuda_deterministic': cuda_deterministic,
#     'evaluate_every': evaluate_every,
#     'gae_lambda': gae_lambda,
#     'gamma': gamma,
#     'log_interval': log_interval,
#     'model_path': model_path,
#     'nbatch': nbatch,
#     'num_cores': num_cores,
#     'num_env_steps': num_env_steps,
#     'num_processes': num_processes,
#     'num_processes_eval': num_processes_eval,
#     'PPO_settings': PPO_settings,
#     'save_interval': save_interval,
#     'seed': seed,
#     'use_gae': use_gae,
#     'use_linear_lr_decay': use_linear_lr_decay,
#     'use_proper_time_limits': use_proper_time_limits,
#     "world_settings": world_settings,
# }
#
agent = 'A2C'

# World
env_settings = {
    'n': 8,
    'node_types': np.array([1,1,1,1]).astype(int),
    'window': 1,
    'noise': True,
}
seed_env = None

network_parameters = {"input_dim": 11}

config_enhanced = {
    'model_path': model_path,
    'num_env_steps': num_env_steps,
    'num_processes': num_processes,
    'lr': lr,
    'eps': eps,
    'optimizer': optimizer,
    'entropy_coef': entropy_coef,
    'seed': seed,
    'gamma': gamma,
    'loss_ratio': value_loss_coef,
    "evaluate_every": evaluate_every,
    'trajectory_length': trajectory_length,
    'log_interval': log_interval,
    'env_settings': env_settings,
    'network_parameters': network_parameters,
    'agent': agent,
    'A2C_settings': A2C_settings,
    'seed_env': seed_env,
}
