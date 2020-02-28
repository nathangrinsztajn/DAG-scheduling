import os
import multiprocessing

# Computing env
seed = 42
cuda_deterministic = False
num_cores = multiprocessing.cpu_count()-1
print("using {} CPUs".format(num_cores))
num_processes = num_cores
num_processes_eval = num_cores

# World
world_settings = {"n": 8, "p": 6, "window": 2}

# PPO
clip_param = 0.2  # 0.1
ppo_epoch = 5
mini_batch_size = 16
num_mini_batch = num_processes // mini_batch_size
value_loss_coef = 0.5
entropy_coef = 0.005
lr = 10 ** -4
eps = 10 ** -2
max_grad_norm = 10
use_clipped_value_loss = True

# Training
use_linear_lr_decay = True
gamma = 0.99
use_gae = True
gae_lambda = 0.95
use_proper_time_limits = False
nbatch = world_settings['n'] * world_settings['p']
num_env_steps = 10 ** 9

# logs
model_path = None
evaluate_every = 10
save_interval = 10
log_interval = 1


PPO_settings = {
    "clip_param": clip_param,
    "ppo_epoch": ppo_epoch,
    "num_mini_batch": num_mini_batch,
    "value_loss_coef": value_loss_coef,
    "entropy_coef": entropy_coef,
    "lr": lr,
    "eps": eps,
    "max_grad_norm": max_grad_norm,
    "use_clipped_value_loss": use_clipped_value_loss
}

config_enhanced = {
    'cuda_deterministic': cuda_deterministic,
    'evaluate_every': evaluate_every,
    'gae_lambda': gae_lambda,
    'gamma': gamma,
    'log_interval': log_interval,
    'model_path': model_path,
    'nbatch': nbatch,
    'num_cores': num_cores,
    'num_env_steps': num_env_steps,
    'num_processes': num_processes,
    'num_processes_eval': num_processes_eval,
    'PPO_settings': PPO_settings,
    'save_interval': save_interval,
    'seed': seed,
    'use_gae': use_gae,
    'use_linear_lr_decay': use_linear_lr_decay,
    'use_proper_time_limits': use_proper_time_limits,
    "world_settings": world_settings,
}

