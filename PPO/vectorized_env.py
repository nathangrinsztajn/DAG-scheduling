import copy


class VectorEnv:
    def __init__(self, env, n):
        self.envs = [copy.copy(env) for _ in range(n)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Call this only once at the beginning of training (optional):
    def seed(self, seeds):
        assert len(self.envs) == len(seeds)
        return tuple(env.seed(s) for env, s in zip(self.envs, seeds))

    # Call this only once at the beginning of training:
    def reset(self):
        l = [env.reset() for env in self.envs]
        return l

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return observations, rewards, dones, infos

    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()

