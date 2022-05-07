import gym
from gym.envs.registration import register


register(
    id="HalfCheetah-Reg-v2",
    entry_point="gym_env.mujoco:HalfCheetahEnvReg",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id="HalfCheetah-Opp-v2",
    entry_point="gym_env.mujoco:HalfCheetahEnvOpp",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


def make_env(env_name):
    env = gym.make(env_name)
    return env
