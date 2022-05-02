import gym


def make_env(args):
    env = gym.make(args.env_name)
    return env
