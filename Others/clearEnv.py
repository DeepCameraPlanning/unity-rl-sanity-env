from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(config.env.path, no_graphics=not config.env.display)
env.reset()