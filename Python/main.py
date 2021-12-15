from mlagents_envs.environment import UnityEnvironment

from Python.DQN import DQN, Config
from Python.trainer import Trainer

if __name__ == "__main__":
    env_path = "./sanity-env"
    env = UnityEnvironment(env_path)
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    model = DQN(Config())
    trainer = Trainer(model, env)
    trainer.train(behavior_name, 10000)

    env.close()
