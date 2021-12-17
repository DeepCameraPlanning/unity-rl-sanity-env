from mlagents_envs.environment import UnityEnvironment
from omegaconf import DictConfig

from src.models.trainer import Trainer
from src.models.modules.DQN import DQN


def train(config: DictConfig):
    env = UnityEnvironment(config.env.path, no_graphics=not config.env.display)
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    model = DQN(config.model)
    trainer = Trainer(
        model=model,
        env=env,
        greedy_ratio=config.model.greedy_ratio,
        max_episode_steps=config.model.max_episode_steps,
        max_succeded_episodes=config.model.max_succeded_episodes,
        update_frequency=config.model.update_frequency,
        project_name=config.project_name,
        xp_name=config.xp_name,
        log_dir=config.log_dir,
    )
    trainer.train(behavior_name, config.model.n_episodes)

    env.close()
