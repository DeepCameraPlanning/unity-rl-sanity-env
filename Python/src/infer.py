from mlagents_envs.environment import UnityEnvironment
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.models.dqn_module import DQNModule


def infer(config: DictConfig):
    """TODO"""
    env = UnityEnvironment(config.env.path, no_graphics=not config.env.display)
    env.reset()
    behavior_name = list(env.behavior_specs)[0]

    model = DQNModule(
        batch_size=config.compnode.batch_size,
        lr=config.model.learning_rate,
        env=env,
        behavior_name=behavior_name,
        n_actions=config.env.num_actions,
        gamma=config.model.gamma,
        sync_rate=config.model.sync_rate,
        replay_size=config.model.replay_size,
        warm_start_size=config.model.warm_start_size,
        eps_last_frame=config.model.eps_last_frame,
        eps_start=config.model.eps_start,
        eps_end=config.model.eps_end,
        episode_length=config.model.episode_length,
        warm_start_steps=config.model.warm_start_steps,
        lr_reduce_rate=config.model.lr_reduce_rate,
        weight_decay=config.model.weight_decay,
        run_type=config.run_type,
    )

    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        max_epochs=config.model.n_episodes,
    )
    trainer.test(model, ckpt_path=config.model.checkpoint_path)

    env.close()
