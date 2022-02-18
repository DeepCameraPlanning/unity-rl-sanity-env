from mlagents_envs.environment import UnityEnvironment
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.models.dqn_module import DQNModule
from src.utils.progress_bar import LogBar

def infer(config: DictConfig):
    """TODO"""
    # env = UnityEnvironment(config.env.path, no_graphics=not config.env.display)
    env = UnityEnvironment(config.env.path, no_graphics=not config.env.display, additional_args=["-batchmode"])

    env.reset()
    behavior_name = list(env.behavior_specs)[0]

    # # Initialize callbacks
    # wandb_logger = WandbLogger(
    #     name=config.xp_name,
    #     project=config.project_name,
    #     offline=config.log_offline,
    # )
    checkpoint = ModelCheckpoint(
        monitor="step/loss",
        mode="min",
        save_last=True,
        dirpath=config.model.checkpoint_dir,
        filename=config.xp_name + "-{epoch}",
    )
    progressbar = LogBar()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model = DQNModule(
        batch_size=config.compnode.batch_size,
        lr=config.model.learning_rate,
        env=env,
        behavior_name=behavior_name,
        n_actions=config.env.num_actions,
        gamma=config.model.gamma,
        sync_rate=config.model.sync_rate,
        replay_size=config.model.replay_size,
        eps_last_frame=config.model.eps_last_frame,
        eps_start=config.model.eps_start,
        eps_end=config.model.eps_end,
        episode_length=config.model.episode_length,
        lr_reduce_rate=config.model.lr_reduce_rate,
        weight_decay=config.model.weight_decay,
        run_type=config.run_type,
        max_episodes=config.model.n_episodes,
    )

    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        # callbacks=[lr_monitor, checkpoint],
        # logger=wandb_logger,
        # log_every_n_steps=5,
        max_steps=config.model.n_episodes*config.model.episode_length,
    )
    # I know it's super ugly 
    # but it seems the wandblogger doesn't work in the test mode
    # TODO: need to figure out why
    wandb.init(project=config.project_name, name=config.xp_name, dir=config.log_dir)
    
    trainer.test(model, ckpt_path=config.model.checkpoint_path)

    env.close()
