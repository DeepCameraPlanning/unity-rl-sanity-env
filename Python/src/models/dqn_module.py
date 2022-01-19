from collections import OrderedDict
from typing import List, Tuple

from mlagents_envs.environment import UnityEnvironment
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import DistributedType
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.models.modules.DQN import DQN
from src.models.memory import ReplayBuffer, RLDataset
from src.models.agent import Agent
import time
# batch = states, actions, rewards, dones, next_states
BatchTuple = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class DQNModule(LightningModule):
    """Basic DQN Model.

    :param batch_size: size of the batches.
    :param lr: learning rate.
    :param env: Unity environment.
    :param behavior_name: name of the unity ml-agent behavior.
    :param n_actions: size of the action space.
    :param gamma: discount factor.
    :param sync_rate: how many frames do we update the target network.
    :param replay_size: capacity of the replay buffer.
    :param warm_start_size: how many samples do we use to fill our buffer at
        the start of training.
    :param eps_last_frame: what frame should epsilon stop decaying.
    :param eps_start: starting value of epsilon.
    :param eps_end: final value of epsilon.
    :param episode_length: max length of an episode.
    :param warm_start_steps: max episode reward in the environment.
    :param max_episodes: maximum number of episodes to run.
    :param lr_reduce_rate: learning rate reduction rate.
    :param weight_decay: weight decay value.
    :param run_type: train or infer.
    """

    def __init__(
        self,
        batch_size: int,
        lr: float,
        env: UnityEnvironment,
        behavior_name: str,
        n_actions: int,
        gamma: float,
        sync_rate: int,
        replay_size: int,
        warm_start_size: int,
        eps_last_frame: int,
        eps_start: float,
        eps_end: float,
        episode_length: int,
        warm_start_steps: int,
        max_episodes: int,
        lr_reduce_rate: float,
        weight_decay: float,
        run_type: str,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        self._sync_rate = sync_rate
        self._replay_size = replay_size
        self._warm_start_size = warm_start_size
        self._eps_last_frame = eps_last_frame
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._episode_length = episode_length
        self._warm_start_steps = warm_start_steps
        self._lr_reduce_rate = lr_reduce_rate
        self._weight_decay = weight_decay

        self.env = env

        self.net = DQN(n_actions)
        self.target_net = DQN(n_actions)
        self.criterion = nn.MSELoss()

        self.buffer = ReplayBuffer(self._replay_size)
        self.agent = Agent(self.env, self.buffer, n_actions, behavior_name)
        self.total_reward = 0
        self.cumreward = 0
        self.episode_losses = []
        self.episode_index = 0
        self.max_episodes = max_episodes
        self.step_index = 0
        self.initTime=time.perf_counter()
        self.nowTime=time.perf_counter()


        if run_type == "train":
            self.populate(self._warm_start_steps)

    def populate(self, steps: int) -> None:
        """
        Carries out several random steps through the environment to initially
        fill up the replay buffer with experiences (on CPU).

        :param steps: number of random steps to populate the buffer with.
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0, device="cpu")

    def forward(
        self, obstacle_state: torch.Tensor, camera_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each
        action as an output.

        :param obstacle_state: obstacle state.
        :param camera_state: camera state.
        :return: q values.
        """
        output = self.net(obstacle_state, camera_state)
        return output

    def dqn_mse_loss(self, batch: BatchTuple) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        :param batch: current mini batch of replay data.
        :return: loss.
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states[:, :3], states[:, 3:])
            .gather(1, actions.unsqueeze(-1))
            .squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(
                next_states[:, :3], next_states[:, 3:]
            ).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

            expected_state_action_values = (
                next_state_values * self._gamma + rewards
            )

        loss = self.criterion(
            state_action_values, expected_state_action_values
        )

        return loss

    def training_step(self, batch: BatchTuple, nb_batch: int) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay
        buffer. Then calculates loss based on the minibatch recieved.

        :param batch: current mini batch of replay data.
        :param nb_batch: batch number.
        :return: training loss and log metrics.
        """
        device = self.get_device(batch)
        epsilon = max(
            self._eps_end,
            self._eps_start - self.global_step + 1 / self._eps_last_frame,
        )

        # Step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.cumreward += reward

        # Calculates training loss
        loss = self.dqn_mse_loss(batch)
        if self.trainer._distrib_type in {
            DistributedType.DP,
            DistributedType.DDP2,
        }:
            loss = loss.unsqueeze(0)
        self.episode_losses.append(loss.detach())

        # Soft update of target network
        if self.global_step % self._sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        self.nowTime=time.perf_counter()
        # second condition 
        if done or self.step_index==2000:
            logs = {
                "episode/cumreward": torch.tensor(self.cumreward).to(device),
                "episode/loss": torch.tensor(self.episode_losses).to(device),
                "episode": torch.tensor(self.episode_index).to(device),
                "episode/time": self.nowTime - self.initTime,
            }
            self.cumreward = 0
            self.episode_losses = []
            self.episode_index += 1
            self.step_index = 0
        else:
            logs = {
                "step/reward": torch.tensor(reward).to(device),
                "step/loss": self.episode_losses[-1],
            }
            self.step_index += 1

        outputs = OrderedDict({"loss": loss, "logs": logs})
        return outputs

    def training_epoch_end(self, outputs):
        """Log step or episode metrics."""
        for out in outputs:
            logs = out["logs"]
            for log_name, log_value in logs.items():
                self.log(
                    log_name,
                    log_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

    def test_step(self, batch: BatchTuple, nb_batch: int):
        """Infer the model."""
        device = self.get_device(batch)
        epsilon = 0

        # Step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.cumreward += reward

    def on_test_batch_start(
        self, batch: BatchTuple, batch_idx: int, unused=0
    ) -> int:
        """Stop the test if the number of episode is reached."""
        print(
                f"[Episode {self.episode_index}/{self.max_episodes}] ",
                f"Cumreward: {self.cumreward}",
            )
        if self.episode_index == self.max_episodes:
            print(
                f"[Episode {self.episode_index}/{self.max_episodes}] ",
                f"Cumreward: {self.cumreward}",
            )
            self.cumreward = 0
            self.episode_index += 1
            return -1

        return 0

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(
            self.net.parameters(),
            lr=self._lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self._weight_decay,
            amsgrad=False,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self._lr_reduce_rate
        )
        return {"optimizer": optimizer, "scheduler": scheduler}

    def __dataloader(self) -> DataLoader:
        """
        Initialize the Replay Buffer dataset used for retrieving experiences.
        """
        dataset = RLDataset(self.buffer, self._episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def test_dataloader(self) -> DataLoader:
        """Generate an "empty" dataloader to start the test."""
        return DataLoader(torch.ones(int(1e5), dtype=bool))

    def get_device(self, batch: BatchTuple) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
