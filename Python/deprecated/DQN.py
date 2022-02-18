import os
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, action_size):
        """
        I first try to use occupancy map, but the training result is so bad.
        Thus I replace it with a more intuitive input : position of obstacle.
        Input : obstacle position (x, y, z), camera position (x, y, z), both
            on local space.
        """

        super(Model, self).__init__()

        self.env_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.position_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )

        self.C_value = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_size)
        )

    def forward(self, env, position):
        # env = self.env_encoder(env).view(batch_size, -1)
        # env = self.env_fc(env)

        env = self.env_fc(env)
        pos = self.position_fc(position)

        return self.C_value(torch.cat([env, pos], axis=1))


class DQN(object):
    def __init__(self, config: DictConfig):
        self.config = config

        self.eval_net = Model(config.action_space)
        self.target_net = Model(config.action_space)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = []

        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            amsgrad=False,
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.reduce_rate
        )
        self.criterion = nn.MSELoss()

        self.checkpoint_path = os.path.join(
            config.checkpoint_dir, config.checkpoint_filename
        )
        self.memory_path = os.path.join(
            config.checkpoint_dir, config.memory_filename
        )
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        if self.config.load_checkpoint:
            # Load eval and target checkpoints
            self.load_checkpoint(self.eval_net)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # Load a save memory
            if os.path.exists(self.memory_path):
                self.memory = list(
                    np.load(self.memory_path, allow_pickle=True)
                )

    def store_transition(
        self,
        last_camera_position: Tuple[float, float, float],
        last_obstacle_position: Tuple[float, float, float],
        last_action: int,
        current_reward: int,
        current_camera_position: Tuple[float, float, float],
        current_obstacle_position: Tuple[float, float, float],
    ):
        """Store transition between 2 states."""
        if self.memory_counter < self.config.memory_capacity:
            self.memory.append(
                [
                    last_camera_position,
                    last_obstacle_position,
                    last_action,
                    current_reward,
                    current_camera_position,
                    current_obstacle_position,
                ]
            )
        else:
            index = self.memory_counter % self.config.memory_capacity
            self.memory[index] = [
                last_camera_position,
                last_obstacle_position,
                last_action,
                current_reward,
                current_camera_position,
                current_obstacle_position,
            ]
        self.memory_counter += 1

    def learn(self) -> float:
        """Learning step of the DQN and return the loss value."""
        if self.learn_step_counter % self.config.Q_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

            np.save(self.memory_path, self.memory)
            self.save_checkpoint(self.eval_net)

            self.scheduler.step(self.learn_step_counter // 100)

        if self.memory_counter < self.config.memory_capacity:
            return 0

        self.learn_step_counter += 1

        sample_index = np.random.choice(
            self.config.memory_capacity, self.config.batch_size
        )

        batch_memory = np.array(self.memory, dtype=object)[sample_index, :]

        batch_env = []
        batch_pos = []
        batch_action = []
        batch_reward = []
        batch_next_env = []
        batch_next_pos = []

        for i in range(self.config.batch_size):
            batch_env.append(batch_memory[i][0])
            batch_pos.append(batch_memory[i][1])
            batch_action.append(batch_memory[i][2])
            batch_reward.append(batch_memory[i][3])
            batch_next_env.append(batch_memory[i][4])
            batch_next_pos.append(batch_memory[i][5])

        batch_env = torch.FloatTensor(batch_env)
        batch_pos = torch.FloatTensor(batch_pos)
        batch_action = torch.LongTensor(batch_action).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
        batch_next_env = torch.FloatTensor(batch_next_env)
        batch_next_pos = torch.FloatTensor(batch_next_pos)

        q_eval = self.eval_net(batch_env, batch_pos).gather(1, batch_action)
        q_next = self.target_net(batch_next_env, batch_next_pos).detach()
        q_target = batch_reward + self.config.GAMMA * q_next.max(1)[0].view(
            self.config.batch_size, 1
        )
        loss = self.criterion(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()

    def act(
        self,
        obstacle_position: Tuple[float, float, float],
        camera_position: Tuple[float, float, float],
        greedy_ratio: float,
    ) -> int:
        """Compute the next predicted action."""
        if np.random.rand() < greedy_ratio:
            return np.random.randint(0, self.config.action_space)

        obstacle_position = torch.FloatTensor(obstacle_position).unsqueeze(0)
        camera_position = torch.FloatTensor(camera_position).unsqueeze(0)

        C_value = self.eval_net(obstacle_position, camera_position)
        C_value = C_value.detach().numpy()

        return np.argmax(C_value)

    def save_checkpoint(self, model: nn.Module):
        """
        Save a model checkpoint with:
            - `model_state_dict`
            - `optimizer_state_dict`
            - `scheduler_state_dict`
            - `learn_step_count`
        """
        save_path = self.checkpoint_path
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "learn_step_count": self.learn_step_counter,
            },
            save_path,
        )

    def load_checkpoint(self, model: nn.Module, only_model: bool = False):
        """
        Load a model checkpoint with:
            - `model_state_dict`
            - `optimizer_state_dict`
            - `scheduler_state_dict`
            - `learn_step_count`
        If `only_model`, load only `model_state_dict`.
        """
        load_path = self.checkpoint_path
        checkpoint = torch.load(load_path)

        if only_model:
            model.load_state_dict(checkpoint["model_state_dict"])

        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.learn_step_counter = checkpoint["learn_step_count"]


# class Model(nn.Module):
#     def __init__(self, action_size):
#         '''
#         I first try to use occupancy map, but the training result is so bad.
#         Thus I replace it with a more intuitive input : position of cube.
#         Input : cube position (x, y, z), camera position (x, y, z), both on local space
#         '''

#         super(Model, self).__init__()

#         self.env_encoder = nn.Sequential(
#             nn.Conv3d(in_channels=1, out_channels=8,  kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         self.env_fc = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU()
#         )

#         self.position_fc = nn.Sequential(
#             nn.Linear(4, 64),
#             nn.ReLU(),
#         )

#         self.C_value = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_size)
#         )

#     def forward(self, env, position):
#         batch_size = env.size(0)
#         env = self.env_encoder(env).view(batch_size, -1)
#         env = self.env_fc(env)

#         pos = self.position_fc(position)

#         return self.C_value(torch.cat([env, pos], axis=1))