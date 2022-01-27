import torch
from torch import nn


class DQN(nn.Module):
    """Simple Deep Q-Network.

    :param n_actions: number of discrete actions available in the env.
    """

    def __init__(self, n_actions: int):
        super().__init__()

        self.obstacle_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.camera_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, n_actions)
        )

    def forward(
        self, obstacle_state: torch.Tensor, camera_state: torch.Tensor
    ) -> torch.Tensor:
        obstacle_out = self.obstacle_fc(obstacle_state)
        camera_out = self.camera_fc(camera_state)
        out = self.fuse_fc(torch.cat([obstacle_out, camera_out], axis=1))

        return out
