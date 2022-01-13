from collections import deque
from typing import NamedTuple, Tuple

import numpy as np
from torch.utils.data.dataset import IterableDataset


class Experience(NamedTuple):
    """
    An experience contains the data of one Agent transition.
        - State
        - Action
        - Reward
        - Done flag
        - Next state
    """

    state: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_state: np.ndarray


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn
    from them.

    :param capacity: size of the buffer.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience):
        """Add experience to the buffer.

        :param experience: tuple (state, action, reward, done, new_state).
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    :param buffer: replay buffer.
    :param sample_size: number of experiences to sample at a time.
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
