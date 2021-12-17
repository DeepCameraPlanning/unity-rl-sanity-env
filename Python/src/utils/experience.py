from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    """
    An experience contains the data of one Agent transition.
        - Observation
        - Action
        - Reward
        - Done flag
        - Next Observation
    """

    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_obs: np.ndarray
