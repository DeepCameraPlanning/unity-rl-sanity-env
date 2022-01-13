from typing import Tuple

import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
import torch
from torch import nn

from src.models.memory import Experience, ReplayBuffer


class Agent:
    """Base Agent class handeling the interaction with the environment.

    :param env: training environment.
    :param replay_buffer: replay buffer storing experiences.
    """

    def __init__(
        self,
        env: UnityEnvironment,
        replay_buffer: ReplayBuffer,
        n_actions: int,
        behavior_name: str,
    ):
        self.behavior_name = behavior_name
        self.n_actions = n_actions
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self):
        """Resents the environment and updates the state."""
        self.env.reset()
        self.state, _, _ = self.get_obs()

    def get_obs(self) -> Tuple[np.array, float, bool]:
        """Get observations, reward and done."""
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            new_state = terminal_steps[0].obs[0]
            reward = terminal_steps[0].reward
            done = True
        else:
            new_state = decision_steps[0].obs[0]
            reward = decision_steps[0].reward
            done = False

        return new_state, reward, done

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        :param net: DQN network.
        :param epsilon: probability of taking a random action.
        :param device: current device.
        :return: action.
        """
        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)

        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state[:, :3], state[:, 3:])
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def update_state(self, action: int):
        """Update sate with an action."""
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]]))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float,
        device: str,
    ) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the
        environment.

        :param net: DQN network.
        :param epsilon: probability of taking a random action.
        :param device: current device.
        :return: reward, done.
        """
        action = self.get_action(net, epsilon, device)
        self.update_state(action)

        new_state, reward, done = self.get_obs()
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()

        return reward, done
