from collections import defaultdict
from typing import Any, List

from mlagents_envs.environment import ActionTuple, UnityEnvironment
import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.utils.experience import Experience


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        env: UnityEnvironment,
        greedy_ratio: float,
        max_episode_steps: int,
        max_succeded_episodes: int,
        update_frequency: int,
        project_name: str,
        xp_name: str,
        log_dir: str,
    ):
        self.model = model
        self.env = env

        # Initialize experience history mappings
        self.agent_to_experiences = {}
        self.agent_to_lastobs = {}
        self.agent_to_lastaction = {}
        self.agent_to_lastreward = {}
        self.agent_to_cumreward = {}
        self.agent_to_loss = defaultdict(list)
        self.cumulative_rewards = []
        self.buffer = []
        self._succeded_episodes = 0
        self.episode_n_steps = []

        self._greedy_ratio = 1e-4
        self._max_episode_steps = max_episode_steps
        self._max_succeded_episodes = max_succeded_episodes
        self._update_frequency = update_frequency

        # Initialize logger
        wandb.init(project=project_name, name=xp_name, dir=log_dir)

    def _handle_terminals(self, terminal_steps: List[Any]):
        """For all agents with a terminal step."""
        for terminal_id in terminal_steps:
            # Create its last experience
            last_experience = Experience(
                obs=self.agent_to_lastobs[terminal_id].copy(),
                reward=terminal_steps[terminal_id].reward,
                done=not terminal_steps[terminal_id].interrupted,
                action=self.agent_to_lastaction[terminal_id].copy(),
                next_obs=terminal_steps[terminal_id].obs[0],
            )

            # Clear its last observation and action
            self.agent_to_lastobs.pop(terminal_id)
            self.agent_to_lastaction.pop(terminal_id)

            # Report the cumulative reward
            cumulative_reward = (
                self.agent_to_cumreward.pop(terminal_id)
                + terminal_steps[terminal_id].reward
            )
            self.cumulative_rewards.append(cumulative_reward)
            # Add the experience and the last experience to the buffer
            episode = self.agent_to_experiences.pop(terminal_id)
            episode.append(last_experience)
            self.buffer.append(episode)

    def _handle_decisions(self, decision_steps: List[Any]):
        """For all agents with a decision step."""
        for decision_id in decision_steps:
            # If the agent does not have an experience, create an empty one
            if decision_id not in self.agent_to_experiences:
                self.agent_to_experiences[decision_id] = []
                self.agent_to_cumreward[decision_id] = 0

            # If the agent requesting a decision has a "last observation"
            if decision_id in self.agent_to_lastobs:
                # Create an experience from the last observation and the
                # decision step
                exp = Experience(
                    obs=self.agent_to_lastobs[decision_id].copy(),
                    reward=decision_steps[decision_id].reward,
                    done=False,
                    action=self.agent_to_lastaction[decision_id].copy(),
                    next_obs=decision_steps[decision_id].obs[0],
                )

                # Update the experience of the agent and its cumulative reward
                self.agent_to_experiences[decision_id].append(exp)
                self.agent_to_cumreward[decision_id] += decision_steps[
                    decision_id
                ].reward

            # Store the observation and rewadr as the new "last ..."
            self.agent_to_lastobs[decision_id] = decision_steps[
                decision_id
            ].obs[0]
            self.agent_to_lastreward[decision_id] = decision_steps[
                decision_id
            ].reward

    def _store_actions(self, actions: np.array, decision_steps: List[Any]):
        """Store actions that were picked."""
        for agent_index, agent_id in enumerate(decision_steps.agent_id):
            self.agent_to_lastaction[agent_id] = actions[agent_index]

    def train(self, behavior_name: str, n_episodes: int):
        """Train an agent over multiple episodes."""
        for episode_index in tqdm(range(n_episodes)):
            self._train_episode(behavior_name)

            episode_losses = self.agent_to_loss.pop(0)
            mean_episode_loss = np.mean(episode_losses)
            episode_cumreward = self.cumulative_rewards[episode_index]
            episode_n_steps = self.episode_n_steps[episode_index]
            wandb.log(
                {
                    "episode/loss": mean_episode_loss,
                    "episode/cumreward": episode_cumreward,
                    "episode/n_steps": episode_n_steps,
                    "episode": episode_index,
                }
            )

            if self._succeded_episodes > self._max_succeded_episodes:
                print("Maximum number of succeded episodes reached.")
                break

    def _train_episode(self, behavior_name: str):
        """Train an agent over 1 episode."""
        self.env.reset()

        for step_index in range(self._max_episode_steps):
            # Get the decision steps and terminal steps of the agents
            decision_steps, terminal_steps = self.env.get_steps(behavior_name)

            # Check if there is a terminal step: the episode is over
            if (len(terminal_steps) > 0) and (step_index > 0):
                self._update_model(terminal_steps)
                self._handle_terminals(terminal_steps)
                self.episode_n_steps.append(step_index)
                break

            if step_index % self._update_frequency == 0:
                if step_index > 0:
                    self._update_model(decision_steps)

                self._handle_decisions(decision_steps)

                actions = self._get_actions(decision_steps)
                self._store_actions(actions, decision_steps)
                action_tuple = ActionTuple()
                action_tuple.add_discrete(actions)
                self.env.set_actions(behavior_name, action_tuple)

            # Perform a step in the simulation
            self.env.step()

            if step_index == self._max_episode_steps - 1:
                self._succeded_episodes += 1
                self.cumulative_rewards.append(self._max_episode_steps)
                self.episode_n_steps.append(self._max_episode_steps)

    def _update_model(self, agent_steps: List[Any]):
        """
        Generate an action for all the agents that requested a decision
        Compute the values for each action given the observation
        """
        for agent_id in agent_steps:
            last_obs = self.agent_to_lastobs[agent_id]
            last_action = self.agent_to_lastaction[agent_id]
            current_reward = agent_steps[agent_id].reward
            current_obs = agent_steps[agent_id].obs[0]

            last_camera_position = last_obs[:3]
            last_obstacle_position = last_obs[3:]
            current_camera_position = current_obs[:3]
            current_obstacle_position = current_obs[3:]

            self.model.store_transition(
                last_camera_position,
                last_obstacle_position,
                last_action.item(),
                current_reward,
                current_camera_position,
                current_obstacle_position,
            )

            agent_loss = self.model.learn()
            wandb.log({"step/loss": agent_loss})
            self.agent_to_loss[agent_id].append(agent_loss)

    def _get_actions(self, agent_steps: List[Any]) -> np.array:
        """
        Generate an action for all the agents that requested a decision
        Compute the values for each action given the observation
        """
        actions = []
        for agent_id in agent_steps:
            current_obs = agent_steps[agent_id].obs[0]
            current_camera_position = current_obs[:3]
            current_obstacle_position = current_obs[3:]

            agent_action = self.model.act(
                current_camera_position,
                current_obstacle_position,
                self._greedy_ratio,
            )
            actions.append([agent_action])
        return np.array(actions)
