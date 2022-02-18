from typing import Tuple

import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
import torch
from torch import nn
from src.models.modules.modelUtils import versionControl
import src.utils.utils as utils

from src.models.memory import Experience, ReplayBuffer

class ObsData:
    """
    data structure for saving and vizing obs data
    """
    def __init__(self,
      state: np.array,
      reward: float,
      done: bool):
      self.reward = reward
      self.state  = state
      self.done   = done
    
    def print(self):
      print("obs.state: ", self.state)
      print("obs.reward: ", self.reward)
      print("isDone: ", self.done)
      
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
        # self.reset()

        # new observation
        self._obs={}
        # saved observation
        self.obs={}
        self.numAgent=-1
        self.reset()
        
    def allDone(self):
        for agentId in self.obs:
          if(not self.obs[agentId].done):
            return False
        return True
      
    def anyDone(self):
        """an OR operation of all done"""
        for agentId in self.obs:
          if(self.obs[agentId].done):
            return True
        return False
    
    def allRewards(self):
        """average of all rewards"""
        return np.mean([self.obs[k].reward for k in self.obs])
  
    # TODO: Xi
    # the func resets ALL agents in the environment
    # most efficient way should be to reset everyone single agent   
    def reset(self):
        """Resents the environment and updates the state."""
        self.env.reset()
        # self.state, _, _ = self.get_obs()
        self.numAgent = self.get_obs()
        self.obs=self._obs.copy()
        # print("numAgent: ", self.numAgent)
    
    def get_obs(self):
        """Get observations, reward and done."""
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        for t_idx in terminal_steps.agent_id:
            t_step = terminal_steps[t_idx]
            new_state, reward, agentID = t_step.obs[0], t_step.reward, t_step.agent_id
            self._obs[agentID] = ObsData(new_state, reward, True)
        
        for d_idx in decision_steps.agent_id:
            # agent step can re-occur in decision step even if terminal signal sent, 
            # ignored the newly decision step, but maybe a TODO to fix it in later work
            # e.g. the len(terminal_steps.agent_id) + len(decision_steps.agent_id) can be bigger than 4
            if d_idx in terminal_steps.agent_id:  
              continue
            d_step = decision_steps[d_idx]
            new_state, reward, agentID = d_step.obs[0], d_step.reward, d_step.agent_id
            self._obs[agentID] = ObsData(new_state, reward, False)

        return len(decision_steps.agent_id)
      
    def get_action(self, agentId: int, net: nn.Module, epsilon: float, device: str) -> int:
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
            state = torch.tensor([self.obs[agentId].state])

            if device not in ["cpu"]:
                state = state.cuda(device)
            
            obs, cam = versionControl.current_interpreter(state)
            q_values = net(obs, cam)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def update_state(self, action):
        """Update sate with an action."""
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float,
        device: str,
    ) -> Tuple[float, bool, np.array]:
        """
        Carries out a single interaction step between the agent and the
        environment.

        :param net: DQN network.
        :param epsilon: probability of taking a random action.
        :param device: current device.
        :return: reward, done.
        """
        actions = []
        
        for agentId in range(self.numAgent):
          action = self.get_action(agentId, net, epsilon, device)
          actions.append(np.array([action]))
        
        self.update_state(np.array(actions))
        self.get_obs()
        
        # viz
        # new_state_tr = torch.tensor([new_state])
        # utils.viz_occupancy_grid3D(new_state_tr[:, 6:].cpu().numpy(), new_state_tr[:, 3:6].cpu().numpy()[0].astype(int))
        # utils.center_of_array_1d(new_state_tr[:, 6:].cpu().numpy(), new_state_tr[:, 3:6].cpu().numpy()[0].astype(int))
        
        # add exp for each agent:
        for agentId in self._obs:
          # if agentId exist
          if(len(self.obs) == len(self._obs)): 
            # self._obs[agentId].print()
            obsdata = self._obs[agentId]
            exp = Experience(self.obs[agentId].state, actions[agentId][0], obsdata.reward, obsdata.done, obsdata.state)
            self.replay_buffer.append(exp)
          # else:
          #   import ipdb
          #   ipdb.set_trace()

        self.obs = self._obs.copy()
        
        # TODO: Xi: what would be the reset condition ?
        #           here I reset as long as any agent finishes
        # 
        return self.allRewards(), self.anyDone(), self.obs[0].state
      
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()

        return reward, done, new_state
