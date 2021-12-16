import os
import torch
import numpy as np
from torch import nn


class Config(object):
    def __init__(self):
        self.learning_config()
        self.model_config()

    def learning_config(self):
        """
        Learning rate   :   param for optimizer
        Weight_decay    :   param for optimizer
        Reduce_rate     :   learning rate reduction
        Epoch           :   epoch for training
        Batch_size      :   batch size in each iteration
        Memory_capacity :   storage memory, for DQN training, to avoid
            relationship, we maintain a history database, and update it
            iteratively, when train the network, we random select batch_size
            info from the memory
        GAMMA           :   reduction rate of reward
        Q_iteration     :   number of turns to copy weight from eval_net to
            target network
        """
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.reduce_rate = 0.99
        self.epoch = 1000
        self.batch_size = 128
        self.memory_capacity = 1000
        self.GAMMA = 0.9
        self.Q_iteration = 100

    def model_config(self):
        """
        env_size        :   I first plan to use occupancy map to represent the
            env, and this env_size is the resolution of occupancy map
        action_space    :   dimension of action space. 2 : 0-left, 1-right
        """
        self.env_size = 6
        self.action_space = 2

        self.pretrain = True
        self.load_name = "latest"
        self.save_freq = 20
        self.eval_freq = 1
        self.log_dir = "log"
        self.model_path = "model"

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


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
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size)
        )

    def forward(self, env, position):
        # env = self.env_encoder(env).view(batch_size, -1)
        # env = self.env_fc(env)

        env = self.env_fc(env)
        pos = self.position_fc(position)

        return self.C_value(torch.cat([env, pos], axis=1))


class DQN(object):
    def __init__(self, config):
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

        if self.config.pretrain:
            self.load_ckpt(self.eval_net)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            if os.path.exists("memory.npy"):
                self.memory = list(np.load("memory.npy", allow_pickle=True))

    def store_transition(self, env, pos, action, reward, next_env, next_pos):
        if self.memory_counter < self.config.memory_capacity:
            self.memory.append([env, pos, action, reward, next_env, next_pos])
        else:
            index = self.memory_counter % self.config.memory_capacity
            self.memory[index] = [env, pos, action, reward, next_env, next_pos]
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.config.Q_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.scheduler.step(self.learn_step_counter // 100)
            np.save("memory", self.memory)
            self.save_ckpt(self.eval_net)

        if self.memory_counter < self.config.memory_capacity:
            return 0

        self.learn_step_counter += 1

        sample_index = np.random.choice(
            self.config.memory_capacity, self.config.batch_size
        )

        batch_memory = np.array(self.memory)[sample_index, :]

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

    def act(self, env, position, greedy_ratio):
        if np.random.rand() < greedy_ratio:
            return np.random.randint(0, self.config.action_space)

        env = torch.FloatTensor(env).unsqueeze(0)
        position = torch.FloatTensor(position).unsqueeze(0)

        C_value = self.eval_net(env, position)

        C_value = C_value.detach().numpy()

        return np.argmax(C_value)

    def save_ckpt(self, model, cate="", name=None):
        if name is None:
            save_path = os.path.join(
                self.config.model_path, cate + "latest.pth.tar"
            )
        else:
            save_path = os.path.join(
                self.config.model_path, cate + "{}.pth.tar".format(name)
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "learn_step_count": self.learn_step_counter,
            },
            save_path,
        )

    def load_ckpt(self, model, cate="", name=None, only_model=False):
        if name is None:
            load_path = os.path.join(
                self.config.model_path, cate + "latest.pth.tar"
            )
        else:
            load_path = os.path.join(
                self.config.model_path, cate + "{}.pth.tar".format(name)
            )

        if not os.path.exists(load_path):
            return
        print("load checkpoint from {}".format(load_path))

        checkpoint = torch.load(load_path)

        if only_model:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.learn_step_counter = checkpoint["learn_step_count"]
