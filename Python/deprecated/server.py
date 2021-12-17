#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import numpy as np
import time
import zmq
import os
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DQN import DQN
from tqdm import tqdm

class Config(object):
    def __init__(self):
        self.learning_config()
        self.model_config()

    def learning_config(self):
        '''
        Learning rate   :   param for optimizer
        Weight_decay    :   param for optimizer
        Reduce_rate     :   learning rate reduction
        Epoch           :   epoch for training
        Batch_size      :   batch size in each iteration
        Memory_capacity :   storage memory, for DQN training, to avoid relationship, we maintain a
                            history database, and update it iteratively, when train the network, we
                            random select batch_size info from the memory
        GAMMA           :   reduction rate of reward
        Q_iteration     :   number of turns to copy weight from eval_net to target network
        '''
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.reduce_rate = 0.99
        self.epoch = 1000
        self.batch_size = 128
        self.memory_capacity = 1000
        self.GAMMA = 0.9
        self.Q_iteration = 100

    def model_config(self):
        '''
        env_size        :   I first plan to use occupancy map to represent the env, and this env_size
                            is the resolution of occupancy map
        action_space    :   dimension of action space. 2 : 0-left, 1-right
        '''
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

def mag(v):
    return mt.sqrt(np.sum(v ** 2))

class Unity(object):
    '''
    Here I simulate a Unity environment to accelerate the training.
    cube_step       :   used to initialize the starting position of the cube
    cube_position   :   the current position of the cube
    reward          :   1 if no collision, -1 if collision
    direct          :   the motion direction of camera
    theta           :   the central angle of camera relative to character
    camera_position :   the current position of camera, camera_position = (cos(theta), sin(theta))
    '''
    def __init__(self):
        pass

    def init(self):
        # initialize cube and camera position, it shall maintain the camera will not collide with cube
        self.cube_step = np.random.randint(100)
        self.cube_position = np.array(
            [mt.cos(self.cube_step / 100.0 * mt.pi), 0, mt.sin(self.cube_step / 100.0 * mt.pi)])

        self.reward = 1
        self.direct = -5

        while True:
            self.theta = np.random.randint(0, 360)
            self.camera_position = np.array([mt.cos(self.theta / 360 *mt.pi), 0, mt.sin(self.theta / 360 * mt.pi)])
            if not self.collision(self.camera_position):
                break

    def take_action(self, action):
        # change moving direction of camera
        if action == 0:
            self.direct = -5
        else:
            self.direct = 5
        self.reward = 1

    def update(self, timestep):
        # move cube and camera, judge if there is collision
        timestep += self.cube_step
        self.cube_position = np.array(
            [mt.cos(timestep / 100.0 * mt.pi), 0, mt.sin(timestep / 100.0 * mt.pi)])

        self.theta += self.direct
        self.camera_position = np.array([mt.cos(self.theta / 360 *mt.pi), 0, mt.sin(self.theta / 360 * mt.pi)])

        if self.collision(self.camera_position):
            self.reward = -1

    def obs(self):
        # obs of environment

        # Matrix = np.zeros((6, 6, 6), dtype="float32")
        #
        # for i in range(-5, 6, 2):
        #     for j in range(-5, 6, 2):
        #         for k in range(-5, 6, 2):
        #             detect_point = np.array([i, j, k])
        #             hd = detect_point
        #
        #             Matrix[i][j][k] = self.collision(hd)

        return self.cube_position, self.camera_position, self.reward

    def collision(self, hd):
        # judge if camera is occluded by cube by calculate the relative angle between two lines
        hc = self.cube_position
        if np.sum(hc * hd) / (mag(hc) * mag(hd)) < mt.cos(mt.pi / 4):
            return 0
        return 1

class myserver(object):
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        print("Connect success, begin!")

        self.socket.setsockopt(zmq.RCVTIMEO, 10000)

    def Run(self, model):
        step = []
        loss = []
        e = 0

        if os.path.exists("record.npy"):
            d = np.load("record.npy", allow_pickle=True)[()]
            step = d["step"]
            loss = d["loss"]
            e = d["e"]

        episodes = 10000
        greedy_ratio = 0.1

        for e in range(e, episodes):
            s = 0
            lst_env = None
            lst_pos = None
            self.socket.recv()
            self.socket.send('start'.encode('ascii'))

            if e % 300 == 0:
                fig = plt.figure()
                plt.plot(np.arange(len(loss)), loss)
                plt.savefig("loss.png")
                plt.close()

                fig = plt.figure()
                plt.plot(np.arange(len(step)), step)
                plt.savefig("step.png")
                plt.close()

                np.save("record", {"step" : step,
                                   "loss" : loss,
                                   "e"    : e})

            while True:
                try:
                    message = self.socket.recv()
                except Exception as E:
                    print(E)
                    break

                if message == b'End':
                    self.socket.send('Stop'.encode('ascii'))
                    break

                text = message.decode("UTF-8").split(" ")

                env = []
                pos = []

                for i in range(3):
                    env.append(eval(text[i]))
                for i in range(3,6):
                    pos.append(eval(text[i]))
                r = eval(text[6])

                if s > 0:
                    model.store_transition(lst_env, lst_pos, action, r, env, pos)
                    loss.append(model.learn())
                action = model.act(env, pos, greedy_ratio)
                self.socket.send(str(action).encode('ascii'))

                lst_env = env
                lst_pos = pos
                s += 1
                if r < 0:
                    break

            print(e, s, len(model.memory))
            step.append(s)

if __name__ == "__main__":
    Server = myserver()
    MyDQN = DQN(Config())
    Server.Run(MyDQN)
