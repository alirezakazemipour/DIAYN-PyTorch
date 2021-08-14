import torch
from torch import device
import gym
import time
# from mujoco_py.generated import const
from mujoco_py import GlfwContext
import cv2
GlfwContext(offscreen=True)
import numpy as np


class Play:
    def __init__(self, env, agent, max_episode=1):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_to_cpu_mode()
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.VideoWriter = cv2.VideoWriter("Humanoid" + ".avi", self.fourcc, 50.0, (250, 250))

    def concat_state_latent(self, s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):

        for _ in range(self.max_episode):
            for z in range(50):
                s = self.env.reset()
                s = self.concat_state_latent(s, z, 50)
                episode_reward = 0
                for _ in range(self.env._max_episode_steps):
                    action = self.agent.choose_action(s)
                    s_, r, done, _ = self.env.step(action)
                    s_ = self.concat_state_latent(s_, z, 50)
                    episode_reward += r
                    if done:
                        break
                    s = s_
                    self.env.render()
                    # self.env.viewer.cam.type = const.CAMERA_FIXED
                    # self.env.viewer.cam.fixedcamid = 0
                    # time.sleep(0.005)
                    # I = self.env.render(mode='rgb_array')
                    # I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                    # I = cv2.resize(I, (250, 250))
                    # self.VideoWriter.write(I)
                    # cv2.imshow("env", I)
                    # cv2.waitKey(10)
                print(f"skill: {z}, episode reward:{episode_reward:3.3f}")
        self.env.close()
        # self.VideoWriter.release()
        # cv2.destroyAllWindows()
