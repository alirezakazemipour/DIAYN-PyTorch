import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob
from collections import deque


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.running_reward = 0
        self.running_alpha_loss = 0
        self.running_q_loss = 0
        self.running_policy_loss = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)

        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
        if self.config["do_train"] and self.config["train_from_scratch"]:
            self._create_wights_folder(self.log_dir)
            self._log_params()

    @staticmethod
    def _create_wights_folder(dir):
        if not os.path.exists("../models"):
            os.mkdir("../models")
        os.mkdir("models/" + dir)

    def _log_params(self):
        with SummaryWriter("Logs/" + self.log_dir) as writer:
            for k, v in self.config.items():
                writer.add_text(k, str(v))

    def on(self):
        self.start_time = time.time()

    def _off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):
        self._off()

        episode, episode_reward, alpha_loss, q_loss, policy_loss, step = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_reward == 0:
            self.running_reward = episode_reward
            self.running_alpha_loss = alpha_loss
            self.running_q_loss = q_loss
        else:
            self.running_alpha_loss = 0.99 * self.running_alpha_loss + 0.01 * alpha_loss
            self.running_q_loss = 0.99 * self.running_q_loss + 0.01 * q_loss
            self.running_policy_loss = 0.99 * self.running_policy_loss + 0.01 * policy_loss
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_reward

        self.last_10_ep_rewards.append(int(episode_reward))
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            last_10_ep_rewards = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')
        else:
            last_10_ep_rewards = 0  # It is not correct but does not matter.

        memory = psutil.virtual_memory()
        assert self.to_gb(memory.used) < 0.98 * self.to_gb(memory.total)

        if episode % (self.config["interval"] // 3):
            self.save_weights(episode)

        if episode % self.config["interval"] == 0:
            print("EP:{}| "
                  "EP_Reward:{:.2f}| "
                  "EP_Running_Reward:{:.3f}| "
                  "Alpha_Loss:{:.3f}| "
                  "Q-Loss:{:.3f}| "
                  "Policy_Loss:{:.3f}| "
                  "EP_Duration:{:.3f}| "
                  "Alpha:{:.3f}| "
                  "Memory_Length:{}| "
                  "Mean_steps_time:{:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time:{}| "
                  "Step:{}".format(episode,
                                   episode_reward,
                                   self.running_reward,
                                   self.running_alpha_loss,
                                   self.running_q_loss,
                                   self.running_policy_loss,
                                   self.duration,
                                   self.agent.alpha.item(),
                                   len(self.agent.memory),
                                   self.duration / (step / episode),
                                   self.to_gb(memory.used),
                                   self.to_gb(memory.total),
                                   datetime.datetime.now().strftime("%H:%M:%S"),
                                   step
                                   ))

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Episode running reward", self.running_reward, episode)
            writer.add_scalar("Max episode reward", self.max_episode_reward, episode)
            writer.add_scalar("Moving average reward of the last 10 episodes", last_10_ep_rewards, episode)
            writer.add_scalar("Alpha Loss", alpha_loss, episode)
            writer.add_scalar("Q-Loss", q_loss, episode)
            writer.add_scalar("Policy Loss", policy_loss, episode)
            writer.add_scalar("Alpha", self.agent.alpha.item(), episode)

    def save_weights(self, episode):
        torch.save({"policy_network_state_dict": self.agent.policy_network.state_dict(),
                    "q_value_network1_state_dict": self.agent.q_value_network1.state_dict(),
                    "q_value_network2_state_dict": self.agent.q_value_network2.state_dict(),
                    "log_alpha": self.agent.log_alpha,
                    "q_value1_opt_state_dict": self.agent.q_value1_opt.state_dict(),
                    "q_value2_opt_state_dict": self.agent.q_value2_opt.state_dict(),
                    "policy_opt_state_dict": self.agent.policy_opt.state_dict(),
                    "alpha_opt_state_dict": self.agent.alpha_opt.state_dict(),
                    "episode": episode},
                   "models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        self.agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.agent.q_value_network1.load_state_dict(checkpoint["q_value_network1_state_dict"])
        self.agent.q_value_network2.load_state_dict(checkpoint["q_value_network2_state_dict"])
        self.agent.log_alpha = checkpoint["log_alpha"]
        self.agent.q_value1_opt.load_state_dict(checkpoint["q_value1_opt_state_dict"])
        self.agent.q_value2_opt.load_state_dict(checkpoint["q_value2_opt_state_dict"])
        self.agent.policy_opt.load_state_dict(checkpoint["policy_opt_state_dict"])
        self.agent.alpha_opt.load_state_dict(checkpoint["alpha_opt_state_dict"])

        return checkpoint["episode"]
