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
        self.running_disc_loss = 0
        self.running_logq_zs = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self._turn_on = False

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
        self._turn_on = True

    def _off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):
        if not self._turn_on:
            print("First you should turn the logger on once, via on() method to be able to log parameters.")
            return
        self._off()

        episode, episode_reward, skill, disc_loss, logq_zs, step = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_reward == 0:
            self.running_reward = episode_reward
            self.running_disc_loss = disc_loss
            self.running_logq_zs = logq_zs
        else:
            self.running_disc_loss = 0.9 * self.running_disc_loss + 0.1 * disc_loss
            self.running_logq_zs = 0.9 * self.running_logq_zs + 0.1 * logq_zs
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_reward

        self.last_10_ep_rewards.append(int(episode_reward))
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            last_10_ep_rewards = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')
        else:
            last_10_ep_rewards = 0  # It is not correct but does not matter.

        memory = psutil.virtual_memory()
        assert self.to_gb(memory.used) < 0.98 * self.to_gb(memory.total), "RAM usage exceeded permitted limit!"

        if episode % (self.config["interval"] // 3):
            self._save_weights(episode)

        if episode % self.config["interval"] == 0:
            print("EP:{}| "
                  "Skill:{}| "
                  "EP_Reward:{:.1f}| "
                  "EP_Running_Reward:{:.1f}| "
                  "EP_Duration:{:.2f}| "
                  "Memory_Length:{}| "
                  "Mean_steps_time:{:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time:{}| ".format(episode,
                                     skill,
                                     episode_reward,
                                     self.running_reward,
                                     self.duration,
                                     len(self.agent.memory),
                                     self.duration / (step / episode),
                                     self.to_gb(memory.used),
                                     self.to_gb(memory.total),
                                     datetime.datetime.now().strftime("%H:%M:%S"),
                                     ))

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Episode running reward", self.running_reward, episode)
            writer.add_scalar("Max episode reward", self.max_episode_reward, episode)
            writer.add_scalar("Moving average reward of the last 10 episodes", last_10_ep_rewards, episode)
            writer.add_scalar("Running Discriminator Loss", self.running_disc_loss, episode)
            writer.add_scalar("Running logq(z|s)", self.running_logq_zs, episode)
            writer.add_histogram(str(skill), episode_reward, episode)
            
        self.on()

    def _save_weights(self, episode):
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
