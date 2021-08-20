import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.running_logq_zs = 0
        self.max_episode_reward = -np.inf
        self._turn_on = False

        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
        if self.config["do_train"] and self.config["train_from_scratch"]:
            self._create_wights_folder(self.log_dir)
            self._log_params()

    @staticmethod
    def _create_wights_folder(dir):
        if not os.path.exists("Checkpoints"):
            os.mkdir("Checkpoints")
        os.mkdir("Checkpoints/" + dir)

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

        episode, episode_reward, skill, logq_zs, step, *rng_states = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_logq_zs == 0:
            self.running_logq_zs = logq_zs
        else:
            self.running_logq_zs = 0.99 * self.running_logq_zs + 0.01 * logq_zs

        ram = psutil.virtual_memory()
        assert self.to_gb(ram.used) < 0.98 * self.to_gb(ram.total), "RAM usage exceeded permitted limit!"

        if episode % (self.config["interval"] // 3) == 0:
            self._save_weights(episode, *rng_states)

        if episode % self.config["interval"] == 0:
            print("EP:{}| "
                  "Skill:{}| "
                  "EP_Reward:{:.1f}| "
                  "EP_Duration:{:.2f}| "
                  "Memory_Length:{}| "
                  "Mean_steps_time:{:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time:{}| ".format(episode,
                                     skill,
                                     episode_reward,
                                     self.duration,
                                     len(self.agent.memory),
                                     self.duration / step,
                                     self.to_gb(ram.used),
                                     self.to_gb(ram.total),
                                     datetime.datetime.now().strftime("%H:%M:%S"),
                                     ))

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Max episode reward", self.max_episode_reward, episode)
            writer.add_scalar("Running logq(z|s)", self.running_logq_zs, episode)
            writer.add_histogram(str(skill), episode_reward)
            writer.add_histogram("Total Rewards", episode_reward)

        self.on()

    def _save_weights(self, episode, *rng_states):
        torch.save({"policy_network_state_dict": self.agent.policy_network.state_dict(),
                    "q_value_network1_state_dict": self.agent.q_value_network1.state_dict(),
                    "q_value_network2_state_dict": self.agent.q_value_network2.state_dict(),
                    "value_network_state_dict": self.agent.value_network.state_dict(),
                    "discriminator_state_dict": self.agent.discriminator.state_dict(),
                    "q_value1_opt_state_dict": self.agent.q_value1_opt.state_dict(),
                    "q_value2_opt_state_dict": self.agent.q_value2_opt.state_dict(),
                    "policy_opt_state_dict": self.agent.policy_opt.state_dict(),
                    "value_opt_state_dict": self.agent.value_opt.state_dict(),
                    "discriminator_opt_state_dict": self.agent.discriminator_opt.state_dict(),
                    "episode": episode,
                    "rng_states": rng_states,
                    "max_episode_reward": self.max_episode_reward,
                    "running_logq_zs": self.running_logq_zs
                    },
                   "Checkpoints/" + self.log_dir + "/params.pth")


    def load_weights(self):
        model_dir = glob.glob("Checkpoints/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        self.agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.agent.q_value_network1.load_state_dict(checkpoint["q_value_network1_state_dict"])
        self.agent.q_value_network2.load_state_dict(checkpoint["q_value_network2_state_dict"])
        self.agent.value_network.load_state_dict(checkpoint["value_network_state_dict"])
        self.agent.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.agent.q_value1_opt.load_state_dict(checkpoint["q_value1_opt_state_dict"])
        self.agent.q_value2_opt.load_state_dict(checkpoint["q_value2_opt_state_dict"])
        self.agent.policy_opt.load_state_dict(checkpoint["policy_opt_state_dict"])
        self.agent.value_opt.load_state_dict(checkpoint["value_opt_state_dict"])
        self.agent.discriminator_opt.load_state_dict(checkpoint["discriminator_opt_state_dict"])

        self.max_episode_reward = checkpoint["max_episode_reward"]
        self.running_logq_zs = checkpoint["running_logq_zs"]

        return checkpoint["episode"], self.running_logq_zs, *checkpoint["rng_states"]
