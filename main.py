import gym
from agent import SAC
import time
import psutil
from torch.utils.tensorboard import SummaryWriter
from play import Play
import os
import datetime
import numpy as np

np.random.seed(123)
ENV_NAME = "BipedalWalker-v3"
test_env = gym.make(ENV_NAME)
TRAIN = False

if not os.path.exists(ENV_NAME):
    os.mkdir(ENV_NAME)


n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
test_env.close()
del test_env

MAX_EPISODES = 1000
memory_size = 1e+6
batch_size = 256
gamma = 0.99
alpha = 0.1
lr = 3e-4
num_skills = 50
p_z = np.full(num_skills, 1 / num_skills)
reward_scale = 1 # TODO


to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


def log(ep, start_time, episode_reward, memory_length, z, disc_loss, max_ep_reward, int_r):

    ram = psutil.virtual_memory()

    if episode % 20 == 0:
        print(f"EP:{ep}| "
              f"EP_r:{episode_reward:3.1f}| "
              f"Skill:{z}| "
              f"Memory_length:{memory_length}| "
              f"Duration:{time.time() - start_time:3.3f}| "
              f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM| "
              f'Time:{datetime.datetime.now().strftime("%H:%M:%S")}')
        agent.save_weights()

    with SummaryWriter(ENV_NAME + "/logs/") as writer:
        writer.add_histogram("Reward", episode_reward)
        writer.add_histogram(str(z), episode_reward)
        writer.add_scalar("discriminator loss", disc_loss, episode)
        writer.add_scalar("maximum episode reward", max_ep_reward, ep)
        writer.add_scalar("Intrinsic Reward", int_r, ep)


if __name__ == "__main__":
    print(f"Number of states:{n_states}\n"
          f"Number of actions:{n_actions}\n"
          f"Action boundaries:{action_bounds}")

    env = gym.make(ENV_NAME)
    env.seed(123)
    agent = SAC(env_name=ENV_NAME,
                n_states=n_states,
                n_actions=n_actions,
                n_skills=num_skills,
                memory_size=memory_size,
                batch_size=batch_size,
                gamma=gamma,
                alpha=alpha,
                lr=lr,
                action_bounds=action_bounds,
                reward_scale=reward_scale,
                p_z=p_z)

if TRAIN:
    max_ep_rewrad = -np.inf
    for episode in range(1, MAX_EPISODES + 1):
        z = np.random.choice(num_skills, p=p_z)
        state = env.reset()
        state = concat_state_latent(state, z, num_skills)
        episode_reward = 0
        disc_losses = []
        int_rewards = []
        done = 0
        start_time = time.time()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = concat_state_latent(next_state, z, num_skills)
            agent.store(state, z, done, action, next_state)
            disc_loss, int_reward = agent.train()
            disc_losses.append(disc_loss)
            int_rewards.append(int_reward)
            episode_reward += reward
            state = next_state
        if episode_reward > max_ep_rewrad:
            max_ep_rewrad = episode_reward
            print(f"Skill: {z} grabbed the max episode reward award! :D")
        log(episode, start_time, episode_reward, len(agent.memory), z,
            sum(disc_losses) / len(disc_losses), max_ep_rewrad, sum(int_rewards) / len(int_rewards))

else:
    player = Play(env, agent)
    player.evaluate()
