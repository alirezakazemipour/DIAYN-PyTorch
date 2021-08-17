import gym
from torch.utils.tensorboard import SummaryWriter
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
# import mujoco_py


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


def log(ep, start_time, episode_reward, memory_length, z, disc_loss, max_ep_reward, logq):


    if episode % 100 == 0:
        print(f"EP:{ep}| "
              f"EP_r:{episode_reward:3.1f}| "
              f"Skill:{z}| "
              f"Memory_length:{memory_length}| ")
        agent.save_weights()

    with SummaryWriter(ENV_NAME + "/logs/") as writer:
        writer.add_histogram("Reward", episode_reward)
        writer.add_histogram(str(z), episode_reward)
        writer.add_scalar("discriminator loss", disc_loss, episode)
        writer.add_scalar("maximum episode reward", max_ep_reward, ep)
        writer.add_scalar("logq(z|s)", logq, ep)


if __name__ == "__main__":
    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"])
    env.seed(params["seed"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode = logger.load_weights()
            agent.hard_update_target_network()
            agent.alpha = agent.log_alpha.exp()
            min_episode = episode
            print("Keep training from previous run.")

        else:
            min_episode = 0
            print("Train from scratch.")

        np.random.seed(params["seed"]) # it is not completely true when we're not in training_from_scratch mode!
        for episode in range(min_episode, params["max_n_episodes"] + 1):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            disc_losses = []
            logq_zses = []
            done = False
            logger.on()
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                disc_loss, logq_zs = agent.train()
                disc_losses.append(disc_loss)
                logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
            if episode_reward > max_ep_rewrad:
                max_ep_rewrad = episode_reward
                print(f"Skill: {z} grabbed the max episode reward award! :D")
            log(episode, start_time, episode_reward, len(agent.memory), z,
                sum(disc_losses) / len(disc_losses), max_ep_rewrad, sum(logq_zses) / len(logq_zses))

    else:
        player = Play(env, agent)
        player.evaluate()
