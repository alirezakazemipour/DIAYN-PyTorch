import numpy as np
from model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax

torch.manual_seed(123)


class SAC:
    def __init__(self, env_name,
                 n_states,
                 n_actions,
                 n_skills,
                 memory_size,
                 batch_size,
                 gamma, alpha,
                 lr,
                 action_bounds,
                 reward_scale,
                 p_z):

        self.env_name = env_name
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_skills = n_skills
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.action_bounds = action_bounds
        self.reward_scale = reward_scale
        self.memory = Memory(memory_size=self.memory_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.n_actions,
                                            action_bounds=self.action_bounds).to(self.device)
        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.n_actions).to(self.device)
        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.n_actions).to(self.device)
        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills).to(self.device)
        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills).to(self.device)
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()
        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills).to(self.device)

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.lr)

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.Tensor([z]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.alpha * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.value_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs.long())
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs.long()).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.reward_scale * rewards.float() + \
                           self.gamma * self.value_target_network(next_states) * (1 - dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.q_value_loss(q1, target_q)
            q2_loss = self.q_value_loss(q2, target_q)

            policy_loss = (self.alpha * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            logq_z_s = log_softmax(logits, dim=-1)
            logq_z_s.gather(-1, zs.long())
            discriminator_loss = self.cross_ent_loss(logits, zs.long().squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return discriminator_loss.item(), logq_z_s.mean().item()

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def save_weights(self):
        torch.save(self.policy_network.state_dict(), self.env_name + "_weights.pth")

    def load_weights(self):
        self.policy_network.load_state_dict(torch.load(self.env_name + "_weights.pth", map_location=self.device))

    def set_to_eval_mode(self):
        self.policy_network.eval()

    def set_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)

