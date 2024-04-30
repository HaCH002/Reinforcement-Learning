import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(policy, seed, eval_episodes=10):
    eval_env = gym.make("BipedalWalker-v3")
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    eval_env.close()
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,  # reward decay
            tau=0.005,  # interpolation parameter for sot update
            policy_noise=0.2,
            noise_clip=0.5,  # 0.3
            policy_freq=2  # 10
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":
    SEED = 42

    # %%capture
    env = gym.make("BipedalWalkerHardcore-v3")
    #env = gym.wrappers.Monitor(env, "./video", video_callable=lambda ep_id: (ep_id % 25 == 0 and ep_id > 750), force=True)
    # Set seeds

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print('The environment has {} observations and the agent can take {} actions'.format(state_dim, action_dim))
    print('The device is: {}'.format(device))

    if device.type != 'cpu':
        print('It\'s recommended to train on the cpu for this')
    env.seed(SEED)
    env.action_space.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    policy = TD3(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    #evaluations = [eval_policy(policy, SEED)]

    plot_data = []
    reward_list = []
    episode_reward = 0
    max_episodes = 2000
    start_ep = 30
    max_env_timesteps = 2000
    start_timestep = 25e3
    expl_noise = 0.1
    batch_size = 256  # 128
    log_f = open("agent-log.txt", "w+")

    for episode in range(1, max_episodes+1):
        episode_timesteps = 0
        state, done = env.reset(), 0

        while not done:
            episode_timesteps += 1
            if episode < start_ep:
                action = env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            next_state, reward, done, _ = env.step(action)

            if episode_timesteps < max_env_timesteps:
                done_bool = float(done)
            else:
                done_bool = 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if episode >= start_ep:
                policy.train(replay_buffer, batch_size)

        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(
            f"Episode Number: {episode} Episode Timestep: {episode_timesteps} Reward: {episode_reward:.3f}")
        log_f.write(f'episode: {episode}, reward: {episode_reward}\n')
        log_f.flush()
        # Reset environment
        reward_list.append(episode_reward)
        episode_reward = 0


        # print reward data every so often - add a graph like this in your report
        if episode % 10 == 0:
            plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
            reward_list = []
            # plt.rcParams['figure.dpi'] = 100
            plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:red')
            plt.fill_between([x[0] for x in plot_data], [x[1] - x[2] for x in plot_data],
                             [x[1] + x[2] for x in plot_data], alpha=0.2, color='tab:grey')
            plt.xlabel('Episode number')
            plt.ylabel('Episode reward')
            plt.savefig('./training.png')




    env.close()
