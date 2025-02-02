import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            tau=0.005,  # interpolation parameter for soft update
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

    #def save(self, filename):
    #    torch.save(self.critic.state_dict(), filename + "_critic")
    #    torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    #    torch.save(self.actor.state_dict(), filename + "_actor")
    #    torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    #def load(self, filename):
    #    self.critic.load_state_dict(torch.load(filename + "_critic"))
    #    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    #    self.critic_target = copy.deepcopy(self.critic)

    #    self.actor.load_state_dict(torch.load(filename + "_actor"))
    #    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    #    self.actor_target = copy.deepcopy(self.actor)


if __name__ == "__main__":
    SEED = 42

    # %%capture
    env = gym.make("BipedalWalker-v3")
    env = gym.wrappers.RecordVideo(env, "./video", episode_trigger=lambda ep_id: (ep_id % 25 == 0), video_length=0)  # and ep_id > 750
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

    state, done = env.reset(), False
    reward_list = []
    plot_data = []
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_ep_steps = 2000
    max_timesteps = int(1e6)
    start_timesteps = 25e3
    expl_noise = 0.1
    batch_size = 256  # 128
    eval_freq = 5000
    log_f = open("agent-log_og.txt", "w+")
    file_name = "td3"
    for t in range(max_timesteps):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        #env._max_episode_steps = 1600
        if episode_timesteps < max_ep_steps:
            done_bool = float(done)
        else:
            done_bool = 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            log_f.write(f'episode: {episode_num + 1}, reward: {episode_reward}\n')
            log_f.flush()
            # Reset environment
            reward_list.append(episode_reward)
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # print reward data every so often - add a graph like this in your report
            if episode_num % 10 == 0:
                plot_data.append([episode_num, np.array(reward_list).mean(), np.array(reward_list).std()])
                reward_list = []
                # plt.rcParams['figure.dpi'] = 100
                plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:red')
                plt.fill_between([x[0] for x in plot_data], [x[1] - x[2] for x in plot_data],
                                 [x[1] + x[2] for x in plot_data], alpha=0.2, color='tab:grey')
                plt.xlabel('Episode number')
                plt.ylabel('Episode reward')
                plt.show()
                plt.savefig('./training.png')
                #plt.show()

        # Evaluate episode
        #if 750000 < (t + 1):
        #    if 750000 < (t + 1) < 850000 and (t + 1) % 5000 == 0:
        #        eval_n = eval_policy(policy, SEED)
        #        evaluations.append(eval_n)
        #        np.save("./results/td3_750-850", evaluations)
        #        if eval_n > 330:
        #            break
        #    elif 850000 < (t + 1) < 950000 and (t + 1) % 2500 == 0:
        #        eval_n = eval_policy(policy, SEED)
        #        evaluations.append(eval_n)
        #        np.save("./results/td3_850-950", evaluations)
        #        if eval_n > 330:
        #            break
        #    elif 950000 < (t + 1) < 1000000 and (t + 1) % 1000 == 0:
        #        eval_n = eval_policy(policy, SEED)
        #        evaluations.append(eval_n)
        #        np.save("./results/td3_950_1000", evaluations)
        #        if eval_n > 330:
        #            break


    env.close()