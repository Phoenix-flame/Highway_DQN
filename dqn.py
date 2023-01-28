from replay_buffer import ReplayMemory, Transition, PrioritizedReplayMemory
from networks import DQN
from logger import Logger
import gym
from gym.wrappers import FilterObservation, FlattenObservation, TransformObservation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import count
import random
import os

def get_epsilon_scheduler(start, end, decay=0.01, end_fraction=0.1, _type='exp'):
    def exp(idx):
        val = start * np.exp(-decay*idx)
        if val < end:
            return end
        else:
            return val

    def linear(progress):
        if (1 - progress) < end_fraction:
            return end
        return start + ((end - start)/0.9)*progress
    
    if _type == 'exp':
        return exp
    elif _type == 'linear':
        return linear

class Agent:
    def __init__(
        self, 
        env:gym.Env,
        lr = 5e-4,
        batch_size = 32,
        tau = 0.05,
        gamma = 0.9,
        ep_start = 1.0,
        ep_end = 0.05,
        ep_decay = 0.01,
        config = {},
        id: str = ''
    ):
        self.id = id

        ## Environment
        self.env = env
        self.env = TransformObservation(self.env, lambda x: x.flatten())

        ## Hyperparameters
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.scheduler_type = 'linear'
        self.epsilon_scheduler = get_epsilon_scheduler(ep_start, ep_end, _type=self.scheduler_type)

        ## Config
        self.load_config(config)

        self.n_actions = env.action_space.n
        if gym.__version__ == '0.21.0':
            state = self.env.reset()
        else:
            state, _ = self.env.reset()
        self.n_observation = len(state)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.n_observation, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observation, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(15000)
        if self.memory_type == 'per': self.memory = PrioritizedReplayMemory(15000, state_size=self.n_observation, action_size=self.n_actions)

        if self.log: 
            self.logger = Logger(os.path.join('hw5', self.id))
        
        self.reset()

    def _non_deterministic_policy(self, state):
        sample = random.random()

        if self.scheduler_type == 'exp':
            eps_threshold = self.epsilon_scheduler(self.episode_counter)
        elif self.scheduler_type == 'linear':
            eps_threshold = self.epsilon_scheduler(self.progress)

        if self.log: self.logger.log('Train/epsilon', eps_threshold, self.episode_counter)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def __deterministic_policy(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def policy(self, state, deterministic=False):
        if deterministic:
            return self.__deterministic_policy(state)
        else:
            return self._non_deterministic_policy(state)

    def load_config(self, config):
        self.scheduler_type = 'linear'
        self.log = False
        self.checkpoint = False
        self.checkpoint_interval = 0
        self.memory_type = 'normal'
        if config != {}:
            if 'scheduler_type' in config:
                self.scheduler_type = config['scheduler_type']
            if 'checkpoint' in config:
                self.checkpoint = config['checkpoint'][0]
                self.checkpoint_interval = config['checkpoint'][1]
            if 'log' in config:
                self.log = config['log']
            if 'memory_type' in config:
                self.memory_type = config['memory_type']

    def reset(self):
        self.episode_len = []
        self.episode_avg_reward = []
        self.loss = []
        self.reward = []
        self.episode_counter = 0
        self.step_counter = 0
        self.progress = 0.0

    def update_stat(self, episode, max_episodes):
        self.episode_counter = episode
        self.progress = (float(episode) / float(max_episodes))

    def save(self, idx='', path='./'):
        torch.save(self.policy_net.state_dict(), path + 'policy_net' + str(idx) + '.pth')
        torch.save(self.target_net.state_dict(), path + 'target_net' + str(idx) + '.pth')

    def load(self, idx='', path='./'):
        saved_variables = torch.load(path + 'policy_net' + str(idx) + '.pth', map_location=self.device)

        self.policy_net.load_state_dict(saved_variables)
        self.target_net.load_state_dict(self.policy_net.state_dict())

class dqnAgent(Agent):
    def __init__(
        self,
        env:gym.Env,
        lr = 1e-4,
        batch_size = 32,
        tau = 0.05,
        gamma = 0.8,
        ep_start = 1.0,
        ep_end = 0.05,
        ep_decay = 0.01,
        config = {},
        id = ''
    ):
        super(dqnAgent, self).__init__(
            env, 
            lr,
            batch_size,
            tau,
            gamma,
            ep_start,
            ep_end,
            ep_decay,
            config,
            id,
        )


    def learn(self, num_episode):
        for episode in range(num_episode):
            self.update_stat(episode, num_episode)

            if gym.__version__ == '0.21.0':
                state = self.env.reset()
            else:
                state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            avg_reward = []
            for t in count():
                action = self.policy(state, deterministic=False)
                if gym.__version__ == '0.21.0':
                    observation, reward, terminated, _ = self.env.step(action.item())
                else:
                    observation, reward, terminated, _, _ = self.env.step(action.item())

                avg_reward.append(reward)

                reward = torch.tensor([reward], device=self.device)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                loss, td_error = self.optimize()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    if self.log: self.logger.log("Train/ep_len", t+1, self.episode_counter)
                    if self.log: self.logger.log("Train/avg_reward", np.mean(np.array(avg_reward)), self.episode_counter)
                    if self.log: self.logger.log("Train/reward", np.sum(np.array(avg_reward)), self.episode_counter)
                    if self.checkpoint and (episode != 0) and (episode%self.checkpoint_interval == 0):
                        self.save(idx=episode, path='./' + self.id)
                    break

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None, None
        

        weights = None
        if self.memory_type == 'normal': 
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

        elif self.memory_type == 'per': 
            batch, weights, data_idxs = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*batch))
        
        

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        q_value = self.policy_net(state_batch).gather(1, action_batch)

        if weights is None:
            weights = torch.ones_like(q_value)
        
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q_next = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            q_next[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        q_target = ((q_next * self.gamma) + reward_batch).unsqueeze(1)
        assert q_value.shape == q_target.shape, f"{q_value.shape}, {q_target.shape}"
    

        td_error = torch.abs(q_value - q_target).detach()    
        loss = torch.mean(torch.multiply(torch.pow(q_value - q_target, 2), weights))
        if self.log: self.logger.log("Train/loss", loss, self.episode_counter)

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


        if self.memory_type == 'per': self.memory.update(data_idxs, td_error.numpy())

        return loss, td_error

    def predict(self, state):
        pass


    
