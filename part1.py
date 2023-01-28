import gym
import highway_env
from matplotlib import pyplot as plt
import torch
from dqn import dqnAgent
import numpy as np

print(gym.__version__)

if gym.__version__ == '0.21.0':
    env = gym.make('merge-v0', config={'observation': {"type": "Kinematics"}}) ## , config={'lane_change_reward': 0, 'observation': {"type": "TimeToCollision", 'horizon':10}} 
else:
    env = gym.make('merge-v0', render_mode='rgb_array')

env.metadata['render_fps'] = 30
agent = dqnAgent(env=env, config={'log': True, 'memory_type':'per', 'scheduler_type':'exp'}, ep_decay=0.005, id='_2')
agent.reset()

## Load Learned Model
agent.load(idx='_test_per')


## Train From Scratch
# agent.learn(1000)
# agent.save(idx='_test')


## Test agent
from IPython import display as ipythondisplay
from gym.wrappers import RecordVideo
from pathlib import Path
import base64

def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = agent.env

if gym.__version__ == '0.21.0':
    env.render(mode='human')
else:
    env = record_videos(env)

if gym.__version__ == '0.21.0':
    state = env.reset()
else:
    state, _ = env.reset()


state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
avg_reward = []
for i in range(100):
    action = agent.policy(state, deterministic=True)
    if gym.__version__ == '0.21.0':
        observation, reward, terminated, info = env.step(action.item())
        env.render(mode='human')
    else:
        observation, reward, terminated, _, info = env.step(action.item())

    avg_reward.append(reward)
		
    if terminated:
        next_state = None
        break
    else:
        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        
    state = next_state


