import gym
import highway_env
from matplotlib import pyplot as plt
import torch
from dqn import dqnAgent
import numpy as np

print(gym.__version__)

def render(env, idx, to_file=False):
    if gym.__version__ == '0.21.0':
        if to_file:
            img = env.render(mode='rgb_array')
            Image.fromarray(img).save(f'{idx}.png')
        else:
            env.render(mode='human')
    else:
        env.render()


to_file = False

if gym.__version__ == '0.21.0':
    env = gym.make('merge-v0', config={'observation': {"type": "Kinematics"}}) ## , config={'lane_change_reward': 0, 'observation': {"type": "TimeToCollision", 'horizon':10}} 
else:
    if to_file:
    	env = gym.make('merge-v0', render_mode='rgb_array')
    else:
    	env = gym.make('merge-v0', render_mode='rgb_array')

env.metadata['render_fps'] = 30
agent = dqnAgent(env=env, config={'log': True, 'memory_type':'per'}, ep_decay=0.005, id='_2')
agent.reset()
agent.load(idx='_test')

# agent.learn(1000)
# agent.save(idx='_test')

from gym.wrappers import FilterObservation, FlattenObservation, TransformObservation
env = TransformObservation(env, lambda x: x.flatten())
from PIL import Image

if gym.__version__ == '0.21.0':
    if to_file:
    	img = env.render(mode='rgb_array')
    else:
    	env.render(mode='human')
else:
    env.render()




## Test agent



from IPython import display as ipythondisplay
from gym.wrappers import RecordVideo
from pathlib import Path
import base64

def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


if gym.__version__ == '0.21.0':
    render(env, 0, to_file)
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
        render(env, i + 1, to_file)
    else:
        observation, reward, terminated, _, info = env.step(action.item())

    avg_reward.append(reward)
		
    if terminated:
        next_state = None
        break
    else:
        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        
    state = next_state


