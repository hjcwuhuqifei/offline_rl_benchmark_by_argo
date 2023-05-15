import math
import numpy as np
import d3rlpy
import pickle
import gym
from pathlib import Path
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL
from d3rlpy.algos import BC
from d3rlpy.algos import BCQ
from d3rlpy.algos import AWAC
from d3rlpy.algos import CRR
from d3rlpy.algos import IQL
from d3rlpy.algos import PLAS
import gym_env
import numpy

ENV_NAME = 'carla-v2'
env = gym.make(ENV_NAME)

# prepare algorithm
plas = PLAS.from_json('PLAS/params.json')
plas.load_model('PLAS/model.pt')


episode_rewards = []
speed_ = []
collision_time = 0
env.seed(100)
numpy.random.seed(100)
for _ in range(2330):
    observation = env.reset()
    episode_reward = 0.0

    while True:
        # take action
        action = plas.predict([observation])[0]

        observation, reward, done, collision = env.step(action)
        episode_reward += reward
        speed_.append(action*10 + 10)
        # env.render()
        if collision:
            collision_time += 1
        if done:
            # env.close()
            break
    episode_rewards.append(episode_reward)
print(1 - (2330 - collision_time)/2330)
print(sum(speed_)/len(speed_))
print(sum(episode_rewards)/len(episode_rewards))