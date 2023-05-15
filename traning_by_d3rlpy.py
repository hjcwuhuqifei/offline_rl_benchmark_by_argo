import math
import numpy as np
import d3rlpy
import pickle
import gym
from pathlib import Path
from d3rlpy.dataset import MDPDataset
from gym_env.envs.offline_gym import get_reward
# prepare dataset

ENV_NAME = 'carla-v1'

dataset_from_td3 = pickle.load(open( 'dataset_expert', 'rb'))

rewards_ = dataset_from_td3['rewards']
terminals_ = dataset_from_td3['terminals']
observations_ = dataset_from_td3['observations']
actions_ = dataset_from_td3['actions']

while True:
    if terminals_[-1] == 0:
        terminals_ = terminals_[:-1]
        rewards_ = rewards_[:-1]
        observations_ = observations_[:-1]
    else:
        break

dataset = MDPDataset(observations_, actions_, rewards_, terminals_)

a = dataset.episode_terminals
env = gym.make(ENV_NAME)
env = env.unwrapped

#########################################################################################################################################
d3rlpy.seed(10)
env.seed(10)

# prepare algorithm
bc = d3rlpy.algos.BC(use_gpu=True)

# train
bc.fit(
    dataset,
    eval_episodes=dataset.episodes,
    tensorboard_dir='runs_expert_10',
    n_epochs=100,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
    },
)

