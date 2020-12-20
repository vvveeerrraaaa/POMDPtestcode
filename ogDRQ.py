#OG DRQ

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IPython.display import clear_output
from matplotlib import pyplot as plt
%matplotlib inline

from timeit import default_timer as timer
from datetime import timedelta
import math
import random

from utils.wrappers import *

from agents.DQN import Model as DQN_Agent

from networks.network_bodies import SimpleBody, AtariBody

from utils.ReplayMemory import ExperienceReplayMemory
from utils.hyperparameters import Config

config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = config.device

#epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 30000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA=0.99
config.LR=1e-4

#memory
config.TARGET_NET_UPDATE_FREQ = 1024
config.EXP_REPLAY_SIZE = 10000
config.BATCH_SIZE = 32

#Learning control variables
config.LEARN_START = 10000
config.MAX_FRAMES=1500000

#Nstep controls
config.N_STEPS=1

#DRQN Parameters
config.SEQUENCE_LENGTH=8