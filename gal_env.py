
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
#shut up
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import gym
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from model_set import EEGNet
import capilab_dataset2
from sklearn.model_selection import train_test_split


class EEGChannelOptimze(gym.Env):
    def __init__(self, config):
        self.data = config['data'] #X, y, x_test and y_test
        self.checkpoint_path = config['checkpoint_path']
        self.action_space = gym.spaces.Discrete(config['action_space'])
        self.observation_space = gym.spaces.MultiBinary(config['state_space'])
        self.fold = config['fold']
        self.reset()    

    def _step(self, X, y, x_val, y_val, x_test, y_test, verbose = False):
       pass
    
    def kfold_eval(self, x,y,x_test, y_test):
        pass
    
    def step(self, action):
        
        done = False
        info = {}
        reward = 0

        return reward, done, info

    def reset(self):
        pass

if __name__ == "__main__":
    
   pass