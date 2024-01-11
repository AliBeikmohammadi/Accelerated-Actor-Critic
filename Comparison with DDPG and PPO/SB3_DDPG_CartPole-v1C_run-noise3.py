import gym_examples
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy #TD3 has different MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise #TD3

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3 import DDPG


import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1:cpu, 0:first gpu
import argparse
import pandas as pd
import random
import pylab
import copy
import math
from tensorboardX import SummaryWriter
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

arg_pass=argparse.ArgumentParser()
arg_pass.add_argument(
  "--Seed",
  help='Seed; default 1',
  type=int,  
  default=1,
)
arg_pass.add_argument(
  "--algo", help='default DDPG',
  default='DDPG',
)
arg_pass.add_argument(
  "--env", help='default CartPole',
  default='CartPole',
)
arg_pass.add_argument(
  "--Adap_B",
  nargs='*', help='Adaptive_Beta ; default True',
  type=str2bool,  
  default=True,
)
arg_pass.add_argument(
  "--Just_B",
  nargs='*', help='Just_Beta ; default True',
  type=str2bool,  
  default=True,
)
arg_pass.add_argument(
  "--w",
  help='omega, weighting values to the control objective; default 10',
  type=float, 
  default=10,
)
arg_pass.add_argument(
  "--b",
  help='initial_beta; default 1  if 0: only rl loss, if 100: only mse loss',
  type=float, 
  default=1,
)
arg_pass.add_argument(
  "--s",
  help='start_iteration_number; default 0',
  type=int,  
  default=0,
)
arg_pass.add_argument(
  "--e",
  help='end_iteration_number; default 1000',
  type=int,  
  default=1000,
)
arg_pass.add_argument(
  "--SIGMA",
  help='SIGMA; default 0',
  type=float, 
  default=0,
)
arg_pass.add_argument(
  "--MU",
  help='MU; default 0',
  type=float, 
  default=0,
)
args = arg_pass.parse_args()

Seed= args.Seed
algo= args.algo
if args.env == 'CartPole':
    envname='CartPoleContinuous-v1'
Adaptive_Beta=args.Adap_B 
Just_Beta = args.Just_B 
MSE_Weight=args.w
Initial_Beta=args.b
start_iteration_number=args.s
end_iteration_number=args.e
SIGMA=args.SIGMA
MU=args.MU

def run_main(Seed= 1, algo= 'DDPG', envname='CartPoleContinuous-v1', Adaptive_Beta=True, Just_Beta = True, MSE_Weight=1, Initial_Beta=1,  start_iteration_number=0 , end_iteration_number=10, SIGMA=0, MU=0):
    SEED=Seed #1
    
    assert (start_iteration_number < end_iteration_number), f"`start_iteration_number` must be less than `end_iteration_number`. Currently start_iteration_number={start_iteration_number} and end_iteration_number={end_iteration_number}"
    Save_results='./Results_'+envname+'/'
    if Adaptive_Beta:
        if Just_Beta:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Just_Beta'+str(Initial_Beta)+'_s'+str(start_iteration_number)+'_e'+str(end_iteration_number)+'_sig'+str(SIGMA)+'_mu'+str(MU)+'_seed'+str(SEED)
        else:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Beta'+str(Initial_Beta)+'_s'+str(start_iteration_number)+'_e'+str(end_iteration_number)+'_sig'+str(SIGMA)+'_mu'+str(MU)+'_seed'+str(SEED)
    else:
        if Just_Beta:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Just_Fixed_Beta'+str(Initial_Beta)+'_sig'+str(SIGMA)+'_mu'+str(MU)+'_seed'+str(SEED)
        else:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Fixed_Beta'+str(Initial_Beta)+'_sig'+str(SIGMA)+'_mu'+str(MU)+'_seed'+str(SEED)
    if Initial_Beta==100:
        logname=envname+'_'+algo+'_only_MSE'+'_Weight'+str(MSE_Weight)+'_sig'+str(SIGMA)+'_mu'+str(MU)+'_seed'+str(SEED)
    if Initial_Beta==0:
        logname=envname+'_'+algo+'_only_RL_seed'+str(SEED)
    if SEED==None:
        np.random.seed(SEED)

    env = gym.make('gym_examples/'+ envname) #without any wrapper and monitor
    new_logger = configure(Save_results+logname, ["stdout", "csv", "tensorboard"])
    total_timesteps=300000 
    learning_starts=100 
    train_freq= (1, 'episode') #buffer_size=1000000
  
    model = DDPG('MlpPolicy', env, learning_rate=0.001, buffer_size=1000000, learning_starts=learning_starts, 
                 batch_size=100, tau=0.005, gamma=0.99, train_freq=train_freq, gradient_steps=-1, 
                 action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None, 
                 optimize_memory_usage=False, tensorboard_log=Save_results, create_eval_env=True, 
                 policy_kwargs=None, verbose=1, seed=SEED, device='auto', _init_setup_model=True) #gamma=0.98
    
    model.set_logger(new_logger)

    print('Seed:', SEED, '| algo:', algo, '| envname:', envname, '| SIGMA:', SIGMA, '| MU:', MU)
    print('Adaptive_Beta:', Adaptive_Beta, '| Just_Beta:', Just_Beta, '| Initial_Beta:', Initial_Beta, '| MSE_Weight:', MSE_Weight )
    print('total_timesteps', total_timesteps,  '| start_iteration_number:', start_iteration_number , '| end_iteration_number:', end_iteration_number) #TD3
    model.learn(total_timesteps=total_timesteps, tb_log_name=logname) #total_timesteps will be larger than this value if total_timesteps%n_steps!=0, total_timesteps=(ceil(total_timesteps/n_steps))*n_steps
    model.save(Save_results+'Weights/'+logname) # Save the agent

run_main(Seed= Seed, algo= algo, envname=envname, Adaptive_Beta=Adaptive_Beta, Just_Beta = Just_Beta, 
         MSE_Weight=MSE_Weight, Initial_Beta=Initial_Beta,  
         start_iteration_number=start_iteration_number , end_iteration_number=end_iteration_number,
         SIGMA=SIGMA, MU=MU)

