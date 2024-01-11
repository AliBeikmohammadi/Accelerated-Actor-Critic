# Accelerated-Actor-Critic-Algorithms
Implementation of **Accelerated-Actor-Critic-Algorithms**, as presented in:
* Accelerating Actor-Critic-Based Algorithms via Pseudo-Labels Derived from Prior Knowledge. Information Sciences. 2024.



# Dependencies
```
gym==0.21.0
box2d==2.3.10
box2d-py
pyglet==1.5.23
pydot==1.4.2
tensorboardX==2.5
pyvirtualdisplay==3.0
xvfbwrapper==0.2.9
ffmpeg==1.4
stable-baselines==2.10.2
stable-baselines3==1.5.0
opencv-python-headless==4.5.5.64
sb3-contrib==1.5.0
PyYAML==6.0
optuna==2.10.0
plotly==5.14.1
scikit-plot==0.3.7
seaborn==0.11.2
torch==1.11.0
```
# Importing

> To run a new test .
```
import sys
import os
import argparse
import pandas as pd
import random
import pylab
import copy
import math
from tensorboardX import SummaryWriter
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
```

> To draw output figures.
```
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import pathlib
import seaborn as sns
```

# Usage
## How to Plot the Results

> The scripts below are used to draw the main output figures.
```
python plot_results.py
```

> The scripts below are used to draw figures related to the ablation study.
```
python plot_results_ablation.py

```


## How to Run Experiments
### Running Accelerated-TD3 on Pendulum-v1
> The script below runs a new test on the Pendulum-v1 environment using Accelerated-TD3 with customized settings.
```
python SB3_TD3_Pendulum-v1_run.py -h

usage: SB3_TD3_Pendulum-v1_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                  [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E] [--Noise_STD NOISE_STD]
                                  [--Noise_Prob NOISE_PROB]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 3531619893
  --algo ALGO           default TD3
  --env ENV             default Pendulum
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --Noise_STD NOISE_STD
                        Noise_STD; default 0
  --Noise_Prob NOISE_PROB
                        Noise_Prob; default 0
```

### Running Accelerated-TD3 on BipedalWalker-v3
> The script below runs a new test on the BipedalWalker-v3 environment using Accelerated-TD3 with customized settings.
```
python SB3_TD3_BipedalWalker-v3_run.py -h

usage: SB3_TD3_BipedalWalker-v3_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                       [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E] [--SIGMA SIGMA]
                                       [--MU MU]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 1493355531
  --algo ALGO           default TD3
  --env ENV             default Bipedal
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only R^T
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --SIGMA SIGMA         SIGMA; default 0
  --MU MU               MU; default 0
```

### Running Accelerated-TD3 on CartPole-v1C
> The script below runs a new test on the CartPole-v1C environment using Accelerated-TD3 with customized settings.
```
python SB3_TD3_CartPole-v1C_run-noise3.py -h

usage: SB3_TD3_CartPole-v1C_run-noise3.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                          [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E] [--SIGMA SIGMA]
                                          [--MU MU]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 1
  --algo ALGO           default TD3
  --env ENV             default CartPole
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --SIGMA SIGMA         SIGMA; default 0
  --MU MU               MU; default 0
```


### Running Accelerated-TD3 on LunarLander-v2C
> The script below runs a new test on the LunarLander-v2C environment using Accelerated-TD3 with customized settings.
```
python SB3_TD3_LunarLander-v2C_run.py -h

usage: SB3_TD3_LunarLander-v2C_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                      [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E]
                                      [--Noise_STD NOISE_STD] [--Noise_Prob NOISE_PROB]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 3510913823
  --algo ALGO           default TD3
  --env ENV             default Lunar
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --Noise_STD NOISE_STD
                        Noise_STD; default 0
  --Noise_Prob NOISE_PROB
                        Noise_Prob; default 0
```


### Running Accelerated-TD3 on MountainCar-v0C
> The script below runs a new test on the MountainCar-v0C environment using Accelerated-TD3 with customized settings.
```
python SB3_TD3_MountainCar-v0C_run.py -h

usage: SB3_TD3_MountainCar-v0C_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                      [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E]
                                      [--Noise_STD NOISE_STD] [--Noise_Prob NOISE_PROB]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 2228253929
  --algo ALGO           default TD3
  --env ENV             default Mountain
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --Noise_STD NOISE_STD
                        Noise_STD; default 0
  --Noise_Prob NOISE_PROB
                        Noise_Prob; default 0
```


### An Illustration of working with Accelerated-SAC
> The script below runs a new test on the Pendulum-v1 environment using Accelerated-SAC with customized settings.
```
python SB3_SAC_Pendulum-v1_run.py -h

usage: SB3_SAC_Pendulum-v1_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                  [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E] [--Noise_STD NOISE_STD]
                                  [--Noise_Prob NOISE_PROB]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 900598021
  --algo ALGO           default SAC
  --env ENV             default Pendulum
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --Noise_STD NOISE_STD
                        Noise_STD; default 0
  --Noise_Prob NOISE_PROB
                        Noise_Prob; default 0
```

### Running Accelerated-SAC on BipedalWalker-v3
> The script below runs a new test on the BipedalWalker-v3 environment using Accelerated-SAC with customized settings.
```
python SB3_SAC_BipedalWalker-v3_run.py -h

usage: SB3_SAC_BipedalWalker-v3_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                       [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E] [--SIGMA SIGMA]
                                       [--MU MU]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 2778197655
  --algo ALGO           default SAC
  --env ENV             default Bipedal
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only R^T
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --SIGMA SIGMA         SIGMA; default 0
  --MU MU               MU; default 0
```

### Running Accelerated-SAC on CartPole-v1C
> The script below runs a new test on the CartPole-v1C environment using Accelerated-SAC with customized settings.
```
python SB3_SAC_CartPole-v1C_run-noise3.py -h

usage: SB3_SAC_CartPole-v1C_run-noise3.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                          [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E] [--SIGMA SIGMA]
                                          [--MU MU]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 1
  --algo ALGO           default SAC
  --env ENV             default CartPole
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --SIGMA SIGMA         SIGMA; default 0
  --MU MU               MU; default 0
```


### Running Accelerated-SAC on LunarLander-v2C
> The script below runs a new test on the LunarLander-v2C environment using Accelerated-SAC with customized settings.
```
python SB3_SAC_LunarLander-v2C_run.py -h

usage: SB3_SAC_LunarLander-v2C_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                      [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E]
                                      [--Noise_STD NOISE_STD] [--Noise_Prob NOISE_PROB]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 1536726603
  --algo ALGO           default SAC
  --env ENV             default Lunar
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --Noise_STD NOISE_STD
                        Noise_STD; default 0
  --Noise_Prob NOISE_PROB
                        Noise_Prob; default 0
```


### Running Accelerated-SAC on MountainCar-v0C
> The script below runs a new test on the MountainCar-v0C environment using Accelerated-SAC with customized settings.
```
python SB3_SAC_MountainCar-v0C_run.py -h

usage: SB3_SAC_MountainCar-v0C_run.py [-h] [--Seed SEED] [--algo ALGO] [--env ENV] [--Adap_B [ADAP_B [ADAP_B ...]]]
                                      [--Just_B [JUST_B [JUST_B ...]]] [--w W] [--b B] [--s S] [--e E]
                                      [--Noise_STD NOISE_STD] [--Noise_Prob NOISE_PROB]
optional arguments:
  -h, --help            show this help message and exit
  --Seed SEED           Seed; default 4258880664
  --algo ALGO           default SAC
  --env ENV             default Mountain
  --Adap_B [ADAP_B [ADAP_B ...]]
                        Adaptive_Beta ; default True
  --Just_B [JUST_B [JUST_B ...]]
                        Just_Beta ; default True
  --w W                 omega, weighting values to the control objective; default 10
  --b B                 initial_beta; default 1 if 0: only rl loss, if 100: only mse loss
  --s S                 start_iteration_number; default 0
  --e E                 end_iteration_number; default 1000
  --Noise_STD NOISE_STD
                        Noise_STD; default 0
  --Noise_Prob NOISE_PROB
                        Noise_Prob; default 0
```




# Example
```
python SB3_TD3_Pendulum-v1_run.py --Seed 0 --algo TD3 --env Pendulum --b 0
```

# Citation
Accelerating Actor-Critic-Based Algorithms via Pseudo-Labels Derived from Prior Knowledge. Information Sciences. 2024.
* Published in Information Sciences. 2024.
  
Please cite the accompanying paper, if you find this useful:
```
@article{beikmohammadi2024IS,
  title={Accelerating Actor-Critic-Based Algorithms via Pseudo-Labels Derived from Prior Knowledge},
  author={Beikmohammadi, Ali and Magn{\'u}sson, Sindri},
  journal={Information Sciences},
  volume={x},
  pages={x--x},
  year={2024},
  publisher={Elsevier}
}
```
