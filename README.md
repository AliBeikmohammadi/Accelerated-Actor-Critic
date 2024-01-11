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
### AN ILLUSTRATION: Random Walk
> The script below runs a new test on the Random Walk environment with customized settings.
```
python RandomWalk.py -h


usage: RandomWalk.py [-h] [--N N] [--E E] [--R R] [--b [B ...]] [--l [L ...]] [--d D] [--save_dir SAVE_DIR]

optional arguments:
  -h, --help           show this help message and exit
  --N N                Number of states (including the ending states); default 5
  --E E                Number of episodes per each test; default 100
  --R R                Number of independent test; default 100
  --b [B ...]          initial_beta; default [0, 1, 100] if 0: only R^T, if 100: only only R^A, if (0, 1) if adaptive beta*R^A + (1-adaptive beta)*R^T where adaptive beta =
                       initial_beta*((lambda)**episode)
  --l [L ...]          lambda; default [0.1, 0.5, 0.9]
  --d D                debug_level; default 0 0:nothing print 1:print average result over runs, 2:print result after end of each episode, 3:print all information, like actions and reward in each state
  --save_dir SAVE_DIR  Save Directory; default ./Results/RandomWalk/
```
### OPTIMAL CONTROL PROBLEMS WITH CONSTRAINTS: Optimal Temperature Control with Constraint
> The scripts below run a new test on the Optimal Temperature Control with Constraint environment with customized settings.
```
python Optimal_Temperature_Control_with_Constraint.py -h


usage: Optimal_Temperature_Control_with_Constraint.py [-h] [--e E] [--b B] [--E E] [--w W] [--save_dir SAVE_DIR] [--d [D ...]]

optional arguments:
  -h, --help           show this help message and exit
  --e E                Total episodes to train through all environments; default 8000
  --b B                initial_beta; default 1 if 0: only R^T, if (0, 1] adaptive_beta*R^A + (1-adaptive_beta)*R^T where beta = max((E-e)*initial_beta/E),0)
  --E E                Episode in which the beta is zero; default 4000
  --w W                omega, weighting values to the control objective; default 1
  --save_dir SAVE_DIR  Save Directory; default ./Results/Temperature_Control/
  --d [D ...]          debug_level; default False False:nothing print True:print result per each episode
```

```
python Optimal_Temperature_Control_with_Constraint_added_results.py -h


usage: Optimal_Temperature_Control_with_Constraint_added_results.py [-h] [--a A] [--s S] [--w W] [--save_dir SAVE_DIR] [--d [D ...]]

options:
  -h, --help           show this help message and exit
  --a A                Algorithm (A2C, DDPG, SAC, TD3); default DDPG
  --s S                Seed number; default 0
  --w W                omega, weighting values to the control objective; default 1
  --save_dir SAVE_DIR  Save Directory; default ./Results/Temperature_Control/
  --d [D ...]          debug_level; default False False:nothing print True:print result per each episode
```


### OPTIMAL CONTROL PROBLEMS WITH CONSTRAINTS: A Coupled Four Tank MIMO System
> The scripts below run a new test on the Coupled Four Tank MIMO System environment with customized settings.
```
python A_Coupled_Four_Tank_MIMO_System.py -h


usage: A_Coupled_Four_Tank_MIMO_System.py [-h] [--e E] [--b B] [--E E] [--w W] [--save_dir SAVE_DIR] [--d [D ...]]

optional arguments:
  -h, --help           show this help message and exit
  --e E                Total episodes to train through all environments; default 30000
  --b B                initial_beta; default 0.5 if 0: only R^T, if (0, 1] adaptive_beta*R^A + (1-adaptive_beta)*R^T where beta = max((E-e)*initial_beta/E),0)
  --E E                Episode in which the beta is zero; default 3000
  --w W                omega, weighting values to the control objective; default 1
  --save_dir SAVE_DIR  Save Directory; default ./Results/Four_Tank/
  --d [D ...]          debug_level; default False False:nothing print True:print result per each episode
```

```
python A_Coupled_Four_Tank_MIMO_System_added_results.py -h


usage: A_Coupled_Four_Tank_MIMO_System_added_results.py [-h] [--a A] [--s S] [--w W] [--save_dir SAVE_DIR] [--d [D ...]]

options:
  -h, --help           show this help message and exit
  --a A                Algorithm (A2C, DDPG, SAC, TD3); default DDPG
  --s S                Seed number; default 0
  --w W                omega, weighting values to the control objective; default 1
  --save_dir SAVE_DIR  Save Directory; default ./Results/Four_Tank/
  --d [D ...]          debug_level; default False False:nothing print True:print result per each episode
```

# Examples
## Plots
### Random Walk
* Use the scripts below to generate the Random Walk environment figures mentioned in the papers.
```
python plot_results.py --Env RW --O_RT True --O_RA True --input_dir ./Results/RandomWalk/RW_s5_r100_e100.csv --save_dir ./Results/RandomWalk/ 
python plot_results.py --Env RW --O_RT True --O_RA False --input_dir ./Results/RandomWalk/RW_s11_r100_e150.csv --save_dir ./Results/RandomWalk/ 
python plot_results.py --Env RW --O_RT True --O_RA False --input_dir ./Results/RandomWalk/RW_s33_r100_e500.csv --save_dir ./Results/RandomWalk/ 
```
### Optimal Temperature Control with Constraint
* Use the scripts below to generate the Optimal Temperature Control with Constraint environment figures mentioned in the AAMAS paper.
```
python plot_results.py --Env TC --x_min 0 --x_max 8000 --TA_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega100_beta1.0_E4000.csv --baseline_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega100_beta0.0_E4000.csv --save_dir ./Results/Temperature_Control/
python plot_results.py --Env TC --x_min 0 --x_max 5000  --TA_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega10_beta1.0_E4000.csv --baseline_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega10_beta0.0_E4000.csv --save_dir ./Results/Temperature_Control/
python plot_results.py --Env TC --x_min 0 --x_max 1200 --TA_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta1.0_E4000.csv --baseline_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta0.0_E4000.csv --save_dir ./Results/Temperature_Control/
```

* Use the scripts below to generate the Optimal Temperature Control with Constraint environment figures mentioned in the ECC paper.
```
python plot_added_results.py --Env TC --TA_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta1.0_E4000.csv --PPO_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega1_beta0.0_E4000.csv --A2C_dir ./Results/Temperature_Control/Temperature_Control_A2C_omega1_R^T_reward.csv --DDPG_dir ./Results/Temperature_Control/Temperature_Control_DDPG_omega1_R^T_reward.csv --SAC_dir ./Results/Temperature_Control/Temperature_Control_SAC_omega1_R^T_reward.csv --TD3_dir ./Results/Temperature_Control/Temperature_Control_TD3_omega1_R^T_reward.csv --save_dir ./Results/Temperature_Control/Figures/  --x_min 0 --x_max 1200

python plot_added_results.py --Env TC --TA_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega10_beta1.0_E4000.csv --PPO_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega10_beta0.0_E4000.csv --A2C_dir ./Results/Temperature_Control/Temperature_Control_A2C_omega10_R^T_reward.csv --DDPG_dir ./Results/Temperature_Control/Temperature_Control_DDPG_omega10_R^T_reward.csv --SAC_dir ./Results/Temperature_Control/Temperature_Control_SAC_omega10_R^T_reward.csv --TD3_dir ./Results/Temperature_Control/Temperature_Control_TD3_omega10_R^T_reward.csv --save_dir ./Results/Temperature_Control/Figures/  --x_min 0 --x_max 5000

python plot_added_results.py --Env TC --TA_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega100_beta1.0_E4000.csv --PPO_dir ./Results/Temperature_Control/Temperature_Control_e8000_omega100_beta0.0_E4000.csv --A2C_dir ./Results/Temperature_Control/Temperature_Control_A2C_omega100_R^T_reward.csv --DDPG_dir ./Results/Temperature_Control/Temperature_Control_DDPG_omega100_R^T_reward.csv --SAC_dir ./Results/Temperature_Control/Temperature_Control_SAC_omega100_R^T_reward.csv --TD3_dir ./Results/Temperature_Control/Temperature_Control_TD3_omega100_R^T_reward.csv --save_dir ./Results/Temperature_Control/Figures/  --x_min 0 --x_max 8000
```


### A Coupled Four Tank MIMO System environment
* Use the scripts below to generate the Coupled Four Tank MIMO System environment figures mentioned in the AAMAS paper.
```
python plot_results.py --Env FT  --x_min 0 --x_max 30000 --TA_dir ./Results/Four_Tank/Four_Tank_e30000_omega1_beta0.5_E3000.csv --baseline_dir ./Results/Four_Tank/Four_Tank_e30000_omega1_beta0.0_E3000.csv --save_dir ./Results/Four_Tank/ 
```

* Use the scripts below to generate the Coupled Four Tank MIMO System environment figures mentioned in the ECC paper.
```
python plot_added_results.py --Env FT --TA_dir ./Results/Four_Tank/Four_Tank_e30000_omega1_beta0.5_E3000.csv --PPO_dir ./Results/Four_Tank/Four_Tank_e30000_omega1_beta0.0_E3000.csv --A2C_dir ./Results/Four_Tank/Four_Tank_A2C_omega1_R^T_reward.csv --DDPG_dir ./Results/Four_Tank/Four_Tank_DDPG_omega1_R^T_reward.csv --SAC_dir ./Results/Four_Tank/Four_Tank_SAC_omega1_R^T_reward.csv --TD3_dir ./Results/Four_Tank/Four_Tank_TD3_omega1_R^T_reward.csv --save_dir ./Results/Four_Tank/Figures/  --x_min 0 --x_max 15000 --y_min -370 --y_max -100
```

## Experiments
### Random Walk
* The scripts below run the test on the Random Walk environment with the same settings mentioned in the papers.
```
python RandomWalk.py --N 7 --E 100 --R 100 --b 0 1 100 --l 0.1 0.5 0.9 --d 0 --save_dir ./Results/RandomWalk/
python RandomWalk.py --N 11 --E 150 --R 100 --b 0 1 100 --l 0.1 0.5 0.9 --d 0 --save_dir ./Results/RandomWalk/
python RandomWalk.py --N 33 --E 500 --R 100 --b 0 1 --l 0.1 0.5 --d 0 --save_dir ./Results/RandomWalk/
```
### Optimal Temperature Control with Constraint
* The scripts below run the test on the Optimal Temperature Control with Constraint environment with the same settings mentioned in the AAMAS paper.
```
python Optimal_Temperature_Control_with_Constraint.py --e 8000 --b 1 --E 4000 --w 1 --d False --save_dir ./Results/Temperature_Control/
python Optimal_Temperature_Control_with_Constraint.py --e 8000 --b 0 --E 4000 --w 1 --d False --save_dir ./Results/Temperature_Control/
python Optimal_Temperature_Control_with_Constraint.py --e 8000 --b 1 --E 4000 --w 10 --d False --save_dir ./Results/Temperature_Control/
python Optimal_Temperature_Control_with_Constraint.py --e 8000 --b 0 --E 4000 --w 10 --d False --save_dir ./Results/Temperature_Control/
python Optimal_Temperature_Control_with_Constraint.py --e 8000 --b 1 --E 4000 --w 100 --d False --save_dir ./Results/Temperature_Control/
python Optimal_Temperature_Control_with_Constraint.py --e 8000 --b 0 --E 4000 --w 100 --d False --save_dir ./Results/Temperature_Control/
```
### A Coupled Four Tank MIMO System environment
* The scripts below run the test on the Coupled Four Tank MIMO System environment with the same settings mentioned in the AAMAS paper.
```
python A_Coupled_Four_Tank_MIMO_System.py --e 30000 --b 0.5 --E 3000 --w 1 --d False --save_dir ./Results/Four_Tank/
python A_Coupled_Four_Tank_MIMO_System.py --e 30000 --b 0 --E 3000 --w 1 --d False --save_dir ./Results/Four_Tank/
```

# Citation
Accelerating Actor-Critic-Based Algorithms via Pseudo-Labels Derived from Prior Knowledge. Information Sciences. 2024.
* Published in Information Sciences. 2024.
Please cite the accompanied paper, if you find this useful:
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
