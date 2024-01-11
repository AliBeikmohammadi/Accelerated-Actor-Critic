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
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
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

arg_pass=argparse.ArgumentParser()
arg_pass.add_argument(
  "--Seed",
  help='Seed; default 1',
  type=int,  
  default=1,
)
arg_pass.add_argument(
  "--w",
  help='omega, weighting values to the control objective; default 10',
  type=float, 
  default=10,
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
MSE_Weight=args.w
end_iteration_number=args.e
SIGMA=args.SIGMA
MU=args.MU

class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        Adaptive_Beta: bool = True, Just_Beta: bool = True, MSE_Weight: float = 1, Initial_Beta: float = 1,  start_iteration_number: int = 0 , end_iteration_number: int = 10, SIGMA: float = 0, MU: float = 0): 


        super(TD3, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
########################################################################################################################################################################################
        self.adaptive_beta=Adaptive_Beta
        self.hu_weight=MSE_Weight
        self.initial_beta=Initial_Beta
        self.just_beta=Just_Beta
        self.sp=start_iteration_number 
        self.ep=end_iteration_number 
        self.sigma=SIGMA 
        self.mu=MU
########################################################################################################################################################################################
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        
########################################################################################################################################################################################
        RL_losses, MSE_losses = [], []
        Main_losses= []
        self.replay_counter = self._episode_num -1 

        assert (self.sp< self.ep
               ), f"`start_iteration_number` must be less than `end_iteration_number`. Currently start_iteration_number={self.sp} and end_iteration_number={self.ep}"
        if self.adaptive_beta:
            self.beta = max(min((self.replay_counter-self.ep)*self.initial_beta/(self.sp-self.ep), self.initial_beta),0)
        else: 
            self.beta=self.initial_beta
        if self.initial_beta == 100:
            self.beta=1
###########################################################################################################################################################################################      
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())
###########################################################################################################################################################################################       
                 ###hu environment
                # g = 9.8
                # mk = 1.0
                # mp = 0.1
                # lp = 0.1*0.5
                # # state matrix
                # a = g/(lp*(4.0/3 - mp/(mp+mk)))
                # A = np.array([[0, 1, 0, 0],
                #               [0, 0, a, 0],
                #               [0, 0, 0, 1],
                #               [0, 0, a, 0]])

                # # input matrix
                # b = -1/(lp*(4.0/3 - mp/(mp+mk)))
                # B = np.array([[0], [1/(mp+mk)], [0], [b]])
                # R = np.eye(1, dtype=int)          # choose R (weight for input)
                # Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)

                # # get riccati solver
                # from scipy import linalg

                # # solve ricatti equation
                # P = linalg.solve_continuous_are(A, B, Q, R)

                # # calculate optimal controller gain
                # K = np.dot(np.linalg.inv(R),
                #            np.dot(B.T, P))

                # print(K)
                # force = -np.dot(K, observation)   # u = -Kx
                # action = float(np.clip(force/10.0, -10, 10)) #normalized with env.env.force_mag =10.0

                observations = replay_data.observations.float() 
                actions_pi = self.actor(replay_data.observations) 
                K=th.tensor([[ -2.23606798],  [-2.5564412],  [-67.02015904],  [-3.79498899]], dtype=th.float, device=th.device("cuda")) 
                hu_actions=-1.*th.mm(observations.cuda(), K)
                hu_actions=th.clamp(hu_actions/30.0, min=-10.0, max=10.0)                
                #Noise type 1
                hu_actions=th.add(hu_actions, self.sigma * th.randn((actions_pi.shape[0], actions_pi.shape[1]), device=th.device("cuda"))+ self.mu) 
                MSE_loss = F.mse_loss(hu_actions, actions_pi)
                MSE_losses.append(MSE_loss.item())
                if self.initial_beta == 100: 
                    loss = self.hu_weight*MSE_loss
                else:
                    if self.just_beta:
                        if self.beta == 1:
                            loss = (self.hu_weight*MSE_loss)+actor_loss
                        elif self.beta == 0:
                            loss = actor_loss
                        else:
                            loss = (self.hu_weight * self.beta * MSE_loss) + actor_loss
                    else:
                        if self.beta == 1:
                            loss = self.hu_weight*MSE_loss
                        elif self.beta == 0:
                            loss = actor_loss
                        else:
                            loss = (self.hu_weight * self.beta * MSE_loss) + (1.-self.beta) * actor_loss            
                Main_losses.append(loss.item())  
###########################################################################################################################################################################################
  

                # Optimize the actor
                self.actor.optimizer.zero_grad()
###########################################################################################################################################################################################
                #actor_loss.backward()
                loss.backward()
###########################################################################################################################################################################################
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
###########################################################################################################################################################################################
            self.logger.record("train/RL_loss", actor_loss.item())
            self.logger.record("train/MSE_loss", MSE_loss.item())
            self.logger.record("train/MSE_losses", np.mean(MSE_losses))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/Main_losses", np.mean(Main_losses))
        self.logger.record("train/Beta", self.beta)
        self.logger.record("train/critic_los", critic_loss.item())
###########################################################################################################################################################################################               
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(TD3, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(TD3, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []


def run_main(Seed= 1, MSE_Weight=1, end_iteration_number=10, SIGMA=0, MU=0):
    SEED=Seed
    algo= 'TD3'
    envname='CartPoleContinuous-v1'
    Adaptive_Beta=True
    Just_Beta = True
    Initial_Beta=1
    start_iteration_number=0 
    
    assert (start_iteration_number < end_iteration_number), f"`start_iteration_number` must be less than `end_iteration_number`. Currently start_iteration_number={start_iteration_number} and end_iteration_number={end_iteration_number}"
    Save_results='./Results_'+envname+'/'
    if Adaptive_Beta:
        if Just_Beta:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Just_Beta'+str(Initial_Beta)+'_s'+str(start_iteration_number)+'_e'+str(end_iteration_number)+'_sig'+str(SIGMA)+'_mu'+str(MU)
        else:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Beta'+str(Initial_Beta)+'_s'+str(start_iteration_number)+'_e'+str(end_iteration_number)+'_sig'+str(SIGMA)+'_mu'+str(MU)
    else:
        if Just_Beta:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Just_Fixed_Beta'+str(Initial_Beta)+'_sig'+str(SIGMA)+'_mu'+str(MU)
        else:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Fixed_Beta'+str(Initial_Beta)+'_sig'+str(SIGMA)+'_mu'+str(MU)
    if Initial_Beta==100:
        logname=envname+'_'+algo+'_only_MSE'+'_Weight'+str(MSE_Weight)+'_sig'+str(SIGMA)+'_mu'+str(MU)
    if Initial_Beta==0:
        logname=envname+'_'+algo+'_only_RL'
    i=1
    while os.path.isdir(Save_results+logname+'_'+str(i)):
        i+=1
    logname=logname+'_'+str(i)
    if SEED==None:
        np.random.seed(SEED)
    env = gym.make('gym_examples/'+ envname) 
    new_logger = configure(Save_results+logname, ["stdout", "csv", "tensorboard"])
    total_timesteps=300000 
    learning_starts=100 
    train_freq= (1, 'episode') 
    model = TD3('MlpPolicy', env, learning_rate=0.001, buffer_size=1000000, learning_starts=learning_starts, 
                batch_size=100, tau=0.005, gamma=0.99, train_freq=train_freq, gradient_steps=-1, 
                action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None, 
                optimize_memory_usage=False, policy_delay=2, target_policy_noise=0.2, 
                target_noise_clip=0.5, tensorboard_log=Save_results, create_eval_env=False, 
                policy_kwargs=None, verbose=1, seed=SEED, device='auto', _init_setup_model=True, 
                Adaptive_Beta=Adaptive_Beta, Just_Beta=Just_Beta, MSE_Weight=MSE_Weight, Initial_Beta=Initial_Beta,  
                start_iteration_number=start_iteration_number , end_iteration_number=end_iteration_number, SIGMA=SIGMA, MU=MU) 

    model.set_logger(new_logger)
    print('Seed:', SEED, '| algo:', algo, '| envname:', envname, '| SIGMA:', SIGMA, '| MU:', MU)
    print('Adaptive_Beta:', Adaptive_Beta, '| Just_Beta:', Just_Beta, '| Initial_Beta:', Initial_Beta, '| MSE_Weight:', MSE_Weight )

    print('total_timesteps', total_timesteps,  '| start_iteration_number:', start_iteration_number , '| end_iteration_number:', end_iteration_number) 
    model.learn(total_timesteps=total_timesteps, tb_log_name=logname) 
    model.save(Save_results+'Weights/'+logname) 
    
run_main(Seed= Seed, MSE_Weight=MSE_Weight, end_iteration_number=end_iteration_number, SIGMA=SIGMA, MU=MU)

