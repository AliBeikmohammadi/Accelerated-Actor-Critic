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
from stable_baselines3.sac.policies import SACPolicy
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
  help='Seed; default 900598021',
  type=int,  
  default=900598021,
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
  "--Noise_STD",
  help='Noise_STD; default 0',
  type=float, 
  default=0,
)
arg_pass.add_argument(
  "--Noise_Prob",
  help='Noise_Prob; default 0',
  type=float, 
  default=0,
)
args = arg_pass.parse_args()

Seed= args.Seed
MSE_Weight=args.w
end_iteration_number=args.e
Noise_STD=args.Noise_STD
Noise_Prob=args.Noise_Prob

class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

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
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
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
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        Adaptive_Beta: bool = True, Just_Beta: bool = True, MSE_Weight: float = 1, Initial_Beta: float = 1,  start_iteration_number: int = 0 , end_iteration_number: int = 10, SIGMA: float = 0, MU: float = 0):


        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

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
        super(SAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

########################################################################################################################################################################################
        RL_losses, MSE_losses = [], []
        Main_losses= []
        self.total_iteration_number=np.ceil((self._total_timesteps-self.learning_starts)/(self.env.num_envs * self.train_freq[0]))-1
        self.replay_counter=self._n_updates/gradient_steps

        assert (self.sp< self.ep
               ), f"`start_iteration_number` must be less than `end_iteration_number`. Currently start_iteration_number={self.sp} and end_iteration_number={self.ep}"
        assert (self.sp< self.total_iteration_number
               ), f"`start_iteration_number` must be less than `total_iteration_number`. Currently start_iteration_number={self.sp} and ceil((_total_timesteps-learning_starts)/(n_num_envs*train_freq))={self.total_iteration_number}"
        
        if self.adaptive_beta:
            self.beta = max(min((self.replay_counter-self.ep)*self.initial_beta/(self.sp-self.ep), self.initial_beta),0)
        else: 
            self.beta=self.initial_beta
        if self.initial_beta == 100:
            self.beta=1
###########################################################################################################################################################################################

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            #print(replay_data)
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
###########################################################################################################################################################################################
             ###hu environment
#                     x, y, angle_velocity = observation
#                     flip = (y < 0.)
#                     if flip:
#                         y *= -1. # now y >= 0
#                         angle_velocity *= -1.
#                     angle = np.arcsin(y)
#                     if x < 0.:
#                         angle = np.pi - angle
#                     if (angle < -0.3 * angle_velocity) or \
#                             (angle > 0.03 * (angle_velocity - 2.5) ** 2. + 1. and \
#                             angle < 0.15 * (angle_velocity + 3.) ** 2. + 2.):
#                         force = 2.
#                     else:
#                         force = -2.

#                     if flip:
#                         force *= -1.
#                     action = np.array([force,])
            observations = replay_data.observations.float() 
            x, y, angle_velocity = observations.cuda()[:,0], observations.cuda()[:,1], observations.cuda()[:,2]
            flip = th.lt(y, 0.)
            y= th.where(flip, -1.*y, y)
            angle_velocity= th.where(flip, -1.*angle_velocity, angle_velocity)
            angle = th.asin(y)
            th.pi = th.acos(th.zeros(1)).item() * 2
            angle = th.where(x < 0., th.pi - angle, angle)
            force = th.where((angle < -0.3 * angle_velocity) | ((angle > 0.03 * (angle_velocity - 2.5) ** 2. + 1.) & (angle < 0.15 * (angle_velocity + 3.) ** 2. + 2.)), 2., -2.)
            force= th.where(flip, -1.*force, force)
            hu_actions=force.view(actions_pi.shape[0],actions_pi.shape[1]) 
            #Noise type 1
            hu_actions=th.where(th.rand((actions_pi.shape[0], actions_pi.shape[1]), device=th.device("cuda")) >= self.mu, hu_actions, self.sigma * th.randn((actions_pi.shape[0], actions_pi.shape[1]), device=th.device("cuda")))
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
                        loss = (self.hu_weight * self.beta * MSE_loss) + (1-self.beta) * actor_loss            
            Main_losses.append(loss.item()) 
###########################################################################################################################################################################################
            
            # Optimize the actor
            self.actor.optimizer.zero_grad()
###########################################################################################################################################################################################
            #actor_loss.backward()
            loss.backward()
###########################################################################################################################################################################################
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

###########################################################################################################################################################################################
        self.logger.record("train/RL_loss", actor_loss.item())
        self.logger.record("train/MSE_loss", MSE_loss.item())
        self.logger.record("train/MSE_losses", np.mean(MSE_losses))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/Main_losses", np.mean(Main_losses))
        self.logger.record("train/Beta", self.beta)
        self.logger.record("train/critic_los", critic_loss.item())
###########################################################################################################################################################################################
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
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
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


def run_main(Seed= 900598021, MSE_Weight=1, end_iteration_number=10, Noise_Prob=0, Noise_STD=0):
    SEED=Seed 
    algo= 'SAC'
    envname='Pendulum-v1'
    Adaptive_Beta=True
    Just_Beta = True
    Initial_Beta=1
    start_iteration_number=0
    
    assert (start_iteration_number < end_iteration_number), f"`start_iteration_number` must be less than `end_iteration_number`. Currently start_iteration_number={start_iteration_number} and end_iteration_number={end_iteration_number}"
    Save_results='./Results_'+envname+'/'
    if Adaptive_Beta:
        if Just_Beta:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Just_Beta'+str(Initial_Beta)+'_s'+str(start_iteration_number)+'_e'+str(end_iteration_number)+'_Noise_Prob'+str(Noise_Prob)+'_Noise_STD'+str(Noise_STD)
        else:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Beta'+str(Initial_Beta)+'_s'+str(start_iteration_number)+'_e'+str(end_iteration_number)+'_Noise_Prob'+str(Noise_Prob)+'_Noise_STD'+str(Noise_STD)
    else:
        if Just_Beta:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Just_Fixed_Beta'+str(Initial_Beta)+'_Noise_Prob'+str(Noise_Prob)+'_Noise_STD'+str(Noise_STD)
        else:
            logname=envname+'_'+algo+'_Weight'+str(MSE_Weight)+'_Fixed_Beta'+str(Initial_Beta)+'_Noise_Prob'+str(Noise_Prob)+'_Noise_STD'+str(Noise_STD)
    if Initial_Beta==100:
        logname=envname+'_'+algo+'_only_MSE'+'_Weight'+str(MSE_Weight)+'_Noise_Prob'+str(Noise_Prob)+'_Noise_STD'+str(Noise_STD)
    if Initial_Beta==0:
        logname=envname+'_'+algo+'_only_RL'
    i=1
    while os.path.isdir(Save_results+logname+'_'+str(i)):
        i+=1
    logname=logname+'_'+str(i)
    if SEED==None:
        np.random.seed(SEED)
    env = gym.make(envname)
    new_logger = configure(Save_results+logname, ["stdout", "csv", "tensorboard"])
    total_timesteps=60000
    learning_starts=100
    train_freq=1
    model = SAC('MlpPolicy', env, learning_rate= 0.001, buffer_size=1000000, 
                 learning_starts=learning_starts, batch_size=256, tau=0.005, gamma=0.99, 
                 train_freq=train_freq, gradient_steps=1, action_noise=None, 
                 replay_buffer_class=None, replay_buffer_kwargs=None, 
                 optimize_memory_usage=False, ent_coef='auto', 
                 target_update_interval=1, target_entropy='auto', 
                 use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False, 
                 tensorboard_log=Save_results, create_eval_env=False, policy_kwargs=None, 
                 verbose=1, seed=SEED, device='auto', _init_setup_model=True, 
                 Adaptive_Beta=Adaptive_Beta, Just_Beta=Just_Beta, MSE_Weight=MSE_Weight, Initial_Beta=Initial_Beta,  
                 start_iteration_number=start_iteration_number , end_iteration_number=end_iteration_number, SIGMA=Noise_STD, MU=Noise_Prob)
    model.set_logger(new_logger)        
    total_iteration_number=np.ceil((total_timesteps-learning_starts)/(1 * train_freq))-1
    print('Seed:', SEED, '| algo:', algo, '| envname:', envname, '| Noise_Prob:', Noise_Prob, '| Noise_STD:', Noise_STD)
    print('Adaptive_Beta:', Adaptive_Beta, '| Just_Beta:', Just_Beta, '| Initial_Beta:', Initial_Beta, '| MSE_Weight:', MSE_Weight )
    print('total_timesteps', total_timesteps, '| total_iteration_number:', total_iteration_number,  '| start_iteration_number:', start_iteration_number , '| end_iteration_number:', end_iteration_number)
    model.learn(total_timesteps=total_timesteps, tb_log_name=logname)
    model.save(Save_results+'Weights/'+logname)

run_main(Seed= Seed, MSE_Weight=MSE_Weight, end_iteration_number=end_iteration_number, Noise_Prob=Noise_Prob, Noise_STD=Noise_STD)

