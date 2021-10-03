import argparse
import os
import time
import warnings
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
import optuna
import yaml
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage

from torch import nn as nn

import utils.import_envs
from utils.callbacks import SaveVecNormalizeCallback, TrialEvalCallback
from utils.hyperparams_opt import HYPERPARAMS_SAMPLER
from utils.utils import ALGOS, get_callback_list, get_latest_run_id, get_wrapper_class, linear_schedule

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from torch import Tensor


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K, L = 2000, 200
noise = dict()

def add_noise(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            n = 2 * torch.randn(m.weight.size()) / np.prod(m.weight.size()) * np.sqrt(np.linalg.norm(m.weight))
            noise[m] = n
            m.weight.add_(n)

def remove_noise():
    with torch.no_grad():
        for m in noise:
            m.weight.sub_(noise[m])
    noise.clear()


class DiscreteActionGapCallback(BaseCallback):
    def __init__(self, exp_manager, freq=10000, verbose=0):
        super(DiscreteActionGapCallback, self).__init__(verbose)
        self.n_calls = 0
        self.freq = freq
        self.exp_manager = exp_manager

    def _on_step(self) -> bool:
        model = self.exp_man.cur_model
        self.n_calls += 1
        if self.n_calls % self.freq == 0:
            obs = Tensor(model.replay_buffer.observations[-K:]).to(device=dev)
            sz = obs.size()
            obs = torch.reshape(obs, (K, ) + sz[2:])
            pred = model.q_net_target.forward(obs).cpu().detach().numpy()
            sorted = np.sort(pred, axis=1)
            abs_value = np.mean(sorted[:,-1] - sorted[:,-2])
            rel_value = np.mean(sorted[:,-1] / sorted[:,-2])
            self.logger.record('discrete_action_gap_abs', abs_value)
            self.logger.record('discrete_action_gap_rel', rel_value)
        return True


def get_actor_action_gap(model):
    with torch.no_grad():
        obs = model.replay_buffer.sample(K, env=model._vec_normalize_env).observations
        actions, _ = model.predict(obs, deterministic=True)
        action_gap = 0
        for i in range(L):
            model.actor.apply(add_noise)
            distorted_actions, _ = model.predict(obs, deterministic=True)
            remove_noise()
            action_gap += np.square(actions - distorted_actions).sum()
    return action_gap / (K * L)

def get_critic_action_gap(model):
    with torch.no_grad():
        obs = model.replay_buffer.sample(K, env=model._vec_normalize_env).observations
        actions, _ = model.predict(obs, deterministic=True)
        q_values_pi = torch.cat(model.critic.forward(obs, Tensor(actions)), dim=1)
        action_gap = 0
        for i in range(L):
            model.actor.apply(add_noise)
            distorted_actions, _ = model.predict(obs, deterministic=True)
            remove_noise()
            q_values_distorted_pi = torch.cat(model.critic.forward(obs, Tensor(distorted_actions)), dim=1)
            action_gap += (q_values_pi - q_values_distorted_pi).detach().numpy().sum()
    return action_gap / (K * L)

def get_cont_action_gap(model):
    with torch.no_grad():
        obs = model.replay_buffer.sample(K, env=model._vec_normalize_env).observations
        actions, _ = model.predict(obs, deterministic=True)
        q_values_pi = torch.cat(model.critic.forward(obs, Tensor(actions)), dim=1)
        action_gap = 0
        for i in range(L):
            model.actor.apply(add_noise)
            distorted_actions, _ = model.predict(obs, deterministic=True)
            remove_noise()
            q_values_distorted_pi = torch.cat(model.critic.forward(obs, Tensor(distorted_actions)), dim=1)
            action_gap += np.abs(actions - distorted_actions)[:,0] \
                            @ (q_values_pi - q_values_distorted_pi).detach().numpy()[:,0]
    return action_gap / (L * K)


class ContinuousActionGapCallback(BaseCallback):
    def __init__(self, exp_man, freq=10000, verbose=0):
        super(ContinuousActionGapCallback, self).__init__(verbose)
        self.n_calls = 0
        self.freq = freq
        self.exp_man = exp_man

    def _on_step(self) -> bool:
        model = self.exp_man.cur_model
        self.n_calls += 1
        if self.n_calls % self.freq == 0:
            self.logger.record('actor_action_gap', get_actor_action_gap(model))
            self.logger.record('critic_action_gap', get_critic_action_gap(model))
            self.logger.record('continuous_action_gap', get_cont_action_gap(model))
        return True