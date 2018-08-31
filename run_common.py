#易错!!!python很多东西是直接返回的
import numpy as np
import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import wrap_deepmind
import roboschool
from osim.env.run import RunEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


INDS_ob = list(range(41))
INDS_x = [18] + INDS_ob[22:36:2] + [38]
IND_pelvis = 1

def get_ob_norm(ob, new=True):
    if new or isinstance(ob, list):
        ob = np.array(ob)
    ob[INDS_x] = ob[INDS_x] - ob[IND_pelvis:IND_pelvis + 1]
    ob[IND_pelvis] = 0
    return ob

def get_obs_norm(obs, new=True):
    if new:
        obs = obs.copy()
    obs[:, INDS_x] = obs[:, INDS_x] - obs[:, IND_pelvis:IND_pelvis + 1]
    obs[:, IND_pelvis] = 0
    return obs


def make_env(env_id, seed, rank=0, log_dir=None, *args, **kwargs):
    def _thunk():
        if env_id == 'Run':
            env = RunEnv(*args, **kwargs)
        else:
            env = gym.make(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env,
                                os.path.join(log_dir,
                                             "{}.monitor.json".format(rank)))
        # Ugly hack to detect atari.
        if env.action_space.__class__.__name__ == 'Discrete':
            env = wrap_deepmind(env)
            env = WrapPyTorch(env)
        return env

    return _thunk
from multiprocessing import cpu_count
def make_envs_local(env_id, seed, log_dir=None, num_processes=None,  *args, **kwargs ):
    if num_processes is None:
        num_processes = cpu_count()
    def _thunk():
        return SubprocVecEnv([
            make_env(env_id, seed, rank=i, log_dir=None, *args, **kwargs)
            for i in range(num_processes)
        ])
    return _thunk


def ResetEnv(e,env_id):
    if env_id == 'Run':
        return e.reset(difficulty=0)
    else:
        return e.reset()

def get_ob_origin(ob,new=False):
    return ob

def get_ob_current( ob, new=True ):
    if new or isinstance(ob, list):
        ob = np.array(ob)
    ob[ INDS_x] = ob[INDS_x] - ob[ IND_pelvis:IND_pelvis + 1]
    ob[IND_pelvis] = 0
    ob[38] = 100
    ob[[39,40]] = 0
    return ob

def get_obs_current( ob , new=True):
    if new:
        ob = np.array(ob)
    ob[:, INDS_x] = ob[:, INDS_x] - ob[:, IND_pelvis:IND_pelvis + 1]
    ob[:, IND_pelvis] = 0
    ob[:,38] = 100
    ob[:,[39,40]] = 0
    return ob


def get_noise_gaussian( noise ):
    def fn_entity( ob, new=True ):
        if new or isinstance(ob, list):
            ob = np.array(ob)
        ob += np.random.normal(0, noise, size=ob.shape )
        return ob
    def fn_ori( ob, new=True ):
        return ob
    if noise > 0:
        return fn_entity
    else:
        return fn_ori





class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

