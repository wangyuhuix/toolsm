import sys
import uuid
from warnings import warn

import Pyro4.naming
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, SubprocVecEnvMulti
from discard.run_common import make_env, make_envs_local
from tools import print_refresh

sys.excepthook = Pyro4.util.excepthook
class EnvClient(object):
    def __init__(self, env_id, host, name_remote, seed=0, num_processes=None, name_host=None, random_name=False, **kwargs):
        if isinstance( host, int ):
            host = '172.1.1.%i'%host
        if random_name:
            if name_remote is None:
                name_remote = str( uuid.uuid1() )
            else:
                name_remote += str( uuid.uuid1() )
        self.id_remote = '{}_{}'.format(env_id, name_remote)
        print_refresh( 'Connecting... {}:host-{},id-{}'.format(name_host, host, self.id_remote) )
        self.env_id = env_id
        with Pyro4.locateNS(host=host, port=13579) as ns:
            item = list(ns.list(prefix='rl_envs').items())[0]
            self.env = Pyro4.Proxy(item[1])
        self.env.make_env( env_id=env_id, num_processes=num_processes, id=self.id_remote, seed=seed, **kwargs )
        print_refresh('Connectted!!! {}:host-{},id-{}'.format(name_host, host, self.id_remote))
        print('')

    @staticmethod
    def make_local( env_id, name_remote, seed=0, random_name=False, **kwargs ):
        return EnvClient(env_id, host='172.1.1.0', name_remote=name_remote, seed=seed, num_processes=0,random_name=random_name, **kwargs)


    def step(self, actions):
        return self.env.step(actions, id=self.id_remote)

    def reset(self, *args, **kwargs):
        if hasattr(self, '__reset'):
            warn( '''This method should only be called once!!! \nSince the env have been reset in the subproc''' )
        self.__reset = True
        return self.env.reset(id=self.id_remote, *args, **kwargs)

    def render(self, *arg):
        #warn( ' The render method cannot be rendered ' )
        #return False
        return self.env.render(self.id_remote, *arg)

    def start_video(self,path):
        return self.env.start_video(path, self.id_remote)

    def render_video(self):
        return self.env.render_video(self.id_remote)

    def close_video(self):
        return self.env.close_video(self.id_remote)

    def close(self):
        self.env.close(id=self.id_remote)
        self.env = None
        return True

    def debug(self):
        return self.env.debug()

    def __getattr__(self, name):
        if name not in self.__dict__:
            self.__dict__[name] = self.env.__getattr__(name)(self.id_remote)
        return self.__dict__[name]

def make_envs_remote(env_id, seed, host, id_remote, num_processes=None, name_host=None):

    def _thunk():
        return EnvClient( env_id, seed, host, id_remote, num_processes, name_host=None )
    return _thunk


num_id_goodserver = 6
num_id_normalserver = 2
num_id_usedserver = 1
#分别表示: 主机名,主机ip(172.1.1.x),在主机上运行多少个子进程
_servers \
    = [('jinxin',       1,                  3),
       ('zhangwen',     2,                  0),
       ('zhangweining', 3,                  2),
       ('liucheng',     '172.26.160.234',   num_id_goodserver),
       ('xieyanping',   5,                  0),#num_id_normalserver
       #('sunqiang',     6,                  0),
       ('weiwenge',     7,                  num_id_usedserver)]

def _get_envs_all(env_id, seed, path_logger):
    envs = []
    envs.append(make_envs_local(env_id=env_id, seed=seed, log_dir=path_logger, num_processes=8))  # 待修改

    for name,ip, num_processes in _servers:
        if num_processes > 0:
            envs.append(make_envs_remote(env_id=env_id, seed=seed, host=ip, num_processes=8))
    env = SubprocVecEnvMulti(envs)
    return env

#另外开一个进程来进行模拟，防止出现一些软件包冲突的问题
def make_env_seperate(env_id, seed, name_remote=None, rank=0, log_dir=None, random_name=False, **kwargs):
    def _thunk():
        env = EnvClient.make_local( env_id=env_id, name_remote=name_remote, seed=seed+rank, random_name=random_name, **kwargs )
        return env

    return _thunk


def _get_envs_all_list(env_id, seed, path_logger, num_processes=4, num_envs=1):
    global _servers
    env = []
    #临时

    if env_id != 'Run':
        '''
        env.append(SubprocVecEnv([
            make_env(env_id, seed, rank=i, log_dir=path_logger, visualize=False)
            for i in range(num_processes)
        ]))
        '''
        for i in range(num_envs):
            env.append( SubprocVecEnv([
                make_env(env_id, seed, rank=i, log_dir=path_logger, visualize=False)
                for i in range(num_processes)
            ]))
        return env

    for i in range(1):
        env.append( SubprocVecEnv([
            make_env(env_id, seed, rank=i, log_dir=path_logger, visualize=False)
            for i in range(num_processes)
        ]))
    return env
    _servers = [_servers[0]]
    for name,host, num_ids in _servers:
        for id in range(num_ids):
            env.append(EnvClient(env_id, seed, host, name_remote=id, num_processes=num_processes, name_host=name))
    return env


from PIL import Image
from . import tools
if __name__ == '__main__':
    import  numpy as np
    #env = EnvClient(env_id='Pendulum-v0', seed=0, host='172.1.1.0', id_remote=1, num_processes=3 )
    #env = EnvClient(env_id='RoboschoolInvertedPendulum-v1', seed=0, host='172.1.1.0', id_remote=1, num_processes=1)

    # env = EnvClient.make_local( env_id='HalfCheetah-v1', name_remote='a' )
    #env = EnvClient.make_local(env_id='RoboschoolInvertedPendulum-v1')
    import itertools
    # env.start_video('/tmp/gym1')
    import gym
    env = gym.make('Ant-v1')
    print(env.observation_space)
    exit()
    while True:

        env.reset()
        arr_pre = env.render('rgb_array')
        a = []
        tools.mkdir( '/media/d/e/v/t/{}/'.format( ind_test )  )
        for i in itertools.count():
            # if i>100:
                # env.close_video()
                # break
            print(i)
            ac = env.action_space.sample()
            #acs = np.array([ac]*env.num_envs)
            ob, reward, done, info = env.step(ac)
            # env.render()
            # x =
            arr = env.render('rgb_array')
            # print(ac)
            if np.all( arr == arr_pre ):
                print('restart env')
                print(ac)
                # exit()
                env = gym.make('HalfCheetah-v1')
                break
            im = Image.fromarray( arr )
            # a.append(arr)
            im.save( '/media/d/e/v/t/{}/{:0>4}.jpg'.format( ind_test, i) )
            arr_pre = arr
            if done:
                break
            # env.render_video()
            #env.render()
        # save_vars('/media/d/e/v/t/{}/data.pkl'.format(ind_test, i), a)
    env.close()
