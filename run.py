import json
import os
import subprocess
from multiprocessing import Pool

import tools

# ----------------------- clipped type ----------------

# settings = dict(
#     clipped_type = ['origin', 'kl2clip'],
#     cliprange = [0.2]
# )
settings = dict(
    # optpolicytype = ['PolicyGradient'],entropy = [False],
    optpolicytype = ['kl'],
    #cliptype=['ratio'],clippingrange = [0.2],
    # cliptype=['kl'],delta_kl = [0.025],
    # policy_old_update_interval = [10],
)
# settings = dict(
#     stepsize_mult = [1.4]#,2.2,2.6
# )
file_exec = 'examples.mujoco_all_sac'

# settings = dict(
#     clipped_type = ['judgekl3'],
#     delta_kl = [0.01],
#     slope = [-0.05, -0.1, -0.2],
# )
# ------------------------ common part-----------------
N_PARALLEL = 3  # 同时运行几个进程
seed = [684, 559, 629]
ARGS_common = dict(
    name_group = '',
    keys_group=list(settings.keys()),
    force_write=-1, #-1: not overide; 0: ask; 1:overide
)
if N_PARALLEL <=1 or ARGS_common['force_write'] > 0:
    input(f"N_PARALLEL:{N_PARALLEL}, force_write:{ARGS_common['force_write']} Continue?")

n_epochs= 10
env_setting = dict(
    # InvertedPendulum=dict(),
    # InvertedDoublePendulum=dict( ),
    # HalfCheetah=dict(n_epochs=int(n_epochs*1e2)),
    # Walker2d=dict(n_epochs=int(n_epochs*1e2)),
    # Hopper=dict(n_epochs=int(n_epochs*1e2)),
    # Swimmer=dict(n_epochs=int(n_epochs*1e2)),
    # Reacher=dict(n_epochs=int(n_epochs*1e2)),
    Humanoid=dict(n_epochs=int(n_epochs*2e3)),
    # Ant=dict(),
)
env = list(env_setting.keys())
settings.update(dict(env=env, seed=seed))



def run_all_pool():
    args_all = []
    if tools.ispc('this'):
        args_base = ['/home/this/miniconda3/envs/rl/bin/python3.6', '-m', file_exec]
    else:
        args_base = ['python', '-m', file_exec]
    args_values = tools.multiarr2meshgrid(settings.values())  # 把dict里所有数组进行排列组合
    for ind in range(args_values.shape[0]):
        # 复制默认的args
        args_dict = ARGS_common.copy()

        # 添加特别任务特定类型(clippedtype)的args
        args_value = args_values[ind]
        for ind_key, key in enumerate(settings.keys()):
            args_dict[key] = args_value[ind_key]

        # 把任务特殊的定义复制进来
        args_dict.update( env_setting[ args_dict['env'] ] )
        args_dict['env'] = f"{args_dict['env']}"

        # 转换为参数形式
        args = args_base.copy()
        for key in args_dict:
            args += [f'--{key}', str(args_dict[key])]
        print(args)
        args_all.append(args)
    print(len(args_all))
    # exit()
    with tools.timed(f'len(args_all):{len(args_all)}, N_PARALLEL:{N_PARALLEL}', print_atend=True):
        p = Pool(N_PARALLEL)
        p.map(start_process, args_all)
        p.close()  # Close Pool
        p.join()  # Wait for all tasks


# TODO: read json file to judge whether to stop now!
def judge_whether_continue(file_path, keys):
    file_path = os.path.join(file_path, 'run.json')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            default_config = {'continue': True}
            json.dump(default_config, f)

    with open(file_path, 'r') as f:
        j = json.load(f)
    for key in keys:
        if not j[key]:
            # print(multiprocessing.current_process())
            return False
    return True


def start_process(args):
    keys_start = ['continue', ]
    whether_continue = judge_whether_continue(file_path=os.path.join(os.getcwd()), keys=keys_start)
    if not whether_continue:
        print('run.json shows stop')
        return
    subprocess.check_call(args)



if __name__ == '__main__':
    run_all_pool()
