import fcntl
import tools

from multiprocessing import Process, Pool
import subprocess

import os


def run_script_parallel(scipt, args_default={}, args_unassembled_all: dict=None, args_assembled_all:list=[], n=1):
    '''
    priority: args_default < args_dict_unassembled = args_assembled < args_specifies
    The low priority defined args would be overwritten by high priority defined args
    :param scipt:
    :param args_default: default args
    :param args_unassembled_all: args dict which is unassembled
        dict(clippingrange:[1, 2], hidden_sizes:[64] )
    :param args_assembled_all: args dict which is assembled
        [ dict(clippingrange=1, hidden_sizes:64), dict(clippingrange=2, hidden_sizes:64) ]
        An example:
        dict(
            clipped_type = ['ratio'],
            clippingrange= [0.2,0.3]
            seed = [684, 559, 629],
            env = dict(
                HalfCheetah=dict(ac_fn='relu')
                Humanoid=dict(num_timesteps=int(2e7)),
            )
        )
    '''

    assert isinstance(args_assembled_all, list)

    # assemble args_unassembled_all
    args_specified = {}
    if args_unassembled_all is not None:
        # specified values
        for argname, arg_value_specifiedargs in args_unassembled_all.items():
            if isinstance( arg_value_specifiedargs, dict ):
                args_specified[argname] = arg_value_specifiedargs
                args_unassembled_all[argname] = list(arg_value_specifiedargs.keys())

        # 把dict里所有数组进行排列组合
        args_values = tools.multiarr2meshgrid(args_unassembled_all.values())
        for ind in range(args_values.shape[0]):
            args_unassembled = {}
            args_value = args_values[ind]
            for ind_key, argname in enumerate(args_unassembled_all.keys()):
                args_unassembled[argname] = args_value[ind_key]
            args_assembled_all.append(args_unassembled)

    args_call_all = []
    args_call_base = ['python', '-m', scipt]
    print( ' '.join(args_call_base) )
    for ind, args_assembled in enumerate(args_assembled_all):
        # 复制默认参数
        args = args_default.copy()

        # 组合后的参数
        args.update(args_assembled)

        # 把特殊的声明复制进来
        for argname, arg_value_specifiedargs in args_specified.items():
            args.update(  arg_value_specifiedargs[ args[argname] ] )

        # 转换为参数形式
        args_call = args_call_base.copy()
        args_call_str = []
        for argname, arg_value in args.items():
            args_call += [f'--{argname}', str(arg_value)]
            args_call_str += [ f'-{argname}', tools.colorize( str(arg_value) , 'green' )  ]
        print( ' '.join(args_call_str) )
        args_call_all.append( (args_call, ind, len(args_assembled_all)))
    print( f'PROCESS COUNT: {len(args_call_all)}' )
    # exit()
    #TODO: log
    with tools.timed(f'len(args_all):{len(args_call_all)}, N_PARALLEL:{n}', print_atend=True):
        p = Pool(n)
        p.map(start_process, args_call_all)
        p.close()
        p.join()



def judge_continue(file_path, keys):
    key = 'continue'
    file_path = os.path.join(file_path, 'run.json')
    if not os.path.exists(file_path):
        tools.save_json( file_path, {key: True} )

    j = tools.load_json( file_path )
    return j[key]


def start_process(args_info):
    args, ind, n_all = args_info
    print( tools.colorize( f'Process: {ind}/{n_all}', 'blue' ))
    keys_start = ['continue', ]
    continue_ = judge_continue(file_path=os.path.join(os.getcwd()), keys=keys_start)
    if not continue_:
        print('run.json shows stop')
        return
    finish = False
    for i in range(10):
        try:
            subprocess.check_call(args)
            break
        except Exception as e:
            import time
            seconds_sleep = i*5
            print(f'An error happened, sleep for {seconds_sleep}s! Exception: {e}')
            time.sleep(seconds_sleep)




class FileLocker:
    def __init__(self, filename):
        self.__filename = filename
        pass

    def acquire(self):
        self.file =  open(self.__filename, 'w+')
        tools.print_refresh(f'acquire {self.__filename}')
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        tools.print_refresh('')

    def release(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()

    def __enter__(self):
        # print(f'acquire file locker {self.__filename}')
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f'release file locker {self.__filename}')
        self.release()

if __name__ == '__main__':
    import tools
    import time
    with FileLocker('t/a.locker'):
        with open('t/a.txt','a') as f:
            f.write(tools.time_now_str()+'\n')
        time.sleep(5)

    exit()
