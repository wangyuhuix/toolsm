import fcntl
import tools

from multiprocessing import Process, Pool
import subprocess

import os



def args_NameAndValues2args_list(args_NameAndValues:dict, args_default:dict={}, args_list = []):
    '''
    This function will do the following three things:
    1. Make product for args_NameAndValues and merge it into args_list;
    2. Merge args_default into each group of args;
    3. Merge specified settting into each group of args;
    An example:
    INPUT:
    args_NameAndValues = dict(
        clipped_type = ['ratio'],
        clippingrange= [0.2,0.3],
        env = dict(
            HalfCheetah=dict(ac_fn='relu'),
            Humanoid=dict(num_timesteps=int(2e7))
        )
    )
    OUTPUT:
    {'clipped_type': 'ratio', 'clippingrange': 0.2, 'env': 'HalfCheetah', 'ac_fn': 'relu'}
{'clipped_type': 'ratio', 'clippingrange': 0.2, 'env': 'Humanoid', 'num_timesteps': 20000000}
{'clipped_type': 'ratio', 'clippingrange': 0.3, 'env': 'HalfCheetah', 'ac_fn': 'relu'}
{'clipped_type': 'ratio', 'clippingrange': 0.3, 'env': 'Humanoid', 'num_timesteps': 20000000}
    ]
    '''

    ''' 
    1. Translate the value of key 'env' into 
        env = [ 'HalfCheetah', 'Humanoid' ]
    2. Copy specified settings into a new variable setting_specified_all
        setting_specified_all = dict(
            env = dict(
                HalfCheetah=dict(ac_fn='relu')
                Humanoid=dict(num_timesteps=int(2e7)),
            )
        )
    '''
    setting_specified_all = {}
    for argname, argvalue in args_NameAndValues.items():
        if isinstance( argvalue, dict ):#This means that it is speicified settings
            setting_specified_all[argname] = argvalue
            args_NameAndValues[argname] = list(argvalue.keys())

    # args_values = tools.multiarr2meshgrid(args_NameAndValues.values())
    import itertools
    args_values = itertools.product( * args_NameAndValues.values() )
    # for ind in range(args_values.shape[0]):
        # args_value = args_values[ind]
    for _, args_value in enumerate(args_values):
        args = {}
        for ind_key, argname in enumerate(args_NameAndValues.keys()):
            args[argname] = args_value[ind_key]
        args_list.append(args)

    for ind in range(len(args_list)):

        args_assembled = args_list[ind]
        # Copy args_default
        args = args_default.copy()
        args.update(args_assembled)

        # Copy settings_specified
        '''
        Update the setting with the specified value:
        The preprocessed value are:
            args = dict(
                    clipped_type='ratio', clippingrange=0.2,
                    env='HalfCheetah'
                )
            settings_specified = dict(
                    env = dict(
                        HalfCheetah=dict(ac_fn='relu')
                        Humanoid=dict(num_timesteps=int(2e7)),
                    )
                )
        The processed value are:
            args = dict(
                    clipped_type='ratio', clippingrange=0.2,
                    env='HalfCheetah',
                    ac_fn='relu'
                )
        '''
        for argname, setting_speicified in setting_specified_all.items():
            args.update(  setting_speicified[ args[argname] ] )

        args_list[ind] = args
    return args_list

# if __name__ == '__main__':
#     args_NameAndValues = dict(
#         clipped_type = ['ratio'],
#         clippingrange= [0.2,0.3],
#         env = dict(
#             HalfCheetah=dict(ac_fn='relu'),
#             Humanoid=dict(num_timesteps=int(2e7))
#         )
#     )
#     args_list = args_NameAndValues2args_list( args_NameAndValues )
#     for args in args_list:
#         print(args)
#     exit()



def run_script_parallel(script,  args_NameAndValues: dict=None, args_default:dict={}, args_list:list=[], n=1):
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
            alg_args = dict(
                SARSA_lambda=dict(n=[1,2,3])
            )
        )
        For env=HalfCheetah, we use ac_fn=relu;
        and for env=Humanoid, we use num_timesteps=int(2e7);
    '''

    assert isinstance(args_list, list)
    args_list = args_NameAndValues2args_list( args_NameAndValues, args_default, args_list  )
    args_call_all = []
    args_call_base = ['python', '-m', script]
    print( ' '.join(args_call_base) )
    for ind, args in enumerate(args_list):
        args_call = args_call_base.copy()
        args_call_str = []
        for argname, arg_value in args.items():
            if isinstance( arg_value, dict ):
                import json
                arg_value = json.dumps(arg_value, separators=(',',':'))
            else:
                arg_value = str(arg_value)
            args_call += [f'--{argname}', arg_value]
            args_call_str += [ tools.colorize(f'-{argname}', color='black', bold=False), tools.colorize( str(arg_value) , 'green', bold=True )  ]
        print( ' '.join(args_call_str) )
        args_call_all.append( (args_call, ind, len(args_list)))
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
    print( tools.colorize( f'Process: {ind+1}/{n_all}', 'blue' ))
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
