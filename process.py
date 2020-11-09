import fcntl
from . import tools

from multiprocessing import Process, Pool
import subprocess

import os


NOARGVALUE = '__NOARGVALUE__'


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
    assert isinstance(args_list, list)
    args_list = args_list.copy() #!important
    setting_specified_all = {}
    for argname, argvalue in args_NameAndValues.items():
        if isinstance( argvalue, dict ):#This means that it is speicified settings
            setting_specified_all[argname] = argvalue
            args_NameAndValues[argname] = argvalue = list(argvalue.keys())

        # TODO: debug, may be wrong
        if not isinstance(argvalue, list):
            args_NameAndValues[argname] = [argvalue]


    # args_values = tools.multiarr2meshgrid(args_NameAndValues.values())
    if len( args_NameAndValues )>0:#We need to judge it. Otherwise when args_NameAndValues={}, itertools.product could output {()}
        import itertools
        args_values = itertools.product( * args_NameAndValues.values() )
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



def run_script_parallel(script, args_NameAndValues: dict={}, args_default:dict={}, args_list:list=[], n=1, debug=False, log_kwargs=dict(path='/tmp/', file_basename=None) ):
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

    import json
    from toolsm.logger import Logger
    logger = Logger( formats='log', **log_kwargs )
    args_NameAndValues_str = 'args_NameAndValues:\n' + json.dumps(args_NameAndValues, indent=4, separators=(',', ':'))
    args_default_str = 'args_default:\n' + json.dumps(args_default, indent=4, separators=(',', ':'))
    logger.log_str( args_NameAndValues_str )
    print(args_NameAndValues_str)
    logger.log_str(args_default_str )
    print(args_default_str)
    logger.log_str( f'n={n}' )
    print( f'n={n}' )

    args_default = args_default.copy()
    args_list = args_list.copy()
    args_list = args_NameAndValues2args_list( args_NameAndValues, args_default, args_list  )
    args_call_all = []
    if tools.ispc('pc211') or tools.ispc('test'):
        python = f'/home/{tools.pcname()}/anaconda3/envs/ed/bin/python'
    else:
        python = 'python'
    args_call_base = [python, '-m', script]
    print( 'call command:' , ' '.join(args_call_base) )

    for ind, args in enumerate(args_list):
        args_call = args_call_base.copy()
        args_call_str = []
        for argname, arg_value in args.items():
            args_call += [f'--{argname}']
            args_call_str += [tools.colorize(f'{argname}', color='black', bold=False)]
            if arg_value != NOARGVALUE:
                if isinstance( arg_value, dict ):
                    arg_value = json.dumps(arg_value, separators=(',',':'))
                else:
                    arg_value = str(arg_value)
                args_call += [ arg_value ]
                args_call_str += [ tools.colorize( str(arg_value) , 'green', bold=True )  ]
        logger.log_str( json.dumps(args, indent=4, separators=(',', ':')) )
        print( ' '.join(args_call_str) )
        args_call_all.append( dict(args_call=args_call, ind=ind, n_total=len(args_list)))
    print( f'PROCESS COUNT: {len(args_call_all)}' )
    if debug:
        exit()
    # exit()
    #TODO: log

    # def call_back(*args, **kwargs):
    #     print(f'completed. args{args}. kwargs{kwargs}')
    from tqdm import tqdm
    # import matplotlib
    # matplotlib.use('TkAgg')#It seems that tqdm_gui oly work for TkAgg mode
    import time
    # with tools.timed(f'len(args_all):{len(args_call_all)}, N_PARALLEL:{n}', print_atend=True):
    tstart = time.time()
    logger.log_str(f'time:{tools.time_now2str()}, count: {len(args_call_all)}')
    with Pool(n) as p:
         with tqdm(enumerate(p.imap_unordered(start_process, args_call_all)), total=len(args_call_all)) as processbar:
             for ind,info in processbar:
                 processbar.set_description(f'process')
                 info_str = json.dumps(info, indent=4, separators=(',', ':'))
                 logger.log_str(f'process:{ind}/{len(args_call_all)}')
                 logger.log_str( info_str )
    logger.log_str(f'time:{tools.time_now2str()}, time_cost:{time.time()-tstart} sec, count: {len(args_call_all)}')
    logger.close()




def judge_continue(file_path, keys):
    key = 'continue'
    file_path = os.path.join(file_path, 'run.json')
    if not os.path.exists(file_path):
        tools.save_json( file_path, {key: True} )

    j = tools.load_json( file_path )
    return j[key]


def start_process(info):
    args_call, ind, n_total = info['args_call'], info['ind'], info['n_total']
    # print( tools.colorize( f'Process: {ind+1}/{n_total}', 'blue' ))
    keys_start = ['continue', ]
    continue_ = judge_continue(file_path=os.path.join(os.getcwd()), keys=keys_start)
    if not continue_:
        print('run.json shows stop')
        return
    import time
    time_start = time.time()
    err_msgs = []
    for i in range(2):
        try:
            subprocess.check_call(args_call)
            break
        except Exception as e:
            import time
            seconds_sleep = i*5
            print(
f'''An error happened, sleep for {seconds_sleep}s!
args_call:{args_call}
Exception: {e}''')
            time.sleep(seconds_sleep)
            err_msgs.append( str(e) )
    info.update( err_msgs=err_msgs, trial_times=i+1, time_cost=time.time()-time_start )

    return info

def tes_run_script_parallel():
    run_script_parallel(script='_t',
        args_list=[dict(i=i) for i in range(10)],
        n=2,
        debug=0
    )


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

def tes_filelocker():

    from . import tools
    import time
    with FileLocker('t/a.locker'):
        with open('t/a.txt','a') as f:
            f.write(tools.time_now2str() + '\n')
        time.sleep(5)

if __name__ == '__main__':
    tes_run_script_parallel()
    exit()
