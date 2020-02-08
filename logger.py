from . import tools
import os.path as osp
import json
import os
import itertools
import numpy as np

import argparse


from dotmap import DotMap
def arg2str(i):
    if isinstance(i, float):
        if float.is_integer(i):
            i = int(i)
        else:
            i = f'{i:g}'
    if isinstance(i, int):
        if i >= int(1e4):
            i = f'{i:.0e}'
    if isinstance(i, DotMap ):
        i = i.toDict()
    if isinstance(i, dict):
        i = tools.json2str_file(i, remove_brace=False)
    return i

# print( int2str(20000) )


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def get_path_package(package):
    import inspect
    if inspect.ismodule(package):
        package = os.path.dirname(package.__file__)
    return package

def is_release(package_or_path):
    '''

    :param path_result_release:
    :type path_result_release:
    :return: Whether the code is the released version
    :rtype:
    '''
    return os.path.exists(os.path.join(get_path_package(package_or_path), '.release'))

def get_logger_dir(dir_log_debug=None, dir_log_release=None, dir_indicator=None):
    if dir_indicator is not None:
        assert dir_log_release is not None
        if is_release(dir_indicator):
            root_dir = get_path_package(dir_indicator)
            if dir_log_release is not None:
                root_dir = os.path.join( root_dir, dir_log_release)
            return root_dir

    if tools.ispc('xiaoming'):
        root_dir = '/media/d/e/et'
    else:
        root_dir = f"{os.environ['HOME']}/xm/et"
    if dir_log_debug is not None:
        root_dir = os.path.join(root_dir, dir_log_debug)
    return root_dir


# TODO: when the dir is running by other thread, we should also exited.
def prepare_dirs(args, key_first=None, keys_exclude=[], dirs_type=['log'], root_dir=''):
    '''
    Please add the following keys to the argument:
        parser.add_argument('--log_dir_mode', default='finish_then_exit_else_overwrite', type=str)#finish_then_exit_else_overwrite
        parser.add_argument('--keys_group', default=['clipped_type'], type=ast.literal_eval)
        parser.add_argument('--name_group', default='', type=str)
        parser.add_argument('--name_run', default="", type=str)

    Please Rememer to add when the program exit!!!!

    The final log_path is:
        root_dir/name_project/dir_type/[key_group=value,]name_group/[key_normalargs=value]name_run
    root_dir: root dir
    name_project: your project
    dir_type: e.g. log, model
    name_group: for different setting, e.g. hyperparameter or just for test

    New version: parser.add_argument('--log_dir_mode', default='', type=str) #append, overwrite, finish_then_exit_else_overwrite, exist_then_exit
    '''
    SPLIT = ','
    from dotmap import DotMap
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if isinstance(args, dict):
        args = DotMap(args)

    assert isinstance(args, DotMap)
    mode = args.log_dir_mode


    # ---------------- get name_group -------------
    name_key_group = ''
    for i,key in enumerate(args.keys_group):
        if i>0:
            name_key_group += SPLIT
        name_key_group += f'{key}={arg2str(args[key])}'

    args.name_group = name_key_group + (SPLIT if name_key_group and args.name_group else '') + args.name_group

    if not args.name_group:
        args.name_group = 'tmpGroup'
        print( f'args.name_group is empty. It is set to be {args.name_group}' )

    # -------------- get root directory -----------
    # if tools.ispc('xiaoming'):
    #     root_dir = '/media/d/e/et'
    # else:
    #     root_dir = f"{os.environ['HOME']}/xm/et"
    #
    # root_dir = f'{root_dir}/{name_project}'
    # root_dir =

    # ----------- get sub directory -----------
    keys_exclude.extend(['log_dir_mode','name_group','keys_group', 'name_run', 'is_multiprocess'])
    keys_exclude.extend(args.keys_group)
    if key_first is not None:
        keys_exclude.append(key_first)
    keys_exclude.extend( args.keys_group )
    # --- add first key
    if key_first is None:
        key_first = list(set(args.keys()).difference( set(keys_exclude) )) [0]
    keys_exclude.append(key_first)
    key = key_first
    name_task = f'{key}={arg2str(args[key])}'

    # --- add keys common
    for key in args.keys():
        if key not in keys_exclude:
            name_task += f'{SPLIT}{key}={arg2str(args[key])}'

            # print( f'{key},{type(args_dict[key])}' )
    # name_task += ('' if name_suffix == '' else f'{split}{name_suffix}')

    key = 'name_run'
    if args.has_key(key) and not args[key] == '':
        name_task += f'{SPLIT}{key}={arg2str(args[key])}'


    # ----------------- prepare directory ----------
    def get_dir_full( d_type, suffix='', print_root=True, print_dirtype=True ):
        paths = []
        if print_root:
            paths.append( root_dir )
        if print_dirtype:
            paths.append( d_type )

        paths.extend([ args.name_group, f'{name_task}{suffix}'])
        return os.path.join( *paths )


    dirs_full = dict()
    for d_type in dirs_type:
        assert d_type
        dirs_full[d_type] = get_dir_full(d_type)
        print( tools.colorize( f'{d_type}_dir:\n{dirs_full[d_type]}' , 'green') )
        setattr( args, f'{d_type}_dir', dirs_full[d_type] )
    # exit()
    # ----- Move Dirs
    EXIST_dir = False
    for d in dirs_full.values():
        if osp.exists(d) and bool(os.listdir(d)):
            print( f'Exist directory\n{d}\n' )
            EXIST_dir = True
    if EXIST_dir:  # 如果"目标文件夹存在且不为空",则（根据要求决定）是否将其转移
        # print(
        #     f"Exsits sub directory: {name_task} in {root_dir} \nMove to discard(y or n)?",
        #     end='')
        # if force_write > 0:
        #     cmd = 'y'
        #     print(f'y (auto by force_write={force_write})')
        # elif force_write < 0:
        #     exit()
        # else:
        #     cmd = input()
        flag_move_dir = 'y'
        if flag_move_dir == 'y':
            for i in itertools.count():
                if i == 0:
                    suffix = ''
                else:
                    suffix = f'{SPLIT}{i}'
                dirs_full_discard = {}
                for d_type in dirs_type:

                    dirs_full_discard[d_type] = get_dir_full( f'{d_type}_del', suffix )
                if not np.any( [osp.exists( d ) for d in dirs_full_discard.values() ]):
                    break

            print(tools.colorize(f"Going to move \n{ get_dir_full( d_type ) }\nto \n{get_dir_full( f'{d_type}_del', suffix=suffix )}\n"+f"Confirm move(y or n)?", 'red'), end='')
            print(f'log_dir_mode={mode}. ', end='')
            if mode == 'append':
                flag_move_dir = 'n'
                print(f'(Append to old directory)')
            elif mode == 'overwrite':#直接覆盖
                flag_move_dir = 'y'
            elif mode == 'finish_then_exit_else_overwrite':
                if np.any( [ osp.exists( get_finish_file(d) )   for d in dirs_full.values() ]):
                    flag_move_dir = 'n'
                    print(f'Exited! Exist file\n{get_finish_file(d)}\nYou can try to rename value of "name_group" or that of "name_run"')
                    exit()
                else:
                    flag_move_dir = 'y'
            elif mode == 'exist_then_exit':
                flag_move_dir = 'n'
                print(f'Exited!\nYou can try to rename value of "name_group" or that of "name_run"')
                exit()
            else:#mode='ask'
                flag_move_dir = input()

            if flag_move_dir == 'y' \
                and \
                np.all(tools.check_safe_path( dirs_full[d_mid], confirm=False) for d_type in dirs_type):
                import shutil
                for d_type in dirs_type:
                    if osp.exists(dirs_full[d_type]):
                        # print( tools.colorize( f'Move:\n{dirs_full[d_type]}\n To\n {dirs_full_discard[d_type]}','red') )
                        shutil.move(dirs_full[d_type], dirs_full_discard[d_type])#TODO: test if not exist?
                print('Moved!')

        else:
            pass


    for d_type in dirs_type:
        tools.makedirs( dirs_full[d_type] )

    args_json = args.toDict()
    args_json['__timenow'] = tools.time_now_str()
    tools.save_json( os.path.join(args.log_dir, 'args.json'), args_json )
    return args


def get_finish_file(path):
    return os.path.join(f'{path}', '.finish_indicator')

def finish_dir(path):
    with open(get_finish_file(path), mode='w'):
        pass


def tes_prepare_dirs():
    args = dict(
        mode = 'exist_then_exit',#append, overwrite, finish_then_exit_else_overwrite, exist_then_exit
        a = 1,
        b=2,
        keys_group = ['a'],
        name_group = '',
    )
    prepare_dirs( args, name_project='tes_prepare_dirs' )
    exit()

# tes_prepare_dirs()

from collections import OrderedDict
import os
import sys
import shutil
import os.path as osp
import json
from enum import Enum, unique
import numpy as np
from warnings import warn

def truncate(s, width_max):
    return s[:width_max - 2] + '..' if len(s) > width_max else s


def fmt_item(x, width, width_max=None):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): s = "%g"%x
    else: s = str(x)
    if width_max is not None:
        s = truncate(s, width_max)
    if width is not None:
        width = max( width, len(s) )
    else:
        width = len(s)
    return  s+" "*max((width - len(s)), 0)


def fmt_row(values, widths=None, is_header=False, width_max=30):
    from collections.abc import Iterable
    # if not isinstance( widths, Iterable):
    #     widths = [widths] * len(row)
    if not isinstance(values, list):
        values = list(values)
    if widths is None:
        widths = [None] * len(values)
    assert isinstance(widths, Iterable )
    assert len(widths) == len(values)
    items = [ fmt_item(x, widths[ind], width_max) for ind, x in enumerate(values)]
    widths_new = list(map( len, items ))

    return items, widths_new

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class OutputFormat(object):
    def write_kvs(self, kvs):
        pass

    def write_str(self, s):
        pass


    def _write_items(self, items):
        pass

    def close(self):
        pass

class CsvOuputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file
        self.has_written_header=False

    def _write_items(self, items):
        for ind, item in enumerate(items):
            if item is None:
                item_str = ''
            else:
                item_str = str(item)
            self.file.write(item_str)
            if ind < len(items)-1:
                self.file.write('\t')
        self.file.write('\n')
        self.file.flush()

    def write_kvs(self, kvs):
        if not self.has_written_header:
            self._write_items( kvs.keys() )
            self.has_written_header = True
        self._write_items(kvs.values())

    def close(self):
        self.file.close()


class TensorflowOuputFormat(OutputFormat):
    def __init__(self, file):
        self.writer = file
        self.global_step = -1

    def write_kvs(self, kvs):
        import tensorflow as tf
        if 'global_step' in kvs:
            global_step = kvs['global_step']
        else:
            self.global_step += 1
            global_step = self.global_step

        summary = tf.Summary()
        for key, value in kvs.items():
            if key != 'global_step':
                summary.value.add( tag=key, simple_value=value )
        self.writer.add_summary( summary, global_step=global_step )
        self.writer.flush()


    def close(self):
        self.writer.close()


class LogOutputFormat(OutputFormat):
    def __init__(self, file, output_header_interval=10, width_max=20):
        self.file = file
        self.output_header_interval = output_header_interval
        self.width_max = width_max
        self.ind = 0
        self.widths = None
        # self.widths_cache = np.zeros( shape=(output_header_interval) , dtype=np.int )


        # Create strings for printing
        # key2str = OrderedDict()
        # for (key, val) in kvs.items():
        #     valstr = '%-8.3g' % (val,) if hasattr(val, '__float__') else val
        #     key2str[self._truncate(key,width_max)] = self._truncate(valstr, width_max)

    def write_str(self, s):
        self.file.write( s+'\n' )
        self.file.flush()

    def _write_items(self, items, widths=None):
        line = "  ".join(items)#column gap
        self.write_str( line )
        # --- update widths
        if widths is not None:
            if self.widths is None:
                self.widths = widths
            else:
                self.widths = np.maximum(self.widths, widths)


    def write_kvs(self, kvs):
        # write header
        if self.ind % self.output_header_interval == 0:
            items, widths = fmt_row(kvs.keys(), self.widths, is_header=True,width_max=self.width_max)
            self._write_items(items, widths  )
        # write body
        items, widths = fmt_row(kvs.values(), self.widths, width_max=self.width_max)
        self._write_items(items, widths  )

        self.ind += 1

    def close(self):
        self.file.close()

class JSONOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def write_kvs(self, kvs):
        for k, v in kvs.items():
            if hasattr(v, np.ndarray):
                v = v.tolist()
                kvs[k] = v
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

@unique
class type(Enum):
    stdout=0
    log = 1
    json = 2
    csv = 3
    tensorflow = 4


import os
def make_output_format(fmt, path='', basename='', append=False):

    mode = 'at' if append else 'wt'
    if isinstance(fmt, str):
        fmt = type[fmt]

    settings = {
        type.stdout: dict( cls=LogOutputFormat),
        type.log: dict(ext='log', cls=LogOutputFormat),
        type.json: dict(ext='json', cls=JSONOutputFormat),
        type.csv: dict(ext='csv', cls=CsvOuputFormat),
        type.tensorflow: dict(ext='tensorflow', cls=TensorflowOuputFormat),
    }
    if fmt == type.stdout:
        file = sys.stdout
    elif fmt == type.tensorflow:
        import tensorflow as tf
        file = tf.summary.FileWriter(logdir=path, filename_suffix=f'.{basename}')
    else:
        file_path = os.path.join(path, basename)
        file = open(f"{file_path}.{settings[fmt]['ext']}", mode)

    return settings[fmt]['cls']( file )



#可以考虑修改下，output_formats，width什么的用起来还是不是太方便
#widths_logger = [ max(10,len(name)) for name in headers]
class Logger(object):
    DEFAULT = None

    def __init__(self, formats=[type.stdout], path='', file_basename=None, file_append=False):
        '''
        :param formats: formats, e.g.,'stdout,log,csv,json'
        :type formats:str
        :param file_basename:
        :type file_basename:
        :param file_append:
        :type file_append:
        '''
        formats = tools.str2list(formats)
        self.kvs_cache = OrderedDict()  # values this iteration
        self.level = INFO
        if file_basename is None:
            file_basename = tools.time_now_str_filename()
        self.base_name = file_basename
        self.path = path
        tools.print_( f'log:\n{path}\n{file_basename}'  ,color='green' )
        self.output_formats = [make_output_format(f, path, file_basename, append=file_append) for f in formats]

    def log_str(self, s, _color=None):
        for fmt in self.output_formats:
            fmt.write_str(s)

    def log_keyvalues(self, **kwargs):
        # assert len(args) <= 2
        # if len(args) == 2:
        #     args = {args[0]:args[1]}
        # if len(args) == 1:
        #     assert isinstance(args, dict)
        #     self.kvs_cache.update( args )


        self.kvs_cache.update(kwargs)
        self.dump_keyvalues()

    def log_keyvalue(self, **kwargs):
        self.kvs_cache.update(kwargs)

    def dump_keyvalues(self):
        row_cache = self.kvs_cache
        if len(row_cache) == 0:
            return

        for fmt in self.output_formats:
            fmt.write_kvs(row_cache)
        row_cache.clear()

    def set_level(self, level):
        self.level = level

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

# Logger.DEFAULT = Logger(formats=[type.stdout])
# def logs_str(args):
#     return Logger.DEFAULT.log_str(args)
#
# def log_keyvalues(**kwargs):
#     return Logger.DEFAULT.log_keyvalues(**kwargs)
#
# def cache_keyvalues( **kwargs):
#     Logger.DEFAULT.log_keyvalue(**kwargs)
#
# def dump_keyvalues():
#     return Logger.DEFAULT.dump_keyvalues()


def tes_logger():

    # dir = "/tmp/a"
    l = Logger(formats='stdout,csv', path='/tmp/a', file_basename='aa')
    #l.width_log = [3,4]
    for i in range(200,30000,100):
        l.log_keyvalues(global_step=i, **{'x/x1':i*2, 'x/x2':i} )
    #l.dumpkvs(1)
    l.close()
    # exit()


# if __name__ == "__main__":
#     tes_logger()
#     exit()



import time
class LogTime():
    def __init__(self, name, path_logger):
        self.time = time.time()
        self.ind = 0
        self.dict = {}
        self.interval_showtitle = 10#np.clip( args.interval_iter_save, 10, 100  )
        self.logger = Logger(dir=path_logger, formats=[type.csv], filename=name,
                             file_append=False, width_kv=20, width_log=20)

    def __call__(self, name):
        self.dict[ name ] = time.time() - self.time
        #self.dict[ name+'_time' ] = time.strftime('%m/%d|%H:%M:%S', time.localtime())
        self.time = time.time()

    def complete(self):
        self.dict['time_end'] = time.strftime('%m/%d|%H:%M:%S', time.localtime())
        if self.ind%self.interval_showtitle==0:
            self.logger.log_row(list(self.dict.keys()))
        self.logger.log_row(list(self.dict.values()))
        self.ind += 1


def _strlist2list(keys):
    if isinstance(keys, str):
        keys = keys.strip(',')
        keys = keys.strip()
        keys = keys.split(',')
    return keys



def group_result(task_setting_all, fun_load, operation, path_root, methods_dir2setting={}):
    task_setting_default = dict(
        key_global_step='total_timesteps',
        keys_args_env='env'
    )
    for task_setting in task_setting_all:
        if 'keys_y' not in task_setting.keys():
            task_setting['keys_y'] = task_setting.dir
        for k in task_setting_default:
            if k not in task_setting:
                task_setting[k] = task_setting_default[k]
    for task_setting in task_setting_all:
        dir_ = task_setting.dir
        path = f'{path_root}/{dir_}'
        path_group = f'{path},group'
        tools.mkdir(path_group)
        keys_y = task_setting['keys_y']
        key_global_step = task_setting['key_global_step']



        keys_args_env = task_setting['keys_args_env']
        if operation == 'fig_generateresult':
            results_group = get_group_result(
                path_root = path,
                depth=2,
                keys_args_main=keys_args_env,
                keys_args_sub=-2,
                fun_load=fun_load( keys_y, key_global_step=key_global_step )
            )
            # modify env name
            envs = list( results_group.keys())
            for env in envs:
                env_new = env.replace('env=','')
                results_group[env_new] = results_group.pop(env)
            # modify method name
            if len(methods_dir2setting.keys() ) > 0:
                for env in results_group:
                    dirs_all = list(results_group[env].keys())
                    for dir_ in dirs_all:
                        results_group[env][ methods_dir2setting[dir_]['name']] = results_group[env].pop(dir_)
            tools.save_vars( f'{path_group}/results_group.pkl', results_group, verbose=1 )
        elif operation == 'check':
            results_group = tools.load_vars(f'{path_group}/results_group.pkl')
            pass
        elif operation == 'generate_methods_dir2setting':#for
            results_group = tools.load_vars(f'{path_group}/results_group.pkl')
            for env in results_group:
                for dir_ in results_group[env]:
                    methods_dir2setting[dir_] = "dict()"
        elif operation == 'tf':
            results_group = get_group_result(
                path_root = path,
                depth=2,
                keys_args_main=-2,
                keys_args_sub=keys_args_env,
                fun_load=fun_load( keys_y, key_global_step=key_global_step )
            )
            write_group_result(
                path_root=path,
                results_group=results_group,
                names_y=keys_y,
                overwrite=1
            )
        else:
            raise NotImplementedError
    if operation == 'generate_methods_dir2setting':
        methods_jsonstr = tools.json2str( methods_dir2setting, remove_quotes_key=False, remove_brace=False, remove_quotes_value=True , indent='\t' )
        methods_jsonstr = methods_jsonstr.replace('"',"'")
        print( methods_jsonstr )



def get_group_result(path_root, depth, keys_args_main, keys_args_sub, fun_load, file_args='args.json'):
    '''
    load from directories.
    The directories load:
        - contains <finish> file
        - contains args file(optional, e.g., args.json): it is used for generate name of figures.
    :param path_root:
    :type path_root:
    :param depth:
    :type depth:
    :param keys_args_main:
    :type keys_args_main:
    :param keys_args_sub:
    :type keys_args_sub:
    :param funs_load:
    :type funs_load:
    :param funs_name:
    :type funs_name:
    :param file_args:
    :type file_args:
    :return: results[keys_args_main][key and value of keys_args_sub]
    :rtype:
    '''
    from . import tools
    import pandas as pd

    if path_root[-1] == '/':
        path_root = path_root[:-1]



    # names_funs =

    paths = tools.get_dirs(path_root, depth=depth, only_last_depth=True, filter_=lambda x: all([  (s not in x) for s in ['notusing',',tmp'] ]) )
    # for p in paths:
    #     print(p)

    # group_keys = 'alg,alg_args'.split(',')
    if isinstance(keys_args_main, str):
        keys_args_main = _strlist2list(keys_args_main)

    if isinstance(keys_args_sub, str):
        keys_args_sub = _strlist2list(keys_args_sub)


    import os.path as osp
    # if osp.exists(path_root_new):
    #     tools.safe_delete(path_root_new, require_not_containsub=False)


    from dotmap import DotMap

    import numpy as np

    import re
    results_group = dict()
    if isinstance(keys_args_sub, list):
        contain_subtask = len( keys_args_sub ) > 0
    else:
        contain_subtask = keys_args_sub is not None
    from tqdm import tqdm
    process = tqdm( total=len(paths) )
    keys_args_main_ori = keys_args_main
    for p in paths:
        # if any(  [(not os.path.exists( f'{p}/{f}' )) for f in [file_args]] ):
        #     tools.warn_( f'{file_args} not exists' )
        #     continue

        path_split = p.split('/')
        if not os.path.exists( get_finish_file(p) ):
            tools.warn_(f'not finish:\n{p}')
            continue

        args = tools.load_json(f'{p}/{file_args}')
        if 'env' in args.keys():
            args['env'] = args['env'].split('-v')[0]
        if isinstance(keys_args_main, list):
            keys_args_main = keys_args_main_ori.copy()
            # TODO: may have bug
            for i_,k_ in list(enumerate(keys_args_main)):
                if k_ not in args.keys():
                    keys_args_main.remove(k_)
            name_method = tools.json2str(args, separators=(',', '='), keys_include=keys_args_main, remove_quotes_key=True, remove_brace=True)
        elif isinstance(keys_args_main, int):
            name_method = path_split[keys_args_main]
            name_method = name_method.replace('Link to ','')
            name_method = name_method.replace(',tidy.eval', '')
        term = DotMap(path_all=[], args_all=[])
        # for name in names_funs:
        #     term[f'{name}_all'] = []

        if not contain_subtask:
            if name_method not in results_group.keys():
                results_group[name_method] = term.copy()
            obj = results_group[name_method]
        else:#contains sub figure
            if isinstance(keys_args_sub, list):
                name_task = tools.json2str(args, separators=(',', '='), keys_include=keys_args_sub, remove_quotes_key=True, remove_brace=True)
            elif isinstance(keys_args_sub, int):
                name_task = path_split[keys_args_sub]
                name_task = name_task.replace('Link to ','')
                name_task = name_task.replace(',tidy.eval', '')

            if name_method not in results_group.keys():
                results_group[name_method] = dict()
            if name_task not in results_group[name_method].keys():
                results_group[name_method][name_task] = term.copy()
            obj = results_group[name_method][name_task]

        obj.path_all.append( p )
        obj.args_all.append( args )


        # global_steps, names, values = f(p, args)
        items = fun_load(p, args)
        if items is None:
            continue
        for item in items:
            if item is None:
                continue
            name, global_steps, values = item
            tools.save_s(f'{p}/len={len(global_steps)}', '')
            if f'{name}_all' not in obj.keys():
                obj[f'{name}_global_steps'] = global_steps #overwrite the old values
                obj[f'{name}_global_steps_p'] = p
                obj[f'{name}_all'] = []
            else:
                if len( global_steps ) != len( obj[f'{name}_global_steps'] ):
                    tools.warn_( f"length not equal:\n{p}\noldp:{obj[f'{name}_global_steps_p']}" )
                    continue
            obj[f'{name}_all'].append( values )


        process.update(1)
    return results_group

def plot_group_result(task_setting_all, alg_setting_all, path_root_data, path_root_save, fontsize , IS_DEBUG=False):
    import toolsm.plt
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle

    for alg in alg_setting_all:
        if not alg_setting_all[alg].has_key('window_length'):
            alg_setting_all[alg].window_length = 9

    xticks_setting_default = DotMap(div=1e6, unit=r'$\times 10^6$', n=5, round=1)

    for task_setting in task_setting_all:
        if not task_setting.has_key('dir'):
            task_setting.dir = task_setting.name
        if not task_setting.has_key('linewidth'):
            task_setting.linewidth = 1.5
        for env_name in task_setting.env:
            env_setting = task_setting.env[env_name]
            if not env_setting.has_key('xticks_setting'):
                env_setting.xticks_setting = xticks_setting_default
            else:
                xticks_setting = xticks_setting_default.copy()
                xticks_setting.update(env_setting.xticks_setting)
                env_setting.xticks_setting = xticks_setting

            if not env_setting.has_key('ci'):
                env_setting.ci = 60

            if not env_setting.has_key('legend'):
                env_setting.legend = False
    for task_setting in task_setting_all:
        task_setting.path = f"{path_root_data}/{task_setting.dir},group/results_group.pkl"
        f = open(task_setting.path, 'rb')
        results_group = pickle.load(f)
        task_setting.ylabel = task_setting['ylabel']
        task_setting.xaxis = f"{task_setting['name']}_global_steps"
        task_setting.yaxis = f"{task_setting['name']}_all"

        for env_name, env_setting in task_setting['env'].items():
            legends = []
            # fig, ax = plt.subplots()
            fig = plt.figure()
            ax = plt.axes()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 保证results_group里的方法再algs里都有
            for alg in results_group[env_name]:
                # assert alg in algs,
                if alg not in alg_setting_all:
                    tools.warn_(f"'{alg}' not in algs")

            for alg in alg_setting_all:
                if alg not in results_group[env_name]:
                    continue
                alg_setting = alg_setting_all[alg].copy()
                if 'algs' in env_setting.keys() and alg in env_setting['algs'].keys():
                    alg_setting_specify = env_setting['algs'][alg]
                    print(alg_setting_specify)
                    alg_setting.update(alg_setting_specify)
                # print(env_name,alg)
                x_axis = results_group[env_name][alg][task_setting.xaxis]
                y_axis = results_group[env_name][alg][task_setting.yaxis]
                y_axis = savgol_filter(y_axis, window_length=alg_setting.window_length, polyorder=1)
                # print(alg_setting.pltargs.toDict())
                # print(np.max(x_axis))
                sns.tsplot(y_axis,
                           x_axis,
                           # color=alg_setting.color,
                           # linestyle=alg_setting.linestyle,
                           linewidth=task_setting.linewidth,
                           legend=True,
                           ci=env_setting.ci,
                           **alg_setting.pltargs.toDict()
                           )
                legends.append(alg)  # legend of this plot

            # set grid
            ax.grid(linestyle='--', linewidth=0.3, color='black', alpha=0.15)

            if env_setting.has_key('ylim_start'):
                # ax.set_ylim(bottom=0, top=120)
                plt.ylim(bottom=env_setting.ylim_start)
            if env_setting.has_key('ylim_end'):
                # print(env_setting.ylim_end)
                ax.set_ylim(top=env_setting.ylim_end)

            # set labels and titles
            plt.ylabel(task_setting.ylabel, fontsize=fontsize, labelpad=4)
            xlabel = "Timesteps"
            xticks_setting = env_setting.xticks_setting
            if xticks_setting.has_key('unit'):
                xlabel = f'{xlabel}({xticks_setting.unit})'
            plt.xlabel(xlabel, fontsize=fontsize, labelpad=4)
            # env_name = env_name.replace('', '').replace('-v1', '').replace('-v2', '').replace('-v3', '')
            if '-v' in env_name:
                env_name = env_name.split('-v')[0]
            env_name = env_name.replace('NoFrameskip', '')

            plt.title(env_name, fontsize=fontsize + 3)

            path_save = f"{path_root_save}/{task_setting.dir}"
            tools.mkdirs(path_save)
            ylabel_ = task_setting.ylabel.replace(' ', '_')

            locs, labels = plt.xticks()
            loc_min, loc_max = locs[0], locs[-1]
            # print(loc_min, loc_max)
            locs = np.linspace(loc_min, loc_max, xticks_setting.n + 1)
            labels = [''] * locs.size
            for ind_, loc in enumerate(locs):
                labels[ind_] = round(loc / xticks_setting.div, xticks_setting.round)
                if xticks_setting.round == 0:
                    labels[ind_] = int(labels[ind_])
            plt.xticks(locs)
            ax.set_xticklabels(labels)



            # set legend
            if env_setting.legend:
                # print(env_setting.legend)
                h = plt.gca().get_lines()
                leg = plt.legend(handles=h, labels=legends, handlelength=4.0,
                                 ncol=1, **env_setting.legend.toDict())


            print(f'{path_save}/{env_name}')
            # exit()
            plt.savefig(f'{path_save}/{env_name}.pdf', bbox_inches="tight",
                        pad_inches=0.03)  # pad_inches: 留白大小
            if IS_DEBUG:
                toolsm.plt.set_position()
                plt.show()


def write_group_result(path_root, results_group, names_y, group_name=None, overwrite=False):

    path_root_new = f'{path_root},group'
    if group_name:
        path_root_new += f',{group_name}'

    if isinstance(names_y, str):
        names_y = _strlist2list(names_y)
    tools.mkdir(path_root_new)
    contain_subtask = not ('path_all' in list(results_group.values())[0].keys())

    del_first_time = True
    for ind_group,name_main in enumerate(results_group.keys()):
        path_log = f'{path_root_new}/{name_main}'
        if osp.exists(path_log):
            if overwrite:
                if tools.safe_delete(path_log, confirm=del_first_time):
                    del_first_time= False
                else:
                    overwrite = False # not ask again next time
                    continue
            else:
                continue

        logger = Logger('tensorflow,csv', path=path_log, file_basename='group')
        logger_log = Logger('log', path=path_log, file_basename='group')

        def log_result(_obj, name_sub=''):

            logger_log.log_str(f"name_main:{name_main},name_sub:{name_sub},len:{len(_obj.path_all)},paths:\n{_obj.path_all}\n\n")
            for name_y in names_y:
                if f'{name_y}_all' not in _obj.keys():
                    continue

                values_all = _obj[f'{name_y}_all']
                global_steps = _obj[f'{name_y}_global_steps']
                values = np.mean(values_all, axis=0)
                # print(values_all)

                for ind, global_step in enumerate(global_steps):
                    keyvalues = dict(global_step=global_step)
                    keyvalues[f'{name_y}{name_sub}'] = values[ind]
                    logger.log_keyvalues(**keyvalues)


                for i in range(ind_group, ind_group+2):
                    keyvalues = dict(global_step=i)
                    keyvalues[f'count_{name_y}{name_sub}'] = len(values_all)
                    logger.log_keyvalues(**keyvalues)


        if not contain_subtask:
            log_result( results_group[name_main] )
        else:
            for _,name_sub in enumerate( results_group[name_main].keys() ):
                # print(f'{name_main},{name_sub}')
                log_result( results_group[name_main][name_sub], f'/{name_sub}' )


        logger.close()

    tools.print_(f'Written grouped result to:\n{path_root_new}', color='green')


def tes_groupresult():
    pass
    # root = '/media/d/e/et/baselines'
    root = '/media/root/q'
    group_result( f'{root}/log_tune,tidy.eval', depth=2, key_x='total_timesteps', key_y='eprewmean_eval', keys_dir='cliptype,clipargs', keys_fig='env', file_process='progress.csv', read_csv_args=dict( sep=',' ) )
    #TODO: read_csv_args=(sep='\t')


# if __name__ == '__main__':
#     tes_groupresult()
    # import argparse
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # # parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    # parser.add_argument('--env', help='environment ID', type=str, default='Swimmer-v2')
    # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    # parser.add_argument('--play', default=False, action='store_true')
    # parser.add_argument('--clipped-type', default='origin', type=str)
    #
    # parser.add_argument('--name_group', default='', type=str)
    # parser.add_argument('--force-write', default=1, type=int)
    #
    # args = parser.parse_args()
    # prepare_dirs( args, args_dict )
    # exit()



# def group_result_(path_root, depth, key_x, key_y, keys_dir, keys_fig, file_args='args.json', file_process='process.csv', read_csv_args=dict( sep=',' ), name=None, overwrite=False):
#     from . import tools
#     import pandas as pd
#
#     if path_root[-1] == '/':
#         path_root = path_root[:-1]
#
#     paths = tools.get_dirs(path_root, depth=depth, only_last_depth=True, filter_=lambda x: any([  (s not in x) for s in ['notusing'] ]) )
#     # for p in paths:
#     #     print(p)
#
#     # group_keys = 'alg,alg_args'.split(',')
#
#     keys_dir = _get_keys(keys_dir)
#     keys_fig = _get_keys(keys_fig)
#
#     path_root_new = f'{path_root},group'
#
#
#
#     import os.path as osp
#     # if osp.exists(path_root_new):
#     #     tools.safe_delete(path_root_new, require_not_containsub=False)
#
#     tools.mkdir(path_root_new)
#     from dotmap import DotMap
#
#
#
#     import numpy as np
#
#     # key_x = 'global_step'
#     # key_y = 'reward_accumulate'
#
#     import re
#     results_group = DotMap()
#     usefig = len( keys_fig ) > 0
#     for p in paths:
#         print(p)
#         if any(  [(not os.path.exists( f'{p}/{f}' )) for f in [file_args, file_process]] ):
#             tools.warn_( f'{file_args} or {file_process} not exists' )
#             continue
#
#         args = tools.load_json(f'{p}/{file_args}')
#
#         process = pd.read_csv(f'{p}/{file_process}', **read_csv_args)
#         # print(process.columns.values)
#         # if not 'global_step' in process.columns.values:
#         #     tools.safe_delete( p, confirm=False )
#         #     continue
#
#         # print('*******')
#         group_dir = tools.json2str(args, separators=(',', '='), keys_include=keys_dir, remove_quotes_key=True,
#                                remove_brace=True)
#
#
#
#         if not usefig:
#             if group_dir not in results_group.keys():
#                 results_group[group_dir] = DotMap(global_step=process.loc[:, key_x], values_all=[], path_all=[])
#             else:
#                 assert np.all(results_group[group_dir].global_step == process.loc[:, key_x])
#             obj = results_group[group_dir]
#         else:#contains sub figure
#             group_fig = tools.json2str(args, separators=(',', '='), keys_include=keys_fig, remove_quotes_key=True,
#                                        remove_brace=True)
#             if group_dir not in results_group.keys():
#                 results_group[group_dir] = dict()
#             if group_fig not in results_group[group_dir].keys():
#                 results_group[group_dir][group_fig] = DotMap(global_step=process.loc[:, key_x], values_all=[], path_all=[])
#             else:
#                 # assert np.all(results_group[group_dir][group_fig].global_step == process.loc[:, key_x])
#                 pass
#             obj = results_group[group_dir][group_fig]
#
#         obj.values_all.append(process.loc[:, key_y])
#         obj.path_all.append( p )
#
#     del_first_time = True
#     for ind_group,group_dir in enumerate(results_group.keys()):
#         path_log = f'{path_root_new}/{group_dir}'
#         if osp.exists(path_log):
#             if overwrite:
#                 if tools.safe_delete(path_log, confirm=del_first_time):
#                     del_first_time= False
#                 else:
#                     overwrite = False # not ask again next time
#                     continue
#             else:
#                 continue
#
#         logger = Logger('tensorflow,csv', path=path_log, file_basename='group')
#         logger_log = Logger('log', path=path_log, file_basename='group')
#         def log_result( _obj, name='' ):
#             paths = '\n'.join( _obj.path_all )
#             logger_log.log_str( f"name:{name},key:{key_y},len:{len(_obj.values_all)},paths:\n{paths}\n\n" )
#             values = np.mean(_obj.values_all, axis=0)
#
#             for ind, global_step in enumerate(_obj.global_step):
#                 keyvalues = dict(global_step=global_step)
#                 keyvalues[f'{key_y}{name}'] = values[ind]
#                 logger.log_keyvalues(**keyvalues)
#
#
#             for i in range(ind_group, ind_group+2):
#                 keyvalues = dict(global_step=i)
#                 keyvalues[f'count{name}'] = len(_obj.values_all)
#                 logger.log_keyvalues(**keyvalues)
#
#
#         if not usefig:
#             log_result( results_group[group_dir] )
#         else:
#             for _,group_fig in enumerate(results_group[group_dir].keys()):
#                 log_result(results_group[group_dir][group_fig], f'/{group_fig}' )
#
#
#         logger.close()
#
#     tools.print_(f'Written grouped result to:\n{path_root_new}', color='green')



def flatten(unflattened, parent_key='', separator='.'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return dict(items)

def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result