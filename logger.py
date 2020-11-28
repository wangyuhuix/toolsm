from . import tools
import os.path as osp
import json
import os
import itertools
import numpy as np

import argparse

import pandas as pd

from dotmap import DotMap


def arg2str(i, fn_key=lambda __x: __x):
    if isinstance(i, float):
        if float.is_integer(i):
            i = int(i)
        else:
            i = f'{i:g}'
    if isinstance(i, int):
        if i >= int(1e4):
            i = f'{i:.0e}'
    if isinstance(i, DotMap):
        i = i.toDict()
    if isinstance(i, dict):
        i = tools.json2str_file(i, remove_brace=False, fn_key=fn_key)
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


# TODO: rename to get_logger_root_dir
def get_logger_dir(dir_log_debug=None, dir_log_release=None, dir_indicator=None):
    if dir_indicator is not None:
        assert dir_log_release is not None
        if is_release(dir_indicator):
            root_dir = get_path_package(dir_indicator)
            if dir_log_release is not None:
                root_dir = os.path.join(root_dir, dir_log_release)
            return root_dir

    if tools.ispc('xiaoming'):
        root_dir = '/media/d/e/et'
    else:
        root_dir = f"{os.environ['HOME']}/xm/et"
    if dir_log_debug is not None:
        root_dir = os.path.join(root_dir, dir_log_debug)
    return root_dir


def __get_fn_trucate_s(length):
    def __trucate_s(s):
        s_split = s.split('_')

        for i in range(len(s_split)):
            s_split[i] = s_split[i][:length]

        s = '_'.join(s_split)
        # TODO: split by the upper case
        return s

    return __trucate_s

def __split_long_filename(name):
    # TODO: split into 2 or 3 dirs
    symbol_2_cnt = { '()':0, '{}':0 }
    for _i in range(len(name)-1, -1, -1):
        s = name[_i]
        if s == ',':
            if all( [symbol_2_cnt[symbol] == 0 for symbol in symbol_2_cnt  ] ) and len(name[:_i]) <= 256:
                name = f'{name[:_i]}/{name[_i:]}'
                break
        else:
            for symbol in symbol_2_cnt.keys():
                if s == symbol[1]:
                    symbol_2_cnt[symbol] += 1
                elif s == symbol[0]:
                    symbol_2_cnt[symbol] -= 1
    else:
        raise NotImplementedError('SPLIT path failed')

    return name


# TODO: when the dir is running by other thread, we should also exited.
def prepare_dirs(args, key_first=None, key_exclude_all=None, dir_type_all=None, root_dir=''):
    '''
    Please add the following keys to the argument:
        parser.add_argument('--log_dir_mode', default='finish_then_exit_else_overwrite', type=str)#finish_then_exit_else_overwrite
        parser.add_argument('--keys_group', default=['clipped_type'], type=ast.literal_eval)
        parser.add_argument('--name_group_ext', default='', type=str)

    Please rememer to add 'finish' when the program exit!!!!

    :return
    ARGS: all dict in args will be Dotmap

    The final log_path is:
        root_dir/name_project/dir_type/[key_group=value,]name_group/[key_normalargs=value]
    root_dir: root dir
    name_project: your project
    dir_type: E.G. log, model
    name_group: for different setting, E.G. hyperparameter or just for test

    New version: parser.add_argument('--log_dir_mode', default='', type=str) #append, overwrite, finish_then_exit_else_overwrite, exist_then_exit
    '''
    if key_exclude_all is None:
        key_exclude_all = []

    if dir_type_all is None:
        dir_type_all = ['log']

    SPLIT = ','
    from dotmap import DotMap
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if isinstance(args, dict):
        args = DotMap(args)

    assert isinstance(args, DotMap)
    mode = args.log_dir_mode

    # ---------------- get name_group -------------
    def get_name_group(key2str=None):
        if key2str is not None:
            key2str_entity = key2str
        else:
            key2str_entity = lambda __x: __x
        name_group = ''
        for i, key in enumerate(args.keys_group):
            if i > 0:
                name_group += SPLIT
            name_group += f'{key2str_entity(key)}={arg2str(args[key], fn_key=key2str )}'

        name_group = name_group + (SPLIT if name_group and args.name_group_ext else '') + args.name_group_ext

        if not name_group:
            name_group = 'tmpGroup'
            print(f'args.name_group is empty. It is set to be {name_group}')

        return name_group

    name_group = get_name_group()
    key_exclude_all.extend(args.keys_group)

    # TODO: specify cut length
    for length in [1]:
        if len(name_group) > 256:
            name_group = get_name_group(__get_fn_trucate_s(length))
        else:
            break

    if len(name_group) > 256:
        name_group = __split_long_filename( name_group )




    args.name_group = name_group

    # ----------- get sub directory -----------


    key_exclude_all.extend(['log_dir_mode', 'name_group_ext', 'name_group', 'keys_group',  'is_multiprocess'])

    def get_name_task(key2str=None):
        if key2str is not None:
            key2str_entity = key2str
        else:
            key2str_entity = lambda __x: __x

        # --- add first key
        if key_first is not None and key_first not in key_exclude_all:
            key_exclude_all.append(key_first)
            key = key_first
            name_task = f'{SPLIT}{key2str_entity(key)}={arg2str(args[key], fn_key=key2str )}'

            key_exclude_all.append(key_first)

        else:
            # key_first = list(set(args.keys()).difference(set(keys_exclude)))[0]
            name_task = f''

        # --- add keys common
        for key in args.keys():
            if key not in key_exclude_all:
                name_task += f'{SPLIT}{ key2str_entity(key) }={arg2str(args[key], fn_key=key2str)}'

        # key = 'name_run'
        # if args.has_key(key) and not args[key] == '':
        #     name_task += f'{SPLIT}{key2str_entity(key)}={arg2str(args[key], fn_key=key2str)}'

        name_task = name_task[1:]
        return name_task

    name_task = get_name_task()

    for length in [1]:
        if len(name_task) > 256:
            name_task = get_name_task(__get_fn_trucate_s(length))
        else:
            break
    if len(name_task) > 256:
        name_task = __split_long_filename(name_task)


    # ----------------- prepare directory ----------
    def get_dir_full(d_type, suffix='', print_root=True, print_dirtype=True):
        paths = []
        if print_root:
            paths.append(root_dir)
        if print_dirtype:
            paths.append(d_type)

        paths.extend([args.name_group, f'{name_task}{suffix}'])
        return os.path.join(*paths)

    dirs_full = dict()
    for d_type in dir_type_all:
        assert d_type
        dirs_full[d_type] = get_dir_full(d_type)
        # print(tools.colorize(, 'green'))
        tools.print_importantinfo( f'{d_type}_dir:\n{dirs_full[d_type]}' )
        setattr(args, f'{d_type}_dir', dirs_full[d_type])
    # exit()
    # ----- Move Dirs
    EXIST_dir = False
    for d in dirs_full.values():
        if osp.exists(d) and bool(os.listdir(d)):
            print(f'Exist directory\n{d}\n')
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
                for d_type in dir_type_all:
                    dirs_full_discard[d_type] = get_dir_full(f'{d_type}_del', suffix)
                if not np.any([osp.exists(d) for d in dirs_full_discard.values()]):
                    break

            print(tools.colorize(
                f"Going to move \n{ get_dir_full( d_type ) }\nto \n{get_dir_full( f'{d_type}_del', suffix=suffix )}\n" + f"Confirm move(y or n)?",
                'red'), end='')
            print(f'log_dir_mode={mode}. ', end='')
            if mode == 'append':
                flag_move_dir = 'n'
                print(f'(Append to old directory)')
            elif mode == 'overwrite':  # 直接覆盖
                flag_move_dir = 'y'
            elif mode == 'finish_then_exit_else_overwrite':
                if np.any([exist_finish_file(d) for d in dirs_full.values()]):
                    flag_move_dir = 'n'
                    print(
                        f'Exited! Exist file\n{get_finish_file(d)}\nYou can try to rename value of "name_group"')
                    exit()
                else:
                    flag_move_dir = 'y'
            elif mode == 'exist_then_exit':
                flag_move_dir = 'n'
                print(f'Exited!\nYou can try to rename value of "name_group"')
                exit()
            else:  # mode='ask'
                flag_move_dir = input()

            if flag_move_dir == 'y' \
                    and \
                    np.all(tools.check_safe_path(dirs_full[d_type], confirm=False) for d_type in dir_type_all):
                import shutil
                for d_type in dir_type_all:
                    if osp.exists(dirs_full[d_type]):
                        # print( tools.colorize( f'Move:\n{dirs_full[d_type]}\n To\n {dirs_full_discard[d_type]}','red') )
                        shutil.move(dirs_full[d_type], dirs_full_discard[d_type])  # TODO: test if not exist?
                print('Moved!')

        else:
            pass

    for d_type in dir_type_all:
        tools.makedirs(dirs_full[d_type])

    args_json = args.toDict()
    args_json['__timenow'] = tools.time_now2str()
    tools.save_json(os.path.join(args.log_dir, 'args.json'), args_json)
    return args


def exist_finish_file(path):
    return os.path.exists(get_finish_file(path)) or os.path.exists(os.path.join(f'{path}', '.finish_indicator'))


def get_finish_file(path):
    return os.path.join(f'{path}', 'finish_indicator')


def finish_dir(path):
    with open(get_finish_file(path), mode='w'):
        pass


def tes_prepare_dirs():
    args = dict(
        mode='exist_then_exit',  # append, overwrite, finish_then_exit_else_overwrite, exist_then_exit
        a=1,
        b=2,
        keys_group=['a'],
        name_group='',
    )
    prepare_dirs(args, name_project='tes_prepare_dirs')
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
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float):
        s = "%g" % x
    else:
        s = str(x)
    if width_max is not None:
        s = truncate(s, width_max)
    if width is not None:
        width = max(width, len(s))
    else:
        width = len(s)
    return s + " " * max((width - len(s)), 0)


def fmt_row(values, widths=None, is_header=False, width_max=30):
    from collections.abc import Iterable
    # if not isinstance( widths, Iterable):
    #     widths = [widths] * len(row)
    if not isinstance(values, list):
        values = list(values)
    if widths is None or len(widths) != len(values):
        widths = [None] * len(values)
    assert isinstance(widths, Iterable)

    items = [fmt_item(x, widths[ind], width_max) for ind, x in enumerate(values)]
    widths_new = list(map(len, items))

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
        self.has_written_header = False

    def _write_items(self, items):
        for ind, item in enumerate(items):
            if item is None:
                item_str = ''
            else:
                item_str = str(item)
            self.file.write(item_str)
            if ind < len(items) - 1:
                self.file.write('\t')
        self.file.write('\n')
        self.file.flush()

    def write_kvs(self, kvs):
        if not self.has_written_header:
            self._write_items(kvs.keys())
            self.has_written_header = True
        self._write_items(kvs.values())

    def close(self):
        self.file.close()


class TensorflowOuputFormat(OutputFormat):
    # NOTE: If global_step is not included, we will add the global_step automatically
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
                summary.value.add(tag=key, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
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
        self.keys = []
        self.kvs_old = None
        # self.widths_cache = np.zeros( shape=(output_header_interval) , dtype=np.int )


        # Create strings for printing
        # key2str = OrderedDict()
        # for (key, val) in kvs.items():
        #     valstr = '%-8.3g' % (val,) if hasattr(val, '__float__') else val
        #     key2str[self._truncate(key,width_max)] = self._truncate(valstr, width_max)

    def write_str(self, s):
        self.file.write(s + '\n')
        self.file.flush()

    def _write_items(self, items, widths=None):
        line = "  ".join(items)  # column gap
        self.write_str(line)
        # --- update widths
        if widths is not None:
            if self.widths is None or len(widths) != len(self.widths):
                self.widths = widths
            else:
                self.widths = np.maximum(self.widths, widths)

    def write_kvs(self, kvs):
        # The kvs may be different from the old one
        kvs_old = self.kvs_old
        if kvs_old is None:
            kvs_old = dict()
        len_old = len(kvs_old)
        kvs_old.update(kvs)
        kvs = kvs_old

        # write header
        if self.ind % self.output_header_interval == 0 or len(kvs) != len_old:
            items, widths = fmt_row(kvs.keys(), self.widths, is_header=True, width_max=self.width_max)
            self._write_items(items, widths)
        # write body
        items, widths = fmt_row(kvs.values(), self.widths, width_max=self.width_max)
        self._write_items(items, widths)

        self.ind += 1

        for key in kvs:
            kvs[key] = ''
        self.kvs_old = kvs

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
    stdout = 0
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
        type.stdout: dict(cls=LogOutputFormat),
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
        # from torch.utils.tensorboard import SummaryWriter
        # print(path)
        # file = SummaryWriter( log_dir=path )
    else:
        file_path = os.path.join(path, basename)
        file = open(f"{file_path}.{settings[fmt]['ext']}", mode)

    return settings[fmt]['cls'](file)


# 可以考虑修改下，output_formats，width什么的用起来还是不是太方便
# widths_logger = [ max(10,len(name)) for name in headers]
class Logger(object):
    DEFAULT = None

    def __init__(self, formats=[type.stdout], path='', file_basename=None, file_is_append=False, log_time_when_dump=False):
        '''
        formats = 'stdout'
        :param formats: formats, E.G.,'stdout,log,csv,json,tensorflow'
        :type formats:str
        :param file_basename:
        :type file_basename:
        :param file_is_append:
        :type file_is_append:
        '''
        formats = tools.str_split(formats)
        self.kvs_cache = OrderedDict()  # values this iteration
        self.level = INFO
        if file_basename is None:
            file_basename = tools.time_now_str2filename()
        self.base_name = file_basename
        self.path = path
        # tools.print_(f'log:\n{path}\n{file_basename}', color='green')
        self.output_formats = [make_output_format(f, path, file_basename, append=file_is_append) for f in formats]
        self.log_time_when_dump = log_time_when_dump
        if log_time_when_dump:
            self.timer = tools.Timer(verbose=False, reset=False )

    def log(self, *args, **kwargs):
        for arg in args:
            self.log_str(arg)
        self.log_keyvalues(**kwargs)

    def log_str(self, s, _color=None):
        for fmt in self.output_formats:
            fmt.write_str(s)

    def log_and_dump_keyvalues(self, **kwargs):
        self.kvs_cache.update(kwargs)
        self.dump()

    # NOTE that the previous version execute dump immediately
    def log_keyvalues(self, **kwargs):
        self.kvs_cache.update(kwargs)

    def need_dump(self):
        return len(self.kvs_cache) > 0

    def dump(self):
        kvs_cache = self.kvs_cache
        if len(kvs_cache) == 0:
            return

        if self.log_time_when_dump:
            kvs_cache['time'] = self.timer.time()

        for fmt in self.output_formats:
            fmt.write_kvs(kvs_cache)
        kvs_cache.clear()

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
    l = Logger(formats='stdout,csv,tensorflow', path='/tmp/', file_basename='aa')
    # l.width_log = [3,4]
    for i in range(200, 30000, 100):
        l.log_and_dump_keyvalues(global_step=i, **{'x/x1': i * 2, 'x/x2': i})
    # l.dumpkvs(1)

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
        self.interval_showtitle = 10  # np.clip( args.interval_iter_save, 10, 100  )
        self.logger = Logger(dir=path_logger, formats=[type.csv], filename=name,
                             file_append=False, width_kv=20, width_log=20)

    def __call__(self, name):
        self.dict[name] = time.time() - self.time
        # self.dict[ name+'_time' ] = time.strftime('%m/%d|%H:%M:%S', time.localtime())
        self.time = time.time()

    def complete(self):
        self.dict['time_end'] = time.strftime('%m/%d|%H:%M:%S', time.localtime())
        if self.ind % self.interval_showtitle == 0:
            self.logger.log_row(list(self.dict.keys()))
        self.logger.log_row(list(self.dict.values()))
        self.ind += 1


def _strlist2list(keys):
    if isinstance(keys, str):
        keys = keys.strip(',')
        keys = keys.strip()
        keys = keys.split(',')
    return keys


# NOTE: This function has been changed in 2020/07/02, please modfiy your code
# TODO: not group the result that has been handled
def group_result(
                path,
                setting,
                depth=2,
                _type='tensorflow',#TODO: type
             ):
    '''
    :param setting
        fn_loadresult
        __dirmain  = {
            '<name of main directory>': setting
        }
    :return:
    '''

    from copy import deepcopy, copy
    # NOT GENERAL! Personal habit!
    setting_global_default = DotMap(
        key_global_step_IN_result='global_step',
        key_env_IN_info='env'
    )
    if setting_global is None:
        setting_global = setting_global_default
    else:
        setting_global = tools.update_dict(setting_global_default, setting_global)

    if algdir_2_setting_global is None:
        algdir_2_setting_global = DotMap()

    def print_algdir_2_setting(result_grouped, depth_algdir, __algdir_2_setting=None):

        if __algdir_2_setting is None:
            __algdir_2_setting = dict()
            is_first = True
        else:
            is_first = False

        if depth_algdir == 0:
            for key in result_grouped:
                __algdir_2_setting[key] = "DotMap(name='',pltargs=DotMap())"
        else:
            for key in result_grouped:
                print_algdir_2_setting(result_grouped[key], depth_algdir - 1, __algdir_2_setting=__algdir_2_setting)

        if is_first:
            methods_jsonstr = tools.json2str(__algdir_2_setting, remove_quotes_key=False, remove_brace=False,
                                             remove_quotes_value=True, indent='\t')
            methods_jsonstr = methods_jsonstr.replace('"', "'")
            print('\nalgdir_2_algsetting = ', methods_jsonstr)


    setting_global = copy(setting_global)

    if 'algdir_2_setting' in task:
        algdir_2_setting = task.algdir_2_setting
        algdir_2_setting = tools.update_dict(algdir_2_setting_global, algdir_2_setting)
    else:
        algdir_2_setting = deepcopy(algdir_2_setting_global)

    task = tools.update_dict(setting_global, task)

    if isinstance(task.key_y_all, str):
        task.key_y_all = _strlist2list(task.key_y_all)

    path_group = f'{path},group'
    tools.mkdir(path_group)
    key_y_all = task.key_y_all
    key_global_step = task.key_global_step_IN_result

    key_env_IN_info = task['key_env_IN_info']
    # Note: we split the procedures of plot but merge the result for tensorflow, because we usually need to tune the plot.

    if task_type == 'Generate_Tensorflow':
        algdir_2_result = get_result_grouped(
            path_result=path,
            depth=depth,
            fn_loaddata=fn_loaddata
        )
        if len(algdir_2_setting) == 0:
            print_algdir_2_setting(result_grouped=algdir_2_result, depth_algdir=0)

        _write_result_grouped_tensorflow(
            dir_main__2__dirsub__2__result=algdir_2_result,
            path_root=path,
            name_y_all=key_y_all,
            setting=algdir_2_setting,
            overwrite=True
        )
    elif task_type == 'Generate_Result_For_Plot':
        result_grouped = get_result_grouped(
            path_result=path,
            depth=2,
            keys_main=key_env_IN_info,
            key_sub=-2,  # alg dir
            fn_loaddata=fn_get_fn_loadresult(key_y_all, key_global_step=key_global_step)
        )
        if len(algdir_2_setting) == 0:
            print_algdir_2_setting(result_grouped=result_grouped, depth_algdir=1)
        # modify env name
        # envs = list( result_grouped.keys())
        # for env in envs:
        #     env_new = env.replace('env=','')
        #     result_grouped[env_new] = result_grouped.pop(env)
        # modify method name
        # for env in result_grouped:
        #     dirs_all = list(result_grouped[env].keys())
        #     for dir_ in dirs_all:
        #         result_grouped[env][ algdir_2_setting[dir_]['name']] = result_grouped[env].pop(dir_)

        tools.save_vars(f'{path_group}/results_group.pkl', result_grouped, verbose=1)

    elif task_type == 'check':
        result_grouped = tools.load_vars(f'{path_group}/results_group.pkl')
        pass
    elif task_type == 'Generate_algdir_2_env_2_result':
        raise NotImplementedError('The logic of algdir_2_setting has been changed, please modify the code')
        result_grouped = get_result_grouped(
            path_result=path,
            depth=2,
            keys_main=-2,
            key_sub=key_env_IN_info,
            fn_loaddata=fn_get_fn_loadresult(key_y_all, key_global_step=key_global_step)
        )
        if len(algdir_2_setting) == 0:
            print_algdir_2_setting(result_grouped=result_grouped, depth_algdir=0)

        tools.save_vars(f'{path_group}/results_group.pkl', result_grouped, verbose=1)

    else:
        raise NotImplementedError


# def get_fn_key_by_path(  ):


import copy
from dotmap import DotMap
def get_result_grouped(path, depth,
                       setting
                       ):
    '''
    setting:
        groupname_main_setting:
            path_inds:

            json_file:
            json_keys:

        groupname_sub_setting:

        fn_load_data:
    :return:
        main_2_sub_2_key_2_grouped_result
        `groupname` can be either the dir name or argument values in files
    '''
    from . import tools

    if path[-1] == '/':
        path_result = path[:-1]

    setting_default = DotMap(
        groupname_main_setting=DotMap(path_inds=[-2]),
        groupname_sub_setting=DotMap(path_inds=[-1]),
    )
    tools.update_dict_onlyif_notexist( setting, setting_default )
    # NOT GENERAL, just for my personal habit...
    filter_ = lambda x: all([(s not in x) for s in [',notusing', ',tmp']])

    path_result_all = tools.get_dirs(path, depth=depth, only_last_depth=True, filter_=filter_)

    contain_subtask = True #TODO

    main_2_sub_2_key_2_grouped_result = dict()
    from tqdm import tqdm
    process = tqdm(total=len(path_result_all))
    setting_origin = setting
    # TODO: print groupname_main
    # TODO: print groupname_sub
    def get_groupname(setting__, path):
        path_split = path.split('/')
        if setting__.has_key('path_inds'):
            # Use directory name as keys_info_main
            groupname = ','.join([path_split[ind] for ind in setting__.path_inds])
            # NOT GENERAL, just for my personal habit....
            # name_main = name_main.replace('Link to ', '')
            # name_main = name_main.replace(',tidy.eval', '')
        elif setting__.has_key('json_file'):
            info = tools.load_json(f'{path}/{setting__.json_file}')
            keys = setting__.json_keys
            groupname = tools.json2str(info, separators=(',', '='),
                                       keys_include=keys,
                                       remove_quotes_key=True, remove_quotes_value=True, remove_brace=True, remove_key=True
                                       )
        else:
            raise NotImplementedError

        return groupname
    for path_result in path_result_all:
        # NOT GENERAL, just for my personal habit...
        if not exist_finish_file(path_result):
            tools.warn_(f'NOT FINISH:\n{path}')
            continue

        setting = copy.deepcopy(setting_origin)
        groupname_main = get_groupname(setting.groupname_main_setting, path_result)
        setting.groupname_main = groupname_main
        tools.update_dict_self_specifed(setting) # update specified groupname_main
        groupname_main = setting.groupname_main

        def get_obj_new():
            return DotMap(path_all=[])

        if not contain_subtask:
            if groupname_main not in main_2_sub_2_key_2_grouped_result.keys():
                main_2_sub_2_key_2_grouped_result[groupname_main] = get_obj_new()
            group = main_2_sub_2_key_2_grouped_result[groupname_main]
        else:
            if groupname_main not in main_2_sub_2_key_2_grouped_result.keys():
                main_2_sub_2_key_2_grouped_result[groupname_main] = dict()
            groupname_sub = get_groupname(setting.groupname_sub_setting, path_result)
            setting.groupname_sub = groupname_sub
            tools.update_dict_self_specifed(setting) # update specified groupname_main
            groupname_sub = setting.groupname_sub
            if groupname_sub not in main_2_sub_2_key_2_grouped_result[groupname_main].keys():
                main_2_sub_2_key_2_grouped_result[groupname_main][groupname_sub] = get_obj_new()
            group = main_2_sub_2_key_2_grouped_result[groupname_main][groupname_sub]



        group.path_all.append(path_result)
        key_2_result = setting.fn_loaddata(path=path_result, **setting.fn_loaddata_kwargs.toDict() )#TODO: key_all
        group['__key_all'] = set()
        if key_2_result is None:
            continue
        for key, result in key_2_result.items():
            if result is None:
                continue
            x, y = result.x, result.y
            assert len(x) == len(y), f'{path_result}'
            tools.save_s(f'{path}/len={len(x)}', '')
            if f'{key}' not in group.keys():
                group[key] = DotMap(
                        x_all = [],
                        y_all = [],
                        len_all = []
                    )
            else:
                pass
            group[key][f'x_all'].append(x)
            group[key][f'y_all'].append(y)
            group[key][f'len_all'].append(len(x))

        process.update(1)
    # TODO: del value_shor
    def del_value_short(obj):
        len_all = np.array(obj[f'__{key}_length_all'])
        len_max = np.max(len_all)
        ind_remain, = np.where(len_all >= len_max)
        values = obj[f'{key}_all']
        obj[f'{key}_all'] = list(map(lambda i_: values[i_], ind_remain))
        obj[f'{key}_global_steps'] = obj[f'__{key}_global_steps_all'][ind_remain[0]]
        del obj[f'__{key}_length_all']
        del obj[f'__{key}_global_steps_all']

    for name_main in main_2_sub_2_key_2_grouped_result.keys():
        group = main_2_sub_2_key_2_grouped_result[name_main]
        if not contain_subtask:
            del_value_short(group)
        else:
            for name_sub in main_2_sub_2_key_2_grouped_result[name_main].keys():
                del_value_short(group[name_sub])

    return main_2_sub_2_key_2_grouped_result


def write_result_grouped_plot(task_all, path_root_data, path_root_save=None, algdir_2_setting_global=None,
                              setting_global=None, IS_DEBUG=False):
    # TODO: env_2_setting_global
    '''
    We define the settings from the global, task and algorithm level.
    Setting Priority: algorithm(algdir_2_algsetting) > task > global
    Also, the sub-specified-setting has highest priority

    algdir_2_setting: setting for algorithm
        <algdir>: Note that only the algs specified here will be plot
            name:
            <other settings>

        E.G.
        algdir_2_algsetting = {
            'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=false}' :
                DotMap(
                    name='Target',
                    pltargs=DotMap()
                )
        }
    task:
        dir:
        key_y_all: The key to plot y
        ylabel_all: The label of y axis
        only_plot__env
        __env:
            <env>:
                <setting>
        __alg:
            <alg>:
                <setting>

        E.G.
        task_all = [
            DotMap(
                dir='QLearning_MountainCar',
                key_y_all='return_',
                ylabel_all='Reward',
                __env = {
                    'MountainCar': DotMap(legend=dict(loc='upper right', fontsize=10))
                }
            )
        ]


    setting_global: default setting for each plot
        E.G.,
        setting_global_default = DotMap(
            xlabel = 'Timesteps',
            linewidth=1.5,
            # xticks = DotMap(),#E.G.,div=1e6, unit=r'$\times 10^6$', n=5, round=1
            legend=False,
            smooth_window_length = 9,
            fontsize=10,
            file_ext = 'pdf'
        )

    :param task_all: list of tasks, which include the following keys
    :type task_all:
    :param algdir_2_setting:
    :type algdir_2_setting:
    :param path_root_data:
    :type path_root_data:
    :param path_root_save:
    :type path_root_save:
    :param fontsize:
    :type fontsize:
    :param IS_DEBUG:
    :type IS_DEBUG:
    :return:
    :rtype:
    '''
    import toolsm.plt
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    from .plt import COLORS_EASY_DISTINGUISH

    # Set default value of algdir_2_algsetting


    setting_global_default = DotMap(
        xlabel='Timesteps',
        # xticks = DotMap(),#E.G.,div=1e6, unit=r'$\times 10^6$', n=5, round=1
        pltargs=DotMap(ci=60, linewidth=1.5),
        legend=True,
        smooth_window_length=9,
        fontsize=10,
        only_plot__env=0,
        only_plot__algdir=0,
        file_ext='pdf'
    )
    if setting_global is None:
        setting_global = setting_global_default
    else:
        setting_global = tools.update_dict(setting_global_default, setting_global)

    if algdir_2_setting_global is None:
        algdir_2_setting_global = dict()

    from copy import deepcopy, copy
    # Draw result for each task, E.G., the reward or the likelihood ratio.
    for task in task_all:
        # ------- BEGIN Update setting ---------
        setting_global_new = deepcopy(setting_global)
        setting_task = copy(task)
        key_y_all = setting_task.key_y_all
        ylabel_all = setting_task.ylabel_all
        if isinstance(key_y_all, str):
            key_y_all = _strlist2list(key_y_all)
        if isinstance(ylabel_all, str):
            ylabel_all = _strlist2list(ylabel_all)

        for key in ['dir', 'key_y_all', 'ylabel_all']:
            setting_task.pop(key)
        setting_task = tools.update_dict(setting_global_new, setting_task)
        # ------------- END --------------------





        env_2_algdir_2_result = tools.load_vars(f"{path_root_data}/{task.dir},group/results_group.pkl")
        '''
        The format of env_2_algdir_2_result:
            <env>:
                <algdir>:
                    <key_y>_all
                    <key_y>_global_steps: record the global steps
                    <key_y>_global_steps_path: record the global steps from which the global steps are extracted
        '''

        # --- BEGIN update algdir_2_setting
        if '__algdir' in setting_task:
            __algdir_2_setting_specified = setting_task.__algdir = tools.update_dict(algdir_2_setting_global,
                                                                                     setting_task.__algdir)
        else:
            __algdir_2_setting_specified = setting_task.__algdir = deepcopy(algdir_2_setting_global)

        if not setting_task.only_plot__algdir:
            # use result in env_2_algdir_2_result
            algdir_all = set()
            for v in env_2_algdir_2_result.values():
                for algdir in v.keys():
                    algdir_all.add(algdir)
        else:
            algdir_all = list(setting_task.__env.keys())

        COLORS_EASY_DISTINGUISH = COLORS_EASY_DISTINGUISH.copy()
        for algdir in algdir_all:
            if algdir not in __algdir_2_setting_specified:
                __algdir_2_setting_specified[algdir] = DotMap()
            setting_specified = __algdir_2_setting_specified[algdir]

            if 'name' not in setting_specified or setting_specified['name'] == '':
                setting_specified['name'] = algdir

            if 'pltargs' not in setting_specified:
                setting_specified.pltargs = DotMap()
            if 'color' in setting_specified.pltargs:
                if setting_specified.pltargs['color'] in COLORS_EASY_DISTINGUISH:
                    COLORS_EASY_DISTINGUISH.remove(setting_specified.pltargs['color'])

        for algdir, algsetting in __algdir_2_setting_specified.items():
            if 'color' not in algsetting.pltargs:
                algsetting.pltargs['color'] = COLORS_EASY_DISTINGUISH.pop(0)

        # --- END

        for key_y, ylabel in zip(key_y_all, ylabel_all):
            setting_task.key_y = key_y
            key_x_result = f"{key_y}_global_steps"
            key_y_result = f"{key_y}_all"

            if setting_task.only_plot__env:
                env_all = task.__env.keys()
            else:
                env_all = env_2_algdir_2_result.keys()

            for env in env_all:
                legend_all = []

                fig = plt.figure()
                ax = plt.axes()

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Make sure that all algorithms in result are plot
                # for algdir in env_2_algdir_2_result[env]:
                #     if algdir not in algdir_2_setting:
                #         tools.warn_(f"'{algdir}' not in algdir_2_setting")

                setting_task.env = env

                for algdir in setting_task.__algdir.keys():
                    if algdir not in env_2_algdir_2_result[env]:
                        continue
                    setting_task.algdir = algdir
                    setting_task.alg = setting_task.__algdir[algdir]['name']
                    # ------- BEGIN Update setting ---------
                    # setting_task_tmp = deepcopy(setting_task)
                    # setting_alg = algdir_2_setting[algdir]
                    # setting = tools.update_dict(setting_task_tmp, setting_alg)
                    setting = tools.update_dict_self_specifed(setting_task)

                    # ------------- END ---------------------


                    x = env_2_algdir_2_result[env][algdir][key_x_result]
                    y = env_2_algdir_2_result[env][algdir][key_y_result]
                    # Smooth
                    y = savgol_filter(y, window_length=setting.smooth_window_length, polyorder=1)
                    sns.tsplot(y,
                               x,
                               legend=True,
                               **setting.pltargs.toDict()
                               )

                    legend_all.append(setting['name'])  # legend of this plot

                yticks_extra = []

                # NOT GENERAL.
                # -------- BEGIN Draw clippingrange=1.2 -----
                if key_y == 'maximum_ratio':
                    yticks_extra.append(1.2)
                    plt.plot(x, [1.2] * len(x), '--', color='black', linewidth=3.2)
                    legend_all.append('Upper Clipping Range')
                # ------------------- END ---------------



                # set grid
                ax.grid(linestyle='--', linewidth=0.3, color='black', alpha=0.15)
                if setting.has_key('ylim_start'):
                    # ax.set_ylim(bottom=0, top=120)
                    plt.ylim(bottom=setting.ylim_start)
                if setting.has_key('ylim_end'):
                    # print(env_setting.ylim_end)
                    plt.ylim(top=setting.ylim_end)

                # --- set ticks and labels
                xlabel = setting.xlabel
                if setting.has_key('xticks'):
                    xticks = setting.xticks
                    if xticks.has_key('unit'):
                        xlabel = f'{xlabel}({xticks.unit})'

                    locs_xticks, labels_xticks = plt.xticks()
                    loc_min, loc_max = locs_xticks[0], locs_xticks[-1]
                    locs_xticks = np.linspace(loc_min, loc_max, xticks.n + 1)
                    labels_xticks = [''] * locs_xticks.size
                    for ind_, loc in enumerate(locs_xticks):
                        labels_xticks[ind_] = loc / xticks.div
                        if xticks.has_key('round'):
                            labels_xticks[ind_] = round(labels_xticks[ind_], xticks.round)
                            if xticks.round == 0:
                                labels_xticks[ind_] = int(labels_xticks[ind_])
                    plt.xticks(locs_xticks)
                    ax.set_xticklabels(labels_xticks)

                plt.xlabel(xlabel, fontsize=setting.fontsize, labelpad=4)
                plt.ylabel(ylabel, fontsize=setting.fontsize, labelpad=4)

                # plt.yticks(list(plt.yticks()[0]) + yticks_extra)#TODO

                # --- Set Title
                # NOT GENERAL. personal habit
                if '-v' in env:
                    env = env.split('-v')[0]
                env = env.replace('NoFrameskip', '')
                plt.title(env, fontsize=setting.fontsize + 3)

                # --- Set Legend
                if setting.legend or isinstance(setting.legend, DotMap):
                    h = plt.gca().get_lines()
                    if isinstance(setting.legend, DotMap):
                        kwargs = setting.legend.toDict()
                    else:
                        kwargs = dict()
                    leg = plt.legend(handles=h, labels=legend_all, handlelength=4.0,
                                     ncol=1, **kwargs)

                # --- Save figure
                if len(key_y_all) == 1:
                    suffix = ''
                else:
                    suffix = f'{key_y}_'

                if path_root_save is not None:
                    path_save = f"{path_root_save}/{task.dir}"
                else:
                    path_save = f'{path_root_data}/{task.dir},group'
                tools.mkdirs(path_save)
                print(f'{path_save}/{suffix}{env}')
                plt.savefig(f'{path_save}/{suffix}{env}.{setting.file_ext}', bbox_inches="tight",
                            pad_inches=0.0)

                if IS_DEBUG:
                    toolsm.plt.set_position()
                    plt.show()


# NOTE: The interface of _write_result_grouped_tensorflow may be different from those of write_result_grouped_plot. However, just let it go.

def _write_result_grouped_tensorflow(*, dir_main__2__dirsub__2__result, path_root, name_y_all, overwrite=False, name_group=None,
                                     setting=None):
    path_root_new = f'{path_root},group'
    if name_group:
        path_root_new += f',{name_group}'

    tools.mkdir(path_root_new)
    contain_subtask = not ('path_all' in list(dir_main__2__dirsub__2__result.values())[0].keys())

    del_first_time = True
    for ind_group, dir_main in enumerate(dir_main__2__dirsub__2__result.keys()):
        setting__ = DotMap(dir_main=dir_main, name_main=dir_main)
        tools.update_dict_specifed(setting__, setting)

        path_log = f"{path_root_new}/{setting__[dir_main].dir_main}"

        if osp.exists(path_log):
            if overwrite:
                if tools.safe_delete(path_log, confirm=del_first_time):
                    del_first_time = False
                else:
                    overwrite = False  # not ask again next time
                    continue
            else:
                continue

        logger = Logger('tensorflow,csv', path=path_log, file_basename='group')
        logger_log = Logger('log', path=path_log, file_basename='group')

        def log_result(_obj, name_sub=''):
            logger_log.log_str(
                f"name_main:{dir_main},name_sub:{name_sub},len:{len(_obj.path_all)},paths:\n{_obj.path_all}\n\n")
            for name_y in name_y_all:
                if f'{name_y}_all' not in _obj.keys():
                    continue

                values_all = _obj[f'{name_y}_all']
                global_steps = _obj[f'{name_y}_global_steps']
                values = np.mean(values_all, axis=0)
                # print(values_all)

                for ind, global_step in enumerate(global_steps):
                    keyvalues = dict(global_step=global_step)
                    keyvalues[f'{name_y}{name_sub}'] = values[ind]
                    logger.log_and_dump_keyvalues(**keyvalues)

                for i in range(ind_group, ind_group + 2):
                    keyvalues = dict(global_step=i)
                    keyvalues[f'count_{name_y}{name_sub}'] = len(values_all)
                    logger.log_and_dump_keyvalues(**keyvalues)

        if not contain_subtask:
            log_result(dir_main__2__dirsub__2__result[dir_main])
        else:
            for _, name_sub in enumerate(dir_main__2__dirsub__2__result[dir_main].keys()):
                log_result(dir_main__2__dirsub__2__result[dir_main][name_sub], f'/{name_sub}')

        logger.close()

    tools.print_(f'Written grouped result to:\n{path_root_new}', color='green')


def get_load_debugs_fn(key_y_all, **kwargs):
    if not isinstance(key_y_all, list):
        key_y_all = [key_y_all]

    def load_debugs_entity(p, args):

        results_all = []
        filename_global_step = 'process.csv'
        read_csv_args = dict(sep='\t')  # TODO: The ',' version
        file_global_step = f'{p}/{filename_global_step}'
        # key_global_step = 'global_step'

        process = pd.read_csv(file_global_step, **read_csv_args)
        # NOT GENERAL. for history reasons.
        for key_global_step in ['total_timesteps', 'global_step']:
            if key_global_step in list(process.head()):
                break

        global_steps = process.loc[:, key_global_step]

        file_debugs = f'{p}/debugs.pkl'
        if not osp.exists(file_debugs):
            tools.warn_(f'not exist:\n{file_debugs}')
            return None
        for key_y in key_y_all:
            debugs = tools.load_vars(file_debugs)
            # global_steps = []
            values = []
            if key_y == 'fraction_bad_solution_ratio':
                # ratio*adv<adv and |ratio-1|>cliprange
                for ind, item in enumerate(debugs):
                    ratios = item['ratios']
                    advs = item['advs']
                    cliprange = args['clipargs']['cliprange']
                    satisfy = np.logical_and(ratios * advs < advs, np.abs(ratios - 1) > cliprange)
                    frac = np.mean(satisfy.astype(np.float))
                    # global_steps.append(ind)
                    values.append(frac)
            elif key_y == 'fraction_bad_solution_kl':
                # ratio*adv<adv and |ratio-1|>cliprange
                for ind, item in enumerate(debugs):
                    ratios = item['ratios']
                    kls = item['kls']
                    advs = item['advs']
                    klrange = args['clipargs']['klrange']
                    satisfy = np.logical_and(ratios * advs < advs, kls > klrange)
                    frac = np.mean(satisfy.astype(np.float))
                    # global_steps.append(ind)
                    values.append(frac)
            elif key_y == 'fraction_ratio_out_of_range':
                # |ratio-1|>cliprange
                for ind, item in enumerate(debugs):
                    ratios = item['ratios']
                    advs = item['advs']
                    cliprange = args['clipargs']['cliprange']
                    # print(p, '\n', args)
                    # satisfy = np.logical_and( np.abs(ratios-1)> cliprange, ratios*advs >= advs )
                    satisfy = np.abs(ratios - 1) > cliprange
                    frac = np.mean(satisfy.astype(np.float))
                    # global_steps.append(ind)
                    values.append(frac)
            elif key_y == 'maximum_ratio':
                for ind, item in enumerate(debugs):
                    ratios = item['ratios']
                    # global_steps.append(ind)
                    values.append(ratios.max())
            elif key_y == 'maximum_kl':
                for ind, item in enumerate(debugs):
                    kls = item['kls']
                    # global_steps.append(ind)
                    values.append(kls.max())
            else:
                for ind, item in enumerate(debugs):
                    vs = item[key_y]
                    # global_steps.append(ind)
                    values.append(vs)
            results_all.append((key_y, global_steps, values))
        return results_all

    return load_debugs_entity




def _load_csv(path, file, key_x, keys_y, kwargs_readcsv=None, *args, **kwargs):
    if kwargs_readcsv is None:
        kwargs_readcsv = dict(sep='\t')

    file = f'{path}/{file}'
    if not osp.exists(file):
        tools.warn_(f'not exist:\n{file}')
        return None
    process = pd.read_csv(file, **kwargs_readcsv)
    # NOT GENERAL. for history reasons.
    # for key_global_step in ['total_timesteps', 'global_step']:
    #     if key_global_step in list(process.head()):
    #         break
    key_2_result = dict()
    global_steps_ori = process.loc[:, key_x]

    for key in keys_y:
        v_ori = process.loc[:, key]
        # NOT GENERAL. For personal habit
        if key == 'eprewmean_eval':
            if v_ori.dtype == object:
                v_ori[v_ori == 'None'] = np.nan
                v_ori = v_ori.astype(np.float64)
            indexs = np.logical_not(np.isnan(v_ori))
            v = v_ori[indexs]
            global_steps = global_steps_ori[indexs]
        else:
            v = v_ori
            global_steps = global_steps_ori
        key_2_result[key] = DotMap( x=global_steps, y=v )

    return key_2_result


# if __name__ == '__main__':
#     load_csv = get_load_csv_fn(['eprewmean_eval'])
#     load_csv(
#         '/media/d/e/et/baselines/log/cliptype=kl2clip,clipargs={klrange=null,adjusttype=base_clip_upper,cliprange=0.2,kl2clip_opttype=tabular},tidy.eval/env=Hopper-v2,seed=3,lam=0.95,policy_type=MlpPolicyExt,hidden_sizes=64,num_layers=2,num_sharing_layers=0,ac_fn=tanh,lam_decay=False')


def tes_groupresult():
    pass
    # root = '/media/d/e/et/baselines'
    root = '/media/root/q'
    group_result(f'{root}/log_tune,tidy.eval', depth=2, key_x='total_timesteps', key_y='eprewmean_eval',
                 keys_dir='cliptype,clipargs', keys_fig='env', file_process='progress.csv', read_csv_args=dict(sep=','))
    # TODO: read_csv_args=(sep='\t')


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