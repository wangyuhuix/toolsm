import tools
import os.path as osp
import json
import os
import itertools
import numpy as np

import argparse

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




def prepare_dirs(args, key_first=None, keys_exclude=[], dirs_type=['log'], name_project='tmpProject'):
    '''
    Please add the following keys to the argument:
        parser.add_argument('--mode', default='', type=str)
        parser.add_argument('--keys_group', default=['clipped_type'], type=ast.literal_eval)
        parser.add_argument('--name_group', default='', type=str)
        parser.add_argument('--name_run', default="", type=str)

    The final log_path is:
        root_dir/name_project/dir_type/[key_group=value,]name_group/[key_normalargs=value]name_run
    root_dir: root dir
    name_project: your project
    dir_type: e.g. log, model
    name_group: for different setting, e.g. hyperparameter or just for test

    New version: parser.add_argument('--mode', default='', type=str) #append, overwrite, finish_then_exit_else_overwrite, exist_then_exit
    '''
    SPLIT = ','
    from dotmap import DotMap
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if isinstance(args, dict):
        args = DotMap(args)

    assert isinstance(args, DotMap)
    mode = args.mode


    # ---------------- get name_group -------------
    name_key_group = ''
    for i,key in enumerate(args.keys_group):
        if i>0:
            name_key_group += SPLIT
        name_key_group += f'{key}={arg2str(args[key])}'

    args.name_group = name_key_group + (SPLIT if name_key_group and args.name_group else '') + args.name_group

    if not args.name_group:
        args.name_group = 'tmpGroup'
        print( f'args.name_group is empty. it is set to be {args.name_task}' )

    # -------------- get root directory -----------
    if tools.ispc('xiaoming'):
        root_dir = '/media/d/e/et'
    else:
        root_dir = f"{os.environ['HOME']}/xm/et"

    root_dir = f'{root_dir}/{name_project}'

    # ----------- get sub directory -----------
    keys_exclude.extend(['mode','name_group','keys_group', 'name_run'])
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
            print(f'mode={mode}. ', end='')
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

    tools.save_json( os.path.join(args.log_dir, 'args.json'), args.toDict() )
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

    def write_line(self, line):
        pass

    def close(self):
        pass

class CsvOuputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file
        self.has_written_header=False

    def write_line(self, line):
        pass

    def write_items(self,items):

        for ind, item in enumerate(items):
            self.file.write(str(item))
            if ind < len(items)-1:
                self.file.write('\t')
        self.file.write('\n')
        self.file.flush()

    def write_kvs(self, kvs):
        if not self.has_written_header:
            self.write_items( kvs.keys() )
            self.has_written_header = True
        self.write_items(kvs.values())

    def close(self):
        self.file.close()


class TensorflowOuputFormat(OutputFormat):
    def __init__(self, file):
        self.writer = file
        self.global_step = -1

    def write_line(self, line):
        pass

    def write_items(self,items):
        pass

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

    def write_line(self, line):
        self.file.write( line+'\n' )
        self.file.flush()

    def write_items(self, items, widths=None):
        line = "   ".join(items)
        self.write_line( line )
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
            self.write_items(items, widths  )
        # write body
        items, widths = fmt_row(kvs.values(), self.widths, width_max=self.width_max)
        self.write_items(items, widths  )

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

    def __init__(self, formats, path='', file_basename='', file_append=False):
        '''
        :param formats: formats, e.g.,'log,csv,json'
        :type formats:str
        :param file_basename:
        :type file_basename:
        :param file_append:
        :type file_append:
        '''
        if isinstance(formats, str):
            if ',' in formats:
                formats = formats.split(',')
            else:
                formats = [formats]
        self.kvs_cache = OrderedDict()  # values this iteration
        self.level = INFO
        self.base_name = file_basename
        self.path = path
        self.output_formats = [make_output_format(f, path, file_basename, append=file_append) for f in formats]

    def log_str(self, s, _color=None):
        for fmt in self.output_formats:
            fmt.write_line(s)

    def log_keyvalues(self,  **kwargs):
        self.kvs_cache.update(kwargs)
        self.dump_keyvalues(  )

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

Logger.DEFAULT = Logger(formats=[type.stdout])
def logs_str(args):
    return Logger.DEFAULT.log_str(args)

def log_keyvalues(**kwargs):
    return Logger.DEFAULT.log_keyvalues(**kwargs)

def cache_keyvalues( **kwargs):
    Logger.DEFAULT.log_keyvalue(**kwargs)

def dump_keyvalues():
    return Logger.DEFAULT.dump_keyvalues()


def _demo():

    # dir = "/tmp/a"
    l = Logger( formats='stdout,tensorflow,csv', path='/tmp/a', file_basename='aa')
    #l.width_log = [3,4]
    for i in range(9):
        l.log_keyvalues(global_step=i, x=i )
    #l.dumpkvs(1)
    l.close()
    exit()


if __name__ == "__main__":
    _demo()
    exit()



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





if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env', help='environment ID', type=str, default='Swimmer-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--clipped-type', default='origin', type=str)

    parser.add_argument('--name_group', default='', type=str)
    parser.add_argument('--force-write', default=1, type=int)

    args = parser.parse_args()
    prepare_dirs( args, args_dict )
    exit()

