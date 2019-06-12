#encoding:utf-8
import json
import fnmatch
import copy
import sys
import os
import numpy as np
from contextlib import contextmanager
import time

import time


class Timer():
    def __init__(self):
        self.reset()

    def reset(self):
        self._time = time.time()

    def count(self, msg='', p=True, reset=True):
        inverval = time.time() - self._time
        if p:
            print(f'msg: {inverval} s')
        if reset:
            self.reset()
        return inverval





class Namespace(object):
    def __init__(self, kw):
        self.__dict__.update(kw)

    def todict(self):
        return self.__dict__

def mkdir( dir ):
    if not os.path.exists(dir):
        os.mkdir(dir)
        return True
    return False

def makedirs(dir):
    return mkdirs(dir)

def mkdirs(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except Exception as e:
            print( e )
        return True
    return False

def pcname():
    return os.uname().nodename

def ispc(name):
    return pcname().__contains__(name)

def get_savepathroot():
    path_roots = {
        'xiaoming': '/media/d/e/et',
        'hugo': '/home/hugo/Desktop/wxm',
    }
    for key in path_roots:
        if ispc(key):
            path_root = path_roots[key]
            break
    else:
        path_root = f"{os.environ['HOME']}/xm/et"  # TODO
    return path_root

def FlagFromFile(filepath):
    filepath += '.cmd'
    try:
        with open(filepath, 'r+') as file:
            cmd = file.read()
            cmd = cmd.replace('\n', '')
            return cmd == '1'
    except:
        os.mknod(filepath)
        return False

def print_refresh(*s):
    sys.stdout.write('\r')
    result = ''
    for i in s:
        result += str(i)
    sys.stdout.write(result)
    sys.stdout.flush()


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color='red', bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


@contextmanager
def timed(msg, print_atend=True, stdout=print):
    color = 'magenta'
    if not print_atend:
        msg += '...'
        if stdout == print:
            stdout(colorize(msg, color=color), end='')
        else:
            stdout(msg)
    tstart = time.time()
    yield
    if not print_atend:#如果已经输出begin了,就不要再输出一遍了
        msg = ''
    msg += " done in %.3f seconds" % (time.time() - tstart) #TODO 转换成小时
    if stdout == print:
        stdout(colorize(msg, color=color))
    else:
        stdout( msg )


def arr2meshgrid(arr, dim):
    '''
    把一个单独的arr转换成维度为dim的meshgrid
    Example: ARGS: arr=[1,2], dim=2 . RETURN: [1,1],[1,2],[2,1],[2,2]
    Example: ARGS: arr=[1,2,3], dim=3 . RETURN: [1,1,1],[1,1,2],[1,1,3]...
    '''
    meshgrids = [arr for i in range(dim)]
    return multiarr2meshgrid(meshgrids)


def multiarr2meshgrid(arrs):
    '''
    把多个arr转换成维度为len(arrs)的meshgird
    Example: ARGS: arrs=([1,2],[8,9]) . RETURN: [1,8],[1,9],[2,8],[2,9]
    '''
    arr = np.meshgrid(*arrs)
    arr = [i.reshape(-1, 1) for i in arr]
    arr = np.concatenate(arr, axis=1)
    return arr



def _get_files_dirs(path_root='', path_rel='', filter_=None, only_sub=True, type='file', dir_end='', sort=None, suffix=None):
    if suffix is not None:
        filter_suffix = lambda x: x.endswith(suffix)
        if filter_ is not None:
            filter_t = filter_
            filter_ = lambda x: filter_t(x) and filter_suffix(x)
        else:
            filter_ = filter_suffix
    return _get_files_dirs_entity(path_root, path_rel, filter_, only_sub, type, dir_end, sort)

def _get_files_dirs_entity(path_root='', path_rel='', filter_=None, only_sub=True, type='file', dir_end='', sort=None, suffix=None):
    files = []
    dirs = []
    lists = os.listdir(os.path.join(path_root, path_rel))

    for item in lists:
        item_absolute = os.path.join(path_root, path_rel, item)
        item_rel = os.path.join(path_rel, item)
        if os.path.isfile(item_absolute):
            files.append(item_rel)
        elif os.path.isdir(item_absolute):
            dirs.append(item_rel+dir_end)
    # print(dirs)
    if filter_:
        files = list(filter(filter_, files))
        dirs_search = copy.copy(dirs)
        dirs = list(filter(filter_, dirs))
    else:
        dirs_search = copy.copy(dirs)

    if type =='file':
        #if filter_: files = list(filter( filter_, files )) #fnmatch.filter(files, filter_)
        if not only_sub:
            for dir in dirs_search:
                files += _get_files_dirs_entity(path_root, dir, filter_, only_sub, 'file')
        obj_return = files
    elif type == 'dir':
        #if filter_: dirs = list(filter( filter_, dirs ))#fnmatch.filter(dirs, filter_)
        if not only_sub:
            for dir in dirs_search:
                # print(path_root, ' ', path_rel, ' ', dir)
                dirs += _get_files_dirs_entity(path_root, dir, filter_, only_sub, 'dir', dir_end)
        obj_return = dirs
    else:
        NotImplementedError

    if sort is not None:
        obj_return.sort(reverse=sort)
    return obj_return

def get_files(path_root='', path_rel='', filter_=None, only_sub=True, sort=None, suffix=None):
    '''
    :param path_root:
    :type path_root:
    :param path_rel:
    :type path_rel:
    :param filter_:a function returns true or false. e.g. lamabda filename: filename.__contains__('xxx')
    :type filter_:
    :param only_sub:
    :type only_sub:
    :param sort:
    :type sort:
    :param suffix:
    :type suffix:
    :return:
    :rtype:
    '''
    return _get_files_dirs(path_root,path_rel,filter_,only_sub,'file', sort=sort, suffix=suffix)


def get_dirs(path_root='', path_rel='', filter_=None, only_sub=True, dir_end='', sort=None, suffix=None):
    '''

    :param path_root:
    :type path_root:
    :param path_rel:
    :type path_rel:
    :param filter_:a function returns true or false. e.g. lamabda filename: filename.__contains__('xxx')
    :type filter_:
    :param only_sub:
    :type only_sub:
    :param dir_end:
    :type dir_end:
    :param sort:
    :type sort:
    :param suffix:
    :type suffix:
    :return:
    :rtype:
    '''
    return _get_files_dirs(path_root,path_rel,filter_,only_sub,'dir', dir_end=dir_end, sort=sort, suffix=suffix)


import pickle
from warnings import warn

def json2file(obj, keys_remove=[], dependencies={}, **kwargs):
    obj = obj.copy()
    for arg_key in dependencies.keys():
        for arg_value in dependencies[arg_key].keys():
            if arg_value != obj[arg_key]:
                if dependencies[arg_key].__contains__(obj[arg_key]):
                    for key_remove in dependencies[arg_key][arg_value]:#check下当前是不是没有这个key
                        if key_remove not in dependencies[arg_key][ obj[arg_key] ]:
                            keys_remove.append(key_remove)
                else:
                    keys_remove.extend( dependencies[arg_key][arg_value] )
    for key in keys_remove:
        print(key)
        del obj[key]
    args_str = json.dumps(obj, separators=(',', '='), **kwargs)
    for s in ['"', '{', '}']:
        args_str = args_str.replace(s, '')
    return args_str

def load_vars(filename, catchError=False):
    try:
        values = []
        with open(filename,'rb') as f:
            try:
                while True:
                    values.append(pickle.load( f ))
            except EOFError as e:
                if len(values) == 1:
                    return values[0]
                else:
                    return values
    except Exception as e:
        if catchError:
            warn( f'Load Error! {filename}' )
            return None
        # raise e
#
def save_vars(filename, *vs, disp=False, append=False):
    if disp:
        print( f'Write to \n{filename}' )
    mode = 'ab' if append else 'wb'
    with open(filename,mode) as f:
        if len(vs) == 1:
            pickle.dump(vs[0], f)
        else:
            pickle.dump(vs, f)

def tes_save_vars():
    save_vars( 't/a.pkl',1 )
    # save_vars('t/a.pkl', 2, append=True)
    x = load_vars('t/a.pkl')
    pass

def save_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f,indent=4, separators=(',', ':'))

def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj

def save_s(filename, s):
    with open(filename, 'a') as f:
        f.write(s)

def load_s(filename):
    with open(filename, 'r') as f:
        return f.readlines()



def save_np(filename, array):
    import numpy as np
    np.savetxt(filename, array, delimiter=',')

def get_ip(name):
    import netifaces as ni

    ni.ifaddresses(name)
    ip = ni.ifaddresses(name)
    if ni.AF_INET in ip.keys():
        ip = ip[ni.AF_INET][0]['addr']
        return ip
    else:
        return None
import time


def __equal_print(disp, msg):
    if disp:
        print(msg)

def equal(a, b, disp=False):
    '''
    a and b are equal if and only if
    '''
    if isinstance(a, tuple):
        a = list(a)
    if isinstance(b, tuple):
        b = list(b)
    if type(a) != type(b):
        __equal_print( disp, f'different type: {type(a)} vs {type(b)}' )
        return False
    if isinstance( a, list ):
        if len(a) != len(b):
            __equal_print(disp, f'(List) different length: length {len(a)} vs. length {len(b)}')
            return False
        for ind,(x,y) in enumerate(zip(a,b)):
            if not equal(x,y):
                __equal_print(disp, f'(List): item {ind} not equal')
                return False
    else:
        res = a==b
        if isinstance(res, np.ndarray):
            res = res.all()
            if not res:
                __equal_print(disp, '(np.ndarray): not all items equal')
                return False
        return res
    return True

def time_now_str():
    return time.strftime('%m/%d|%H:%M:%S', time.localtime())

def time_now_str_filename():
    return time.strftime('%m_%d_%H_%M_%S', time.localtime())


import shutil
import re
def check_safe_path(path, confirm=True, depth=4, name='Modify'):
    # print(f"^({os.environ['HOME']}|/media)(/[^/]+){{3,}}")
    # exit()
    print( f'check:{path}' )
    assert depth >= 4
    assert re.match(
        ''.join([ "^(", os.environ['HOME'], "|/media)(/[^/]+){",str(depth-1),",}" ])
        ,path), f'At least operate {depth}-th depth sub-directory!'
    if confirm:
        contents = ''
        if not os.path.isfile( path ):
            dirs = get_dirs(path, dir_end='/')
            files = get_files( path )
            for content in dirs+files:
                contents += f'\n       {content}'
        print(f"{name} path '{path}'! It contains {contents}\n (y or n)?", end='')
        cmd = input()
        if cmd == 'y':
            return True
        else:
            return False
    else:
        return True

def safe_move( src, dst, confirm=True ):
    if check_safe_path(src, confirm, name='Move'):
        shutil.move(src, dst)
        print(f"Moved '{src}' \nto '{dst}'")
        return True
    print(f"Cancel moving file '{src}'")
    return False

def safe_delete(path, confirm=True):
    if check_safe_path( path, confirm, name='Delete' ):
        print(f"Deleted '{path}'")
        shutil.rmtree( path )
        return True
    print('Cancel rm file')
    return False



def path2gif(path, suffix, range_, filename='0000000000000.gif'):
    if suffix[0] != '.':
        suffix = '.' + suffix
    files = get_files(path, '', sort=False, suffix=suffix)
    # print(range_)
    if range_ is not None:
        files = files[range_[0]:range_[1]:range_[2]]
        # print(range_)
    print(files)
    files = [ os.path.join(path, f) for f in files ]
    file_target = os.path.join( path, filename )
    imgs2gif(files, file_target)

def imgs2gif(files, file_save, size=None  ):
    import imageio
    import scipy.misc as mics
    if size is not None:
        assert len(size) == 2
    frames = []
    for ind,file in enumerate(files):
        print_refresh(file)
        frame = imageio.imread( file )
        if size is not None:
            frame = mics.imresize(frame, size)
        frames.append( frame )
    imageio.mimsave( file_save , frames, 'GIF-FI', duration=0.1)
    print(f'\nGif create complete: {file_save}')


# path2gif( '/media/d/e/et/bandit/ppo_bandit,continuous,tmp,learningrate=0.01,update_epochs=8,continous_gap=0.8', suffix='.jpg' )
# exit()




if __name__ == '__main__':
    tes_save_vars()
    exit()
    dirs = get_dirs('/media/d/tt/b', only_sub=False)
    print(dirs)
    # print(os.environ['HOME'])
    # safe_move('/root/a/b/c/','/root/b/')
    # safe_rm('/media/d/t/tttt')
    exit()
    files = get_files('/media/d/e/python/utxm', suffix='.py', filter_=lambda x: 'e' in x )
    print(files)
    exit()
    JSONFile('a')
    print(JSONFile('a', value_update={'a':1}))
    print(JSONFile('a', keys=('b','c')))
    #print(get_files('/media/d/e/baselines/ppo1/result_Run',filter='*.meta'))


# ------------------------------ discard

# import demjson
#
# def load_args(dir_root, dir_sub='', name_trial='', file_name='arg', args_default=None, return_args_str=False):
#     '''
#
#     '''
#     if dir_sub != '':
#         if name_trial != '':
#             name_trial = '_' + name_trial
#         path_logger = os.path.join(dir_root, dir_sub + name_trial + '/')
#     else:
#         if name_trial != '':
#             path_logger = os.path.join(dir_root, dir_sub + name_trial + '/')
#         else:
#             path_logger = dir_root
#
#     if not os.path.exists(path_logger):
#         os.mkdir(path_logger)
#     file_arg = os.path.join(path_logger, file_name +'.json')
#     if not os.path.exists(file_arg):
#         if args_default is not None:
#             args = args_default
#             args = args.replace('=', ':')
#             with open(file_arg, 'w') as f:
#                 f.write(args)
#         print('Please initialize args in ' + file_arg)
#         #exit()#临时
#     with open(file_arg, 'r') as f:
#         args_str = f.read()
#         args = demjson.decode(args_str)
#         args = Namespace(args)
#     if return_args_str:
#         return args,path_logger,file_arg,args_str
#     else:
#         return args, path_logger, file_arg
#
#
# def load_config(filename):
#     with open(filename, 'r') as f:
#         args_str = f.read()
#         args = demjson.decode(args_str)
#         args = Namespace(args)
#         return args



'''
def JSONFile(filepath, *keys):
    value_update = None
    value_default = None
    filepath += '.json'
    if not os.path.exists(filepath):
        print('re')
        with open(filepath, 'w') as f:
            json.dump( {},f )
    #-- return value
    value = None
    with open(filepath, 'r') as f:
        j = json.load( f )
    if keys:
        value = []
        fill_default = False
        for key in keys:
            if key not in j.keys():#--initialize
                fill_default = True
            value.append(j[key])
        value = tuple(value)
        if len(value) == 1:
            value = value[0]

        #--- fill default
        if fill_default:
            with open(filepath, 'w') as f:
                json.dump( j, f )
    else:
        value = j
    #-- write value_update
    if value_update is not None:
        if len(keys) == 0:#whole json
            assert isinstance(value_update, dict)
            with open(filepath, 'w') as f:
                json.dump(value_update, f)
        elif len(keys)==1:#specific keys
            key = keys[0]
            if j[key] != value_update:
                j_new = copy.deepcopy(j)
                j_new[key] = value_update
                with open(filepath, 'w') as f:
                    json.dump(j_new, f)
        else:
            raise Exception('Not supported for keys update. Please use json update directly')
    return  value
'''
'''
import time
from timeit import default_timer as timer
def reset_time():
    print_time(reset=True)
__print_time_last = None
def print_time(name=None,reset=False):
    global __print_time_last
    if __print_time_last is None:
        __print_time_last = timer()
    if not reset:
        if name is not None:
            str = f'name:{name}'
        else:
            str = ''
        str += f'time:{timer() - __print_time_last:f} s'
        print(str)
    __print_time_last = timer()
'''

