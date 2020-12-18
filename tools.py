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

import functools

class Timer():
    def __init__(self, verbose=False, reset=True, round_=0):
        self.reset()
        self.time_history = []
        self.time = functools.partial( self.time, verbose=verbose, reset=reset )
        self.round = round_

    def reset(self):
        self._time = time.time()

    def time(self, verbose=False, msg='Time:', reset=True):
        interval = round( time.time() - self._time, self.round )
        if verbose:
            print(f'{msg} {interval} s')
        if reset:
            self.reset()
            self.time_history.append( interval )
        return interval

    @property
    def history_mean(self):
        return np.mean( self.time_history )




class Namespace(object):
    def __init__(self, kw):
        self.__dict__.update(kw)

    def todict(self):
        return self.__dict__


# TODO: 2020-11-28 11:58:08. I have changed the str.
def str_split(s):
    s_all = s
    if isinstance(s_all, str):
        s_all = s_all.split(',')
        s_all = [item.strip() for item in s_all]
    assert isinstance(s_all, list)
    return s_all

def mkdir( dir ):
    if not os.path.exists(dir):
        os.mkdir(dir)
        return True
    return False

def makedirs(dir):
    return mkdirs(dir)
#会自动创建子文件夹
def mkdirs(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except Exception as e:
            print( e )
        return True
    return False

def pcname():
    import platform
    sys = platform.system()
    if sys == 'Windows':
        return platform.uname().node
    else:
        return os.uname().nodename

def ispc(name):
    return pcname().__contains__(name)

# def get_savepathroot():
#     path_roots = {
#         'xiaoming': '/media/d/e/et',
#         'hugo': '/home/hugo/Desktop/wxm',
#     }
#     for key in path_roots:
#         if ispc(key):
#             path_root = path_roots[key]
#             break
#     else:
#         path_root = f"{os.environ['HOME']}/xm/et"  # TODO
#     return path_root

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
    black=30,
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
    if string == '':
        return string
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
import sys

def print_( string, **kwargs ):
    if len(kwargs) > 0 :
        string = colorize( string, **kwargs )
    print( string )

print_importantinfo = functools.partial( print_, color='green', bold=True, highlight=True )
warn_ = functools.partial( print_, color='magenta' )

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


def points_2_meshgrid(arr, dim):
    '''
    把一个单独的arr转换成维度为dim的meshgrid
    Example: ARGS: arr=[1,2], dim=2 . RETURN: [1,1],[1,2],[2,1],[2,2]
    Example: ARGS: arr=[1,2,3], dim=3 . RETURN: [1,1,1],[1,1,2],[1,1,3]...
    '''
    meshgrids = [arr for i in range(dim)]
    return points_each_dim_2_meshgrid(meshgrids)


def points_each_dim_2_meshgrid(arrs):
    '''
    INPUT: the points of each dim, e.g. ([1,2],[8,9])
    Example: ARGS:  .
    RETURN: [1,8],[1,9],[2,8],[2,9]
    '''
    arr_meshgrid = np.meshgrid(*arrs)
    arr_batch = [i.reshape(-1, 1) for i in arr_meshgrid]
    arr_batch = np.concatenate(arr_batch, axis=1)
    return arr_meshgrid, arr_batch



def _get_files_dirs(path_root='', path_rel='', filter_=None, depth=1, only_last_depth=False, type='file', dir_end='', sort=None, suffix=None):
    if suffix is not None:
        filter_suffix = lambda x: x.endswith(suffix)
        if filter_ is not None:
            filter_t = filter_
            filter_ = lambda x: filter_t(x) and filter_suffix(x)
        else:
            filter_ = filter_suffix
    return _get_files_dirs_entity(path_root, path_rel, filter_, depth, only_last_depth, type, dir_end, sort)

import re
def text2int(text):
    return int(text) if text.isdigit() else text

def text2texts_ints(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ text2int(c) for c in re.split(r'(\d+)', text) ]

import inspect



# def _get_files_dirs_entity(path_root='', path_rel='', filter_=None, depth=1, only_last_depth=, type='file', dir_end='', sort_reverse=None,  sort_number=False):
def _get_files_dirs_entity(path_root, path_rel, filter_, depth, only_last_depth, type, dir_end, sort):
    '''
    :param path_root:
    :type path_root:
    :param path_rel:
    :type path_rel:
    :param filter_: Judge whether to contain this file/dir by its name (not entire path).
    :type filter_: function
    :param depth:
    :type depth:
    :param only_last_depth:
    :type only_last_depth:
    :param type:
    :type type:
    :param dir_end:
    :type dir_end:
    :param sort: 'ascend', 'descend','number'
    :type sort: str
    :return:
    :rtype:
    '''
    # kwargs = dict(path_root=path_root, path_rel=path_rel, filter_=filter_, only_sub=only_sub, dir_end=dir_end, sort_reverse=sort_reverse, sort_number=sort_number, )
    kwargs = {}
    for vname in inspect.getargspec(_get_files_dirs_entity)[0]:
        kwargs.update( {f'{vname}': locals()[vname]} )
    files = []
    dirs_ = []
    path_cat = os.path.join(path_root, path_rel)
    if not os.path.exists(path_cat):
        return []
    lists = os.listdir( path_cat )
    if sort is not None:
        kwargs_sort = {}
        if 'descend' in sort:
            kwargs_sort.update(reverse=sort_reverse )
        if 'number' in sort:
            kwargs_sort.update(key=text2texts_ints)
        lists.sort(**kwargs_sort)

    if filter_:
        lists = list(filter(filter_, lists))

    for item in lists:
        item_absolute = os.path.join(path_root, path_rel, item)
        item_rel = os.path.join(path_rel, item)
        if os.path.isfile(item_absolute):
            files.append(item_rel)
        elif os.path.isdir(item_absolute):
            dirs_.append(item_rel+dir_end)


    dirs_search = copy.copy(dirs_)
    if not only_last_depth or depth ==1 :
        obj_return = files if type=='file' else dirs_
    else:
        obj_return = []
    if depth is None or depth > 1:#None means until end
        if depth is not None:
            kwargs['depth'] -= 1
        for d in dirs_search:
            kwargs_t = kwargs.copy()
            kwargs_t['path_rel'] = d
            obj_return += _get_files_dirs_entity(**kwargs_t)

    return obj_return

def get_files(path_rel='',path_root='',  filter_=None, depth=1, only_last_depth=False, sort=None, suffix=None):
    '''
    :param filter_:a function returns true or false. e.g. lamabda filename: filename.__contains__('xxx')
    '''
    kwargs = {}
    for vname in inspect.getargspec(get_files)[0]:
        kwargs.update( {f'{vname}': locals()[vname]} )
    return _get_files_dirs( **kwargs, type='file' )

#TODO contains not contains
def get_dirs(path_rel='', path_root='', filter_=None, depth=1, only_last_depth=False, dir_end='', sort=None, suffix=None):
    '''
    :param filter_:a function returns true or false. e.g. lamabda filename: filename.__contains__('xxx')
    '''
    kwargs = {}
    for vname in inspect.getargspec(get_dirs)[0]:
        kwargs.update( {f'{vname}': locals()[vname]} )
    return _get_files_dirs( **kwargs, type='dir' )
# print(
# get_files(path_root='/media/d/e/et/baselines/model/clipped_type=kl2clip,cliprange=0.2,delta_kl=None,hidden_sizes=128,num_sharing_layers=0,ac_fn=relu,reward_scale=5.0/HalfCheetah-v2,seed=559,policy_type=MlpPolicyMy,explore_timesteps=0,explore_additive_threshold=None,explore_additive_rate=0,coef_predict_task=0/advs', sort=True, sort_number=True)
# )
# exit()
# print(get_dirs(path_root='/media/d/e/et/baselines/t/a1', depth=2,only_last_depth=True))
# exit()
import pickle
from warnings import warn

def json2file_old(obj, keys_remove=[], dependencies={}, **kwargs):
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


def json2str_file(obj, remove_brace=True, keys_exclude=[], fn_key=None):
    return json2str( obj, separators=(',', '='), remove_quotes_key=True, remove_quotes_value=True, remove_brace=remove_brace, keys_exclude=keys_exclude, fn_key=fn_key )

def filter_dict(d, keys_include):
    '''
    :param d:
    :param keys_include:
        list of keys
        The key can be either `key` or the tuple (`key`, `keys_include_sub`)
    :return:
    '''
    if keys_include is None:
        return
    elif isinstance(keys_include, str):
        keys_include = [keys_include]

    d_new = dict()
    for key in keys_include:
        if isinstance(key, str):
            d_new[key] = d[key]
        elif isinstance(key, tuple):
            key, keys_include_sub = key
            filter_dict(d[key], keys_include_sub)
            d_new[key] = d[key]
    d.clear()
    d.update( d_new )

def tes_update_dict_by_key():
    d = dict(
        a=1,
        b=dict(
            b1=3,
            b2=dict(
                b22=3
            )
        ),
    )
    filter_dict(d, keys_include=[ ('b', ['b2'])])
    print(d)
    exit()

# tes_update_dict_by_key()

def json2str(obj, separators=(',', ':'), remove_quotes_key=True, remove_quotes_value=True, remove_brace=True, remove_key=False, keys_include=None, keys_exclude=None, fn_key=None, **jsondumpkwargs):
    # TODO: sub keys for keys_include
    if isinstance(obj, DotMap):
        obj = obj.toDict()
    obj = obj.copy()
    if keys_include is not None:
        filter_dict(obj, keys_include=keys_include )

    if keys_exclude is not None:
        # TODO: keys_exclude support sub key
        if isinstance( keys_exclude, str ):
            keys_exclude = [ keys_exclude ]
        for key in keys_exclude:
            del obj[key]

    if remove_key:
        args_str = ''
        for k in obj:
            if isinstance( obj[k], dict ) or isinstance(obj[k], DotMap):
                s_ = json2str( obj[k],   separators=separators, remove_quotes_key=remove_quotes_key, remove_quotes_value=remove_quotes_value, remove_brace=remove_brace, remove_key=remove_key, keys_include=keys_include, keys_exclude=keys_exclude, fn_key=fn_key, **jsondumpkwargs )
            else:
                s_ = f'{obj[k]}'
            args_str += f'{s_},'

        if len(args_str)>0:
            args_str = args_str[:-1]
    else:
        if fn_key is not None:
            keys_old = list( obj.keys() )
            for key in keys_old:
                key_new = fn_key(key)
                if key_new != key:
                    if key_new in keys_old:
                        warn( f'key {key_new} exist. (old key is {key})' )
                    obj[ key_new ] = obj.pop( key )

        for key in obj.keys():
            if isinstance( obj[key], dict ):
                obj[key] = json2str(obj[key], separators=separators, remove_quotes_key=remove_quotes_key, remove_quotes_value=remove_quotes_value, remove_brace=remove_brace, remove_key=remove_key, keys_include=None, keys_exclude=None, fn_key=fn_key, **jsondumpkwargs)
                # keys_include and keys_exclude are None as I have already handle it at the begining.

        args_str = json.dumps(obj, separators=separators, **jsondumpkwargs)
        # print(args_str)
        if remove_brace:
            args_str = args_str.strip()
            if args_str[0] == '{':
                args_str = args_str[1:]
            if args_str[-1] == '}':
                args_str = args_str[:-1]
    if remove_quotes_key:
        import re
        args_str = re.sub(r'(?<!'+separators[1]+')"(\S*?)"', r'\1', args_str)
    if remove_quotes_value:
        import re
        args_str = re.sub(r'(?<='+separators[1]+')"(\S*?)"', r'\1', args_str)
    return args_str

# print(json2str( dict(a='a=aaa, b=1'), remove_quotes_key=0, remove_quotes_value=1 ))
# print(json2str_file( dict(a='11a')))
# exit()
def load_vars(filename, catchError=False, is_enumerate=False):
    try:
        value_all = []
        with open(filename,'rb') as f:
            # if is_enumerate:
            #     while True:
            #         value = pickle.load(f)
            #         yield value
            # else:
            try:
                while True:
                    value = pickle.load( f )
                    value_all.append(value)
            except EOFError as e:
                if len(value_all) == 1:
                    return value_all[0]
                else:
                    return value_all
    except Exception as e:
        if catchError:
            warn( f'Load Error:\n{filename}' )
            return None
        raise e

def load_vars_enumerate(filename, catchError=False):

    try:
        try:
            with open(filename,'rb') as f:
                while True:
                    value = pickle.load(f)
                    yield value
        except EOFError as e:
            if len(value_all) == 1:
                return value_all[0]
            else:
                return value_all
    except Exception as e:
        if catchError:
            warn( f'Load Error:\n{filename}' )
            return None
        raise e

#
def save_vars(filename, *vs, verbose=0,  append=False):
    if verbose:
        print( f'Save vars to \n{filename}' )
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

def save_json(filename, obj, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        json.dump(obj, f,indent=4, separators=(',', ':'))

def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj

def save_s(filename, s, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
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

import functools
def time2str(t, fmt='%Y-%m-%d %H:%M:%S'):
    if t is None:
        t = time.localtime()
    return time.strftime(fmt, t )


time_now2str = functools.partial(time2str, t=None)
time_str2filename = functools.partial(time2str, fmt ='%Y_%m_%d_%H_%M_%S')
time_now_str2filename = functools.partial(time_str2filename, t=None)




import shutil
import re
def check_safe_path(path, confirm=True, depth=4, require_not_containsub=False, name='Modify'):
    '''
    If the depth of the path is [depth], and it does not contain sub directory (if require_not_containsub is True)
    '''
    # print( f'check:{path}' )
    depth_min = 4
    assert depth >= depth_min, f'depth is at least {depth_min}, please modfiy your code for calling check_safe_path()'
    assert re.match(
        ''.join([ "^/(home|media|root)(/[^/]+){",str(depth-1),",}" ])
        ,path), f'At least operate {depth}-th depth sub-directory!\n{path}'
    contents = ''
    if not os.path.isfile( path ):
        dirs = get_dirs(path, dir_end='/')
        files = get_files( path )
        # for content in dirs+files:
        #     contents += f'\n       {content}'
        if require_not_containsub:
            assert len(dirs) == 0 and len(files) == 0
        contents = f'{len(dirs)} dirs and {len(files)} files'
    if confirm:
        print(f"{name} path '{path}'! It contains {contents}\n (y or n)?", end='')
        cmd = input()
        if cmd == 'y':
            return True
        else:
            return False
    else:
        return True

def safe_move( src, dst, **kwargs ):
    if check_safe_path(src, **kwargs, name='Move'):
        shutil.move(src, dst)
        print(f"Moved:\n{src}\nto:\n{dst}")
        return True
    print(f"Cancel moving file '{src}'")
    return False

def safe_delete(path, **kwargs):
    if check_safe_path( path, **kwargs, name='Delete' ):
        print(f"Deleted:\n{path}")
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

import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


# def load_config(filename):
#     import demjson
#     with open(filename, 'r') as f:
#         args_str = f.read()
#         args = demjson.decode(args_str)
#         args = Namespace(args)
#         return args




def tes_loadconfig():
    a = load_config('')





# def update_dict( dictmain, dictnew ):
#     for key in dictnew:
#         if

from dotmap import DotMap


import functools
def update_dict(dictmain, dictnew, onlyif_notexist=False):
    '''
    Merge `dictnew` into `dictmain`.
    We will recursively update if the sub-value of dictnew is also a dict.
    '''
    if dictmain is None:
        dictmain = dictnew
    else:
        for key in dictnew:
            if key not in dictmain.keys():
                dictmain[key] = copy.copy(dictnew[key])
            else:# key exist in dictmain
                if ( isinstance(dictnew[key], dict) or isinstance(dictnew[key], DotMap) ) and ( isinstance(dictmain[key], dict) or isinstance(dictmain[key], DotMap) ):
                    dictmain[key] = update_dict( dictmain[key], dictnew[key], onlyif_notexist=onlyif_notexist )
                else:
                    if not onlyif_notexist:
                        dictmain[key] = copy.copy(dictnew[key])
    return dictmain

update_dict_onlyif_notexist = functools.partial(update_dict, onlyif_notexist=True  )

def tes_update_dict_ifnotexist():
    from dotmap  import DotMap
    dictmain= dict(a=1, b=2)
    dictnew = dict(a=2,c=3)
    dictmain = DotMap(dictmain)
    dictnew = DotMap(dictnew)
    print( update_dict_onlyif_notexist(dictmain, dictnew  ) )
    exit()


def update_dict_specifed(dictmain, dictnew, onlyif_notexist=False):
    # NOTE: it will change dictmain
    '''
        The priority:
            If onlyif_notexist=False, dictmain < dictnew
            If onlyif_notexist=True, dictmain > dictnew
        NOTE:
          It will change the original dictmain

        For example:
        dictmain:{
                    a:1,
                    b:3
                }
        dictnew: {
                    __a={
                        1:{b:2}
                    }
                }
        RETRUN:
        {
            a:=1,
            b:2
        }
    '''
    if dictnew is not None:
      for key in dictnew.keys():
          if key.startswith('__'):
              # This means that the value are customized for the specific values
              key_interest  = key[2:] #e.g., __cliptype
              if key_interest not in dictmain:
                  continue
              value_interest  = dictmain[key_interest] #Search value from dictmain. e.g., kl_klrollback_constant_withratio

              if value_interest in dictnew[ key ].keys():
                  dictmain = update_dict_specifed( dictmain, dictnew[ key ][value_interest], onlyif_notexist=onlyif_notexist)
          else:
              if key not in dictmain.keys():
                  dictmain[key] = copy.copy(dictnew[key])
              else:# key exist in dictmain
                  if (isinstance(dictnew[key], dict) or isinstance(dictnew[key], DotMap) ) \
                      and key in dictmain.keys() \
                      and ( isinstance(dictmain[key], dict) or isinstance(dictmain[key], DotMap) ):
                      dictmain[key] = update_dict_specifed( dictmain[key], dictnew[key], onlyif_notexist=onlyif_notexist )
                  else:
                      if not onlyif_notexist:
                          dictmain[key] = copy.copy( dictnew[key])
    return dictmain


update_dict_specifed_onlyif_notexist = functools.partial( update_dict_specifed,  onlyif_notexist=True )



def update_dict_self_specifed(d, onlyif_notexist=False):
    # NOTE: It will change the d
    items = list( d.items() )
    dictspecific = type(d)()
    for k, v in items:
        if k.startswith('__'):
            del d[k]
            dictspecific[k] = v
        # else:
        #     dictmain[k] = v
    return update_dict_specifed( d, dictspecific, onlyif_notexist=onlyif_notexist )



def tes_update_dict_self_specifed():
    from dotmap  import DotMap

    dictmain= dict(
            agent=dict(
                name='DQN',
                __name=dict(
                    DQN=dict(  )
                )# Does not support second-level specification. It's little complex to implement.
            ),
            optimizer = 'Adam',
            lr=0.1,
            __optimizer = dict(
                Adam= dict( lr=0.2 )
            )# only support first-level specification.
    )
    update_dict_self_specifed(dictmain)
    print(dictmain)
    exit()


    dictmain= dict(
            optimizer = 'Adam',
            lr=0.1,
            __optimizer = dict(
                Adam= dict( optimizer='Adam1' )
            )# only support first-level specification.
    )
    print( update_dict_self_specifed(dictmain) )
    exit()
# tes_update_dict_self_specifed()

update_dict_self_specifed_onlyif_notexist = functools.partial( update_dict_self_specifed,  onlyif_notexist=True )

def tes_update_dict_specifed_onlyif_notexist():
    from dotmap  import DotMap
    dictmain= dict(a=1, b=2)
    dictnew = dict(__a={1: dict(b=3,c=4) })
    dictmain = DotMap(dictmain)
    dictnew = DotMap(dictnew)
    print( update_dict_specifed_onlyif_notexist(dictmain, dictnew  ) )
    exit()

# tes_update_dict_specifed_onlyif_notexist()

# def update_dict_onlynotexist(dictmain, dictnew):
#     if dictmain is None:
#         dictmain = dictnew
#     else:
#         # dictnew  = copy.copy( dictnew )
#         # dictnew.update(dictmain)
#         # dictmain = dictnew
#         for key in dictnew.keys():
#             if key not in dictmain.keys():
#                 dictmain[key] = dictnew[key]
#             else:
#                 if (isinstance(dictmain[key], dict) or isinstance(dictmain[key], DotMap)) and (isinstance( dictnew[key], dict ) or isinstance( dictnew[key], DotMap ) ):
#                     dictmain[key] = update_dict_ifnotexist( dictmain[key], dictnew[key] )
#     return dictmain



# tes_update_dict_ifnotexist()
# exit()


if __name__ == '__main__':
    # tes_save_vars()
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

