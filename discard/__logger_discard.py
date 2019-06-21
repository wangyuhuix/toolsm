"""

See README.md for a description of the logging API.

OFF state corresponds to having Logger.CURRENT == Logger.DEFAULT
ON state is otherwise

"""

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

@unique
class type(Enum):
    stdout=0
    log = 1
    json = 2
    csv = 3

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
        self.write_header=False

    def write_line(self, line):
        pass

    def write_items(self,items):
        for item in items:
            self.file.write(str(item))
            self.file.write('\t')
        self.file.write('\n')
        self.file.flush()

    def write_kvs(self, kvs):
        if not self.write_header:
            self.write_items( kvs.keys() )
            self.write_header = True
        self.write_items(kvs.values())

    def close(self):
        self.file.close()

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

def make_output_format(fmt, ev_dir=None, filename='tmp', append=False):
    if ev_dir is not None:
        os.makedirs(ev_dir, exist_ok=True)

    mode = 'at' if append else 'wt'
    if isinstance(fmt, str):
        fmt = type[fmt]
    settings = {
        type.stdout: dict( cls=LogOutputFormat),
        type.log: dict(ext='log', cls=LogOutputFormat),
        type.json: dict(ext='json', cls=JSONOutputFormat),
        type.csv: dict(ext='csv', cls=CsvOuputFormat),
    }
    if fmt == type.stdout:
        file = sys.stdout
    else:
        file = open(osp.join(ev_dir, f"{filename}.{settings[fmt]['ext']}"), mode)

    return settings[fmt]['cls']( file )



#可以考虑修改下，output_formats，width什么的用起来还是不是太方便
#widths_logger = [ max(10,len(name)) for name in headers]
class Logger(object):
    DEFAULT = None

    def __init__(self, output_formats, dir=None, filename='', append=False):
        self.row_cache = OrderedDict()  # values this iteration
        self.level = INFO
        self.dir = dir
        self.output_formats = [make_output_format(f, dir, filename, append=append) for f in output_formats]

    def log_str(self, s, _color=None):
        for fmt in self.output_formats:
            fmt.write_line(s)

    def log_row(self, _color=None, **kwargs):
        self.row_cache.update(kwargs)
        self.dump_cols( _color )

    def cache_cols(self, **kwargs):
        self.row_cache.update(kwargs)

    def dump_cols(self, _color=None):
        row_cache = self.row_cache
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

Logger.DEFAULT = Logger(output_formats=[type.stdout])
def log(args,level=INFO, width=None):
    return Logger.DEFAULT.log_row(args, width=width, level=level)

def setlevel(level):
    return Logger.DEFAULT.set_level(level)

def logkv(key, val):
    return Logger.DEFAULT.log_row(key, val)

def dumpkvs(width=None):
    return Logger.DEFAULT.dump_cols(width)


def _demo():

    dir = "/tmp/testlogging1"
    l = Logger(dir=dir, output_formats=['stdout', 'csv', 'log'], filename='aa')
    #l.width_log = [3,4]
    for i in range(11):
        l.log_row(a=10**i,b=2)
    #l.dumpkvs(1)
    exit()


if __name__ == "__main__":
    _demo()



import time
class LogTime():
    def __init__(self, name, path_logger):
        self.time = time.time()
        self.ind = 0
        self.dict = {}
        self.interval_showtitle = 10#np.clip( args.interval_iter_save, 10, 100  )
        self.logger = Logger(dir=path_logger, output_formats=[type.csv], filename=name,
                             append=False, width_kv=20, width_log=20)

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

