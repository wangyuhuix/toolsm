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


def truncate(s, width_max):
    return s[:width_max - 2] + '..' if len(s) > width_max else s


def fmt_item(x, width):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): s = "%g"%x
    else: s = str(x)
    s = truncate(s, width)
    return  s+" "*max((width - len(s)), 0)


def fmt_row(widths, values, is_header=False):
    from collections.abc import Iterable
    # if not isinstance( widths, Iterable):
    #     widths = [widths] * len(row)
    assert isinstance(widths, Iterable )
    assert len(widths) == len(values)
    out = " | ".join(fmt_item(x, widths[ind]) for ind,x in enumerate(values))
    if is_header: out = out + "\n" + "-" * len(out)
    return out

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
    def writekvs(self, kvs):
        """
        Write key-value pairs
        """
        pass

    def write_line(self, line):
        pass

    def close(self):
        pass

class CsvOuputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writeseq(self, args, width):
        if isinstance(args,str):
            args = [args]
        for item in args:
            self.file.write(str(item))
            self.file.write('\t')
        self.file.write('\n')
        self.file.flush()


    def close(self):
        self.file.close()

class LogOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs, width_max):
        # Create strings for printing
        key2str = OrderedDict()
        for (key, val) in kvs.items():
            valstr = '%-8.3g' % (val,) if hasattr(val, '__float__') else val
            key2str[self._truncate(key,width_max)] = self._truncate(valstr, width_max)

        # Find max widths
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in key2str.items():
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')
        # Flush the output to the file
        self.file.flush()


    def close(self):
        self.file.close()

class JSONOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs, width):
        for k, v in kvs.items():
            if hasattr(v, np.ndarray):
                v = v.tolist()
                kvs[k] = v
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

def make_output_format(fmt, ev_dir=None, filename='tmp', append=True):
    if ev_dir is not None:
        os.makedirs(ev_dir, exist_ok=True)

    mode = 'wt' if append else 'at'
    if isinstance(fmt, str):
        fmt = type[fmt]
    settings = {
        type.stdout: dict(ext='stdout', cls=LogOutputFormat),
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

    def __init__(self, output_formats, dir=None, name='', overwrite=True):
        self.row_cache = OrderedDict()  # values this iteration
        self.level = INFO
        self.dir = dir
        self.output_formats = [make_output_format(f, dir, name, append=overwrite) for f in output_formats]

    def log_line(self, s):
        for fmt in self.output_formats:
            fmt.write_line(s)

    def log_row(self, **kwargs):
        self.dump_cols( kwargs )

    def log_col(self, key, val):
        self.row_cache[key] = val
        self.dump_cols()

    def dump_cols(self, row_cache=None):
        if row_cache is None:
            row_cache = self.row_cache
        if len(row_cache) == 0:
            return

        for fmt in self.output_formats:
            fmt.writekvs(row_cache)
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
    return Logger.DEFAULT.log_col(key, val)

def dumpkvs(width=None):
    return Logger.DEFAULT.dump_cols(width)


import time
class LogTime():
    def __init__(self, name, path_logger):
        self.time = time.time()
        self.ind = 0
        self.dict = {}
        self.interval_showtitle = 10#np.clip( args.interval_iter_save, 10, 100  )
        self.logger = Logger(dir=path_logger, output_formats=[type.csv], name=name,
                             overwrite=False, width_kv=20, width_log=20)

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


def _demo():

    dir = "/tmp/testlogging1"
    l = Logger(dir=dir,output_formats=[type.stdout,type.csv],name='aa')
    #l.width_log = [3,4]
    l.log_row(['abc', 'cde'])
    #l.dumpkvs(1)
    exit()


if __name__ == "__main__":
    _demo()
