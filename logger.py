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
from baselines.common.console_util import fmt_row
from enum import Enum, unique
import numpy as np

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep


def fmt_row(width, row, header=False):
    if isinstance(width,list):
        assert len(width) == len(row)
        out = " | ".join(fmt_item(x, width[ind]) for ind,x in enumerate(row))
    else:
        out = " | ".join(fmt_item(x, width) for ind, x in enumerate(row))
    if header: out = out + "\n" + "-"*len(out)
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
    def writekvs(self, kvs, width):
        """
        Write key-value pairs
        """
        pass

    def writeseq(self, args, width):
        """
        Write a sequence of other data (e.g. a logging message)
        """
        pass

    def close(self):
        return

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
        return

class HumanOutputFormat(OutputFormat):
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

    def _truncate(self, s, width_max):
        return s[:width_max-3] + '...' if len(s) > width_max else s

    def writeseq(self, args, width):
        if isinstance(args,str):
            args = [args]
        if isinstance(args,list) or isinstance(args, tuple):
            args = fmt_row( width=width, row=args)
        self.file.write(args)
        self.file.write('\n')
        self.file.flush()


class JSONOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs, width):
        for k, v in kvs.items():
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()


def make_output_format(format, ev_dir=None, prefix='', overwrite=True):
    if ev_dir is not None:
        os.makedirs(ev_dir, exist_ok=True)
    if prefix != '':
        prefix += '_'
    mode = 'wt' if overwrite else 'at'
    if format == type.stdout:
        return HumanOutputFormat(sys.stdout)
    elif format == type.log:
        log_file = open(osp.join(ev_dir, prefix+'log.txt'), mode)
        return HumanOutputFormat(log_file)
    elif format == type.json:
        json_file = open(osp.join(ev_dir, prefix+'progress.json'), mode)
        return JSONOutputFormat(json_file)
    elif format == type.csv:
        csv_file = open(osp.join(ev_dir, prefix+'log.csv'), mode)
        return CsvOuputFormat(csv_file)
    else:
        raise ValueError('Unknown format specified: %s' % (format,))


#可以考虑修改下，output_formats，width什么的用起来还是不是太方便
#widths_logger = [ max(10,len(name)) for name in headers]
class Logger(object):
    DEFAULT = None

    def __init__(self, output_formats, dir=None, name='', width_log=30, width_kv=30, overwrite=True):
        self.name2val = OrderedDict()  # values this iteration
        self.level = INFO
        self.dir = dir
        self.width_log = width_log
        self.width_kv = width_kv
        self.output_formats = [make_output_format(f, dir, name, overwrite=overwrite) for f in output_formats]

    def log(self, args, level=INFO, width=None):
        if width is None:
            width = self.width_log
        if self.level <= level:
            self._log(args, width)

    def logkv(self, key, val):
        self.name2val[key] = val

    def dumpkvs(self, width=None):
        if len(self.name2val) == 0:
            return
        if width is not None:
            self.width_kv = width
        for fmt in self.output_formats:
            fmt.writekvs(self.name2val, self.width_kv)
        self.name2val.clear()

    def _log(self, args, width):
        for fmt in self.output_formats:
            fmt.writeseq(args, width=width)

    def set_level(self, level):
        self.level = level

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

Logger.DEFAULT = Logger(output_formats=[type.stdout])
def log(args,level=INFO, width=None):
    return Logger.DEFAULT.log(args,width=width, level=level)

def setlevel(level):
    return Logger.DEFAULT.set_level(level)

def logkv(key, val):
    return Logger.DEFAULT.logkv(key, val)

def dumpkvs(width=None):
    return Logger.DEFAULT.dumpkvs(width)


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
            self.logger.log( list( self.dict.keys() ) )
        self.logger.log( list(self.dict.values() ) )
        self.ind += 1


def _demo():

    dir = "/tmp/testlogging1"
    l = Logger(dir=dir,output_formats=[type.stdout,type.csv],name='aa')
    #l.width_log = [3,4]
    l.log(['abc','cde'])
    #l.dumpkvs(1)
    exit()


if __name__ == "__main__":
    _demo()
