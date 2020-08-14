from warnings import warn
import numpy  as np
import math
from collections import namedtuple



from collections import namedtuple
import torch


XY  = namedtuple('XY', ('x', 'y'))

def toXY(data):
    return XY( *data )

def fn_null(data):
    return data

class __Buffer_Base():
    def __init__(self, fn_convert_data=fn_null):
        self.fn_convert_data = fn_convert_data


    def __len__(self):
        return self.length

    @property
    def length(self):
        raise NotImplementedError

    def can_sample(self, n_sample):
        return self.length >= n_sample

    def sample(self, n_sample):
        if not self.can_sample(n_sample):
            warn( f'Buffer contains {self.length}, while sample size is {n_sample} and has been adjusted to be {self.length}' )
        ind_all = np.random.choice(self.length, min( self.length, n_sample ), replace=False)
        return self.fn_convert_data( self.get_data_by_inds(ind_all) )


    def get_batch_enumerate(self, n_batch, random=False, fill_last=False):
        if random:
            ind_all = np.random.permutation( self.length ).tolist()
        else:
            ind_all = list(range( self.length ))
        n_iteration = int(math.ceil( self.length / n_batch ))


        for i in range(n_iteration-1):
            data_batch = self.get_data_by_inds(ind_all[i * n_batch: (i + 1) * n_batch])
            yield self.fn_convert_data(data_batch)


        if n_iteration*n_batch > self.length and fill_last:
            ind_all.extend( ind_all[ :n_iteration*n_batch-self.length ]  )
        data_batch = self.get_data_by_inds(ind_all[ (n_iteration-1) * n_batch: ])
        yield self.fn_convert_data(data_batch)


class Buffer(__Buffer_Base):
    '''
        Each piece of data are considered as a atom.

        Save form: [ item_0, item_1, ..., item_n ]
        Get form:
            `item`: [ item_0, item_1, ..., item_{batch_size} ]
            `bundle`:
                    Item Type:      tuple                  dict/Dotmap
                    Return Data:   (X_batch, Y_batch )    (x=X_batch,y=Y_batch)

        item is one piece of data, it can be any type.
        item = (x,y) or dict(x=x,y=y)
    '''
    def __init__(self, n, get_form='item', **kwargs ):
        self._buffer = []
        self.n = n
        self.ind = 0
        self.get_form = get_form
        super().__init__(**kwargs)


    @property
    def length(self):
        return len(self._buffer)

    def get_data_by_inds(self, ind_all):
        result = [self._buffer[ind] for ind in ind_all] #[...item_i...]
        if self.get_form == 'item':
            pass
        elif self.get_form == 'bundle':
            if self.item_type == tuple:
                result = zip(*result)  # ( (...x_i...), (...y_i...) )
                result = map( lambda axis: list(axis), result )
                result = tuple(result)
            elif self.item_type in [dict, DotMap] :
                keys = result[0].keys()
                result_new = self.item_type()
                for key in keys:
                    result_new[key] = map( lambda item: item[key], result )
                    result_new[key] = list( result_new[key] )
                result = result_new

        return result

    def push(self, item):
        if not hasattr( self, 'item_type' ):
            self.item_type = type(item)#TODO: debug
        else:
            assert self.item_type == type(item)


        if len(self._buffer) < self.n:
            self._buffer.append(item)
        else:
            self._buffer[self.ind] = item

        self.ind = (self.ind + 1) % self.n

    def merge(self, replaybuffer):
        for traj in replaybuffer.buffer:
            self.push(traj)

from dotmap import DotMap
class Bundle(__Buffer_Base):
    '''
        Item Type:      tuple               dict() (or Dotmap)
        Save form:      (X, Y)              dict(x=X, y=Y)

        The capital X means that all data are bundled into an entity.
    '''
    def __init__(self, data, **kwargs):
        '''

        :param data: e.g,(X,Y)
        :type data:
        :param kwargs:
        :type kwargs:
        '''
        self._data = data
        if isinstance(data, tuple):
            for i in range( len(data)-1 ):
                assert data[i].shape[0] == data[i+1].shape[0]
        elif isinstance(data, dict) or isinstance(data, DotMap):
            values = list(data.values())
            for i in range( len(values)-1 ):
                assert values[i].shape[0] == values[i+1].shape[0]

        super().__init__(**kwargs)


    @property
    def length(self):
        data = self._data
        if isinstance(data, tuple):
            return len(data[0])#Even for numpy , you can also use len() which return
        elif isinstance(data, dict) or isinstance(data, DotMap):
            return len( list(data.values())[0] )
        else:
            return len(data)

    def get_data_by_inds(self, ind_all):
        data = self._data
        if isinstance(data, tuple):
            result = []
            for i in range(len(data) ):
                result.append( data[i][ind_all] )
            result = tuple(result)
        elif isinstance(data, dict) or isinstance(data, DotMap):
            result = type(data)()
            for key in data.keys():
                result[key] = data[key][ind_all]
        else:
            result = data[ind_all]

        return result

    @property
    def all(self):
        return self.fn_convert_data(self._data)


def tes_bundle():
    x = np.arange( 5 ).reshape( (5,-1) )
    y = np.arange( 5 ).reshape( (5,-1) )
    bundle = Bundle( DotMap(x=x,y=y) )
    for x in bundle.get_batch_enumerate(2, random=False, fill_last=False):
        print(x)

def tes_buffer():
    # --- original
    buffer = Buffer(n=10)
    for i in range(10):
        buffer.push( i )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)

    # --- item: tuple
    buffer = Buffer( n=10 )
    for i in range(10):
        buffer.push( (i,i*10) )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)

    print('--- item: tuple, get_form: bundle')
    buffer = Buffer( n=10, get_form='bundle' )
    for i in range(10):
        buffer.push( (i,i*10) )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)


    print('--- item: dict, get_form: bundle')
    buffer = Buffer( n=10, get_form='bundle' )
    for i in range(10):
        buffer.push( DotMap(x=i,y=i*10) )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)


if __name__ == '__main__':
    # tes_bundle()
    tes_buffer()