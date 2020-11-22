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

class Buffer_Base():
    def __init__(self, fn_convert_push=fn_null, fn_convert_get=fn_null):
        self.fn_convert_push = fn_convert_push
        self.fn_convert_get = fn_convert_get


    def __len__(self):
        return self.length


    def empty(self):
        return len(self) == 0

    @property
    def length(self):
        raise NotImplementedError

    def can_sample(self, n_sample):
        if n_sample is None:
            n_sample = 1
        return self.length >= n_sample

    def sample(self, n_sample):
        if not Buffer_Base.can_sample(self, n_sample):
            warn( f'Buffer contains {self.length}, while sample size is {n_sample} and has been adjusted to be {self.length}' )
        ind_all = np.random.choice(self.length, min( self.length, n_sample ), replace=False)
        return self.fn_convert_get(self.get_data_by_inds(ind_all))


    def get_batch_enumerate(self, n_batch, random=False, fill_last=False):
        if n_batch is None:
            n_batch = self.length
            random = False # It is not necessary to randomize when batch size=buffer size

        if random:
            ind_all = np.random.permutation( self.length ).tolist()
        else:
            ind_all = list(range( self.length ))


        n_iteration = int(math.ceil( self.length / n_batch ))


        for i in range(n_iteration-1):
            data_batch = self.get_data_by_inds(ind_all[i * n_batch: (i + 1) * n_batch])
            yield self.fn_convert_get(data_batch)


        if n_iteration*n_batch > self.length and fill_last:
            ind_all.extend( ind_all[ :n_iteration*n_batch-self.length ]  )
        data_batch = self.get_data_by_inds(ind_all[ (n_iteration-1) * n_batch: ])
        yield self.fn_convert_get(data_batch)



def _get_fn_stack(item):
    if isinstance(item, torch.Tensor):
        fn_stack = lambda r: torch.stack(r)
    elif isinstance(item, np.ndarray):
        fn_stack = lambda r: np.stack(r)
    else:
        fn_stack = lambda r: list(r)
    return fn_stack


# TODO: merge Buffer and Bundle as 'save_form' and 'get_form'
class Buffer(Buffer_Base):
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
    def __init__(self, n=None, get_form='item', temperature_weight=None, **kwargs):
        self._buffer = []
        self._weight = []
        self._has_fetched = []
        self.n = n
        if n is None:
            warn('Buffer size not limited')
        self.ind = 0
        self.get_form = get_form
        self.weight_is_None = None
        self.temperature_weight = temperature_weight
        super().__init__(**kwargs)

    @property
    def has_fetched(self):
        return self._has_fetched.copy()

    @property
    def length(self):
        return len(self._buffer)

    def get_data_by_inds(self, ind_all):
        result = [self.get_from_buffer(ind) for ind in ind_all] #[...item_i...]
        if self.get_form == 'item':
            pass
        elif self.get_form == 'bundle':
            if self.item_type == tuple:
                fn_stack = _get_fn_stack( result[0][0] )
                result = zip(*result)  # ( (...x_i...), (...y_i...) )
                result = map( lambda r: fn_stack(r), result )
                result = tuple(result) # ( X_stack, Y_stack )
            elif self.item_type in [dict, DotMap] :
                fn_stack = _get_fn_stack(  next(iter(result[0].values()))  )
                keys = result[0].keys()
                result_new = self.item_type()
                for key in keys:
                    result_new[key] = list( map( lambda item: item[key], result ) ) # dict( x=(...x_i...), y=(...y_i...) )
                    result_new[key] = fn_stack( result_new[key] ) # dict( x=X_stack,y=Y_stack )
                result = result_new
        return result

    def enumerate(self):
        for ind in range(self.length):
            yield self.get_from_buffer(ind)


    def get_from_buffer(self, ind):
        self._has_fetched[ind] = True
        return self._buffer[ind]

    def push(self, item, weight=None):
        item = self.fn_convert_push( item )
        if not hasattr( self, 'item_type' ):
            self.item_type = type(item) # TODO: debug
        else:
            assert self.item_type == type(item)

        if self.weight_is_None is None:
            self.weight_is_None = weight is None

        assert self.weight_is_None == (weight is None)

        if self.weight_is_None:
            if self.n is None or len(self._buffer) < self.n:
                self._buffer.append(item)
                self._has_fetched.append(False)
            else:
                self._buffer[self.ind] = item
                self._has_fetched[self.ind] = False

            if self.n is not None:
                self.ind = int((self.ind + 1) % self.n)
                #TODO: repair the bug in rlsm
        else:

            assert self.temperature_weight is not None
            # the items are order by weight desc
            # TODO: maybe I need a lock for parallel processing
            weight_new = weight
            for ind in range( len(self._weight) ):
                if weight_new >= self._weight[ind] :
                    break
            else:
                ind = len( self._weight )

            if self.n is not None and len(self._buffer) >= self.n:
                # TODO: give new data more weight.
                _weight = self._weight
                print(_weight)
                weight = np.array(self._weight)

                weight = (weight - np.mean(weight)) / (np.std(weight) + 0.001)
                weight_exp = np.exp( weight * self.temperature_weight)
                weight_exp_normalized = weight_exp / np.sum(weight_exp)
                ind_out = np.random.choice(range(len(weight_exp_normalized)), p=weight_exp_normalized)
                # print( f'{len( self._weight )-ind_out}' )
                # ind_out = len( self._weight ) - 1 # for debu
                del self._buffer[ind_out], self._weight[ind_out], self._has_fetched[ind_out]

                if ind > ind_out:
                    ind -=1
            self._buffer.insert(ind, item)
            self._weight.insert(ind, weight_new)
            self._has_fetched.insert(ind, False)


    def merge(self, replaybuffer):
        for traj in replaybuffer.buffer:
            self.push(traj)


def bundle_cat(buffer, item, n, ind, is_batch):
    # --- Reshape item
    if isinstance(item, torch.Tensor):
        if not is_batch: #item.dim() < buffer.dim():
            item = item.unsqueeze(dim=0)
    elif isinstance(item, np.ndarray):
        if not is_batch: # item.ndim < buffer.ndim:
            item = np.expand_dims(item, axis=0)
    else:
        raise NotImplementedError

    # --- Push item
    if buffer is None:
        buffer = item
    elif n is None or len(buffer) < n:
        if isinstance(item, torch.Tensor):
            buffer = torch.cat((buffer, item), dim=0)
        elif isinstance(item, np.ndarray):
            buffer = np.concatenate((buffer, item), axis=0)
        else:
            raise NotImplementedError
    else:
        buffer[ind] = item

    return buffer

from dotmap import DotMap
from itertools import starmap
class Bundle(Buffer_Base):
    '''
        Item Type:      tuple               dict() (or Dotmap)
        Save form:      (X, Y)              dict(x=X, y=Y)

        The capital X means that all data are bundled into an entity.
    '''
    def __init__(self, n=None, _buffer=None, **kwargs):
        '''

        :param n: e.g,(X,Y)
        :type n:
        :param kwargs:
        :type kwargs:
        '''
        self.n = n
        # if n is None:
        #     warn('Bundle size not limited')
        self.ind = 0
        self.set_buffer( _buffer )
        super().__init__(**kwargs)


    def set_buffer(self, buffer):
        self._buffer = buffer
        if self.n is not None:
            self.n = max( self.n, self.length )

        if isinstance(buffer, tuple):
            for i in range(len(buffer)-1):
                assert buffer[i].shape[0] == buffer[i + 1].shape[0]
        elif isinstance(buffer, dict) or isinstance(buffer, DotMap):
            values = list(buffer.values())
            for i in range( len(values)-1 ):
                assert values[i].shape[0] == values[i+1].shape[0]

    def push(self, item):
        return self._push(item, is_batch=False)

    def push_batch(self, item):
        return self._push(item, is_batch=True)

    def _push(self, item, is_batch=False):
        item = self.fn_convert_push(item)

        if isinstance(item, tuple):
            buffer_new = starmap(
                lambda buffer_sub, item_sub: bundle_cat( buffer_sub, item_sub, n=self.n, ind=self.ind, is_batch=is_batch ),
                zip( self._buffer, item )
            )
            self._buffer = tuple(buffer_new)

        elif isinstance(item, dict) or isinstance(item, DotMap):
            if self._buffer is None:
                buffer_values = [None] * len(item)
                item_values = item.values()
            else:
                buffer_values = self._buffer.values()
                assert len( self._buffer.keys() ) ==len( item.keys() )
                item_values = map( lambda key: item[key], self._buffer.keys()  )
                # Make sure the orders of the values are exactly the same.

            values = starmap(
                lambda buffer_sub, item_sub: bundle_cat( buffer_sub, item_sub, n=self.n, ind=self.ind, is_batch=is_batch ),
                zip( buffer_values, item_values )
            )
            if self._buffer is None:
                self._buffer = type(item)()
            else:
                pass
                # Make sure the order are exactly the same
                # for key_1,key_2 in zip( self._buffer.keys(), item.keys() ):
                #     assert key_1 == key_2

            for k, v in zip(item.keys(), values):
                self._buffer[k] = v


        if self.n is not None:
            self.ind = (self.ind + 1) % self.n
        pass

    @property
    def length(self):
        if self._buffer is None:
            return 0

        data = self._buffer
        if isinstance(data, tuple):
            return len(data[0])#Even for numpy , you can also use len() which return
        elif isinstance(data, dict) or isinstance(data, DotMap):
            return len( list(data.values())[0] )
        else:
            return len(data)

    def get_data_by_inds(self, ind_all):
        data = self._buffer
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
        return self.fn_convert_get(self._buffer)


def tes_bundle():
    x = np.arange( 0, 5, 0.5 ).reshape( (5,-1) )
    y = np.arange( 0, 5, 0.5 ).reshape( (5,-1) )
    bundle = Bundle( n=3 )
    bundle.set_buffer( DotMap(x=x, y=y) )
    bundle.n = 30
    for i in range(100,120):
        bundle.push( dict( x=np.array([i, i+0.1]), y = np.array([i+.2, i+.3]) ) )
    for x in bundle.get_batch_enumerate(2, random=False, fill_last=False):
        print(x)

def tes_buffer():
    from numpy import array
    from torch import tensor
    import numpy as np

    buffer = Buffer(n=5, temperature_weight=1)
    a = list( range(10) )
    a = np.random.permutation( a )
    print(a)
    for i in a:
        buffer.push( i, weight=i )
        print('-'*10)
        print(buffer._buffer)
        print(buffer._weight)
        print('-' * 10)
    # for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
    #     print(x)
    exit()


    buffer = Buffer(n=2)

    buffer.push(0)
    buffer.push(1)
    print(buffer.has_fetched)
    buffer.get_from_buffer(0)
    print(buffer.has_fetched)
    buffer.push(2)
    print(buffer.has_fetched)
    # for x in buffer.enumerate():
    #     print(x)
    #     break


    # --- original
    buffer = Buffer(n=10)
    for i in range(10):
        buffer.push( i )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)

    # --- item: tuple
    buffer = Buffer( n=10 )
    for i in range(10):
        buffer.push( tensor( [i,i+0.1] ) )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)

    print('--- item: tuple, get_form: bundle')
    buffer = Buffer( n=10, get_form='bundle' )
    for i in range(10):
        buffer.push( tensor([i,i+0.1]) )
    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)


    print('--- item: dict, get_form: bundle')
    buffer = Buffer( n=10, get_form='bundle' )


    for i in range(10):
        buffer.push( DotMap(x=tensor(i),y=tensor(i+0.1) ) )

    for x in buffer.get_batch_enumerate(3, random=False, fill_last=False):
        print(x)
    print('sample result')
    for i in range(10):
        print( buffer.sample(3) )





if __name__ == '__main__':
    # tes_bundle()
    tes_buffer()