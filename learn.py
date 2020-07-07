from warnings import warn
import numpy  as np
import math
from collections import namedtuple




class __Buffer_Base():
    def __init__(self, fn_convert_data=lambda x: x):
        self.fn_convert_data = fn_convert_data


    @property
    def len(self):
        raise NotImplementedError

    def can_sample(self, n_sample):
        return self.len >= n_sample

    def sample(self, n_sample):
        if not self.can_sample(n_sample):
            warn( f'Buffer contains {self.len}, while sample size is {n_sample} and has been adjusted to be {self.len}' )
        ind_all = np.random.choice(self.len, min( self.len, n_sample ), replace=False)
        return self.fn_convert_data( self.get_data_by_inds(ind_all) )


    def get_batch_all(self, n_batch, random=False):
        if random:
            ind_all = np.random.permutation( self.len ).tolist()
        else:
            ind_all = list(range( self.len ))
        n_iteration = int(math.ceil( self.len / n_batch ))

        if n_iteration*n_batch > self.len:
            ind_all.extend( ind_all[ :n_iteration*n_batch-self.len ]  )

        for i in range(n_iteration):
            data_batch = self.get_data_by_inds(ind_all[i * n_batch: (i + 1) * n_batch])
            yield self.fn_convert_data(data_batch)



class Buffer(__Buffer_Base):
    def __init__(self, n, **kwargs ):
        self.buffer = []
        self.n = n
        self.ind = 0
        super().__init__(**kwargs)

    @property
    def len(self):
        return len(self.buffer)

    def get_data_by_inds(self, ind_all):
        return [self.buffer[ind] for ind in ind_all]

    def push(self, item):
        if len(self.buffer) < self.n:
            self.buffer.append(item)
        else:
            self.buffer[self.ind] = item

        self.ind = (self.ind + 1) % self.n

    def merge(self, replaybuffer):
        for traj in replaybuffer.buffer:
            self.push(traj)

class Bundle(__Buffer_Base):
    def __init__(self, data, **kwargs):
        '''

        :param data: e.g,(X,Y)
        :type data:
        :param kwargs:
        :type kwargs:
        '''
        self.data = data
        if isinstance(data, tuple) or isinstance(data, list):
            for i in range( len(data)-1 ):
                assert data[i].shape[0] == data[i+1].shape[0]
        super().__init__(**kwargs)


    @property
    def len(self):
        data = self.data
        if isinstance(data, tuple) or isinstance(data, list):
            return len(data[0])#Even for numpy , you can also use len() which return
        else:
            return len(data)

    def get_data_by_inds(self, ind_all):
        data = self.data
        if isinstance(data, tuple) or isinstance(data, list):
            result = []
            for i in range(len(data) ):
                result.append( data[i][ind_all] )
            result = tuple(result)
        else:
            result = data[ind_all]

        return result



if __name__ == '__main__':
    # replaybuffer = Buffer(n=10)
    # for i in range(10):
    #     replaybuffer.push( i )
    #
    # for x in replaybuffer.get_batch_all(3, random=True):
    #     print(x)

    x = np.arange( 10 ).reshape( (5,-1) )
    # y = np.arange( 10,30 ).reshape( (5,-1) )
    dataset = Bundle( (x) )
    for x in dataset.get_batch_all(4, random=False):
        print(x)