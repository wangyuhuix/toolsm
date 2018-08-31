import numpy as np
class Dataset:
  '''
    data: all datas tuple. e.g. (x,label)
    placeholder: tensorflow placeholders tuple. e.g. (x_pl,label_pl)

    (choose either one)
    batch_size:
    batch_div:


  '''
  def __init__(self,
               data, placeholder,
               batch_size=None, batch_div=None,
               epoch_num=1
               ):
    if not isinstance(data, tuple):
      data = (data,)
    if not isinstance(placeholder,tuple):
      placeholder = (placeholder,)

    # count batch_size
    self.count = data[0].shape[0]
    if batch_size is not None:
      self.batch_size = batch_size
    else:
      if batch_div is not None:
        self.batch_size = self.count / batch_div
      else:
        self.batch_size = self.count
    # check whether data count is consistent
    for ind in range(len(data)):
      assert data[ind].shape[0] == data[ind-1].shape[0],'ind%d:%d,ind%d:%d'%( ind,data[ind].shape[0],ind-1,data[ind-1].shape[0] )

    if epoch_num == 1:
      assert self.count%self.batch_size == 0, 'count %d should exact divide size_batch %d' % (self.count, self.batch_size)

    self.feed_dict = dict.fromkeys( placeholder )
    self.data = data
    self.epoch_num = epoch_num
    self.reset()

  def reset(self):
    self.ind_current = 0
    self.ind_epoch = 0

  def next(self):
    data = [None]*len(self.data)
    count_surplus = self.batch_size
    while count_surplus>0:
      if self.ind_current >= self.count:
        if self.is_train:
          self.ind_current = 0
          if self.epoch_num is not None:
            self.ind_epoch = self.ind_epoch+1
            if self.ind_epoch >= self.epoch_num:
              raise StopIteration()
        else:
          raise StopIteration()
      if self.ind_current>=self.count:
        self.ind_current = 0
      if self.ind_current + count_surplus <= self.count-1+1:
        count_use = count_surplus
      else:
        count_use = self.count - 1 - self.ind_current + 1
      for ind in range( len( self.data ) ):
        data[ind] = self._append(data[ind], self.data[ind][self.ind_current:self.ind_current + count_use])
      self.ind_current += count_surplus
      count_surplus -= count_use
    for ind,key in enumerate(self.feed_dict.iterkeys()):
        self.feed_dict[key] = data[ind]
    return self.feed_dict

  def _append(self,bundle,item):
    if bundle is None:
      bundle = item
    else:
      bundle = np.append(bundle, item, axis=0)
    return bundle