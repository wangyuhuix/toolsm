#encoding:utf-8
from multiprocessing import Queue
import threading
import time
from multiprocessing import Pool
import inspect
import ctypes

class PreLoader(object):
    '''
    预加载器
    '''
    def __init__(self, items_raw, fun_generator, size_cache, size_pool=4, num_epoch=1):
        '''
        :param items_raw: 未加工产品
        :param fun_generator: 生产产品函数
        :param size_cache: 缓存的产品个数
        :param size_pool: 一次并发生产产品的个数
        :param num_epoch:
        '''
        self.__items_raw = items_raw
        self.__ind_items_raw = 0
        self.__items = Queue(maxsize=size_cache)

        self.__fun_generator = fun_generator

        self.__size_pool = size_pool
        self.__multiprocess = Pool(size_pool)

        self.__num_epoch = num_epoch
        self.__ind_epoch = 0

        self.__lock = threading.Condition()
        self.__useup = False


        self.__StartProduce()

    def __iter__(self):
        return self

    def __next__(self):
        print(self.__items)
        '''
        if self.__items.empty():
            print('queue empty')

            if self.__useup:
                raise StopIteration()
            '''
        item = self.__items.get()
        print(item)
        return item


    def __StartProduce(self):
        '''
        启动生产产品线程
        :return:
        '''
        self.thread_tmp = threading.Thread(target=self.__ProduceUtilFull)
        self.thread_tmp.start()

    def __ProduceUtilFull(self):
        '''
        生产直到将池子加满，等待每个产品的生产完再继续生产下一个产品，直到把所有产品全部生产完
        :return:
        '''
        while True:
            time.sleep(1)
            if self.__ind_items_raw == len(self.__items_raw):
                self.__ind_epoch += 1
                if self.__ind_epoch < self.__num_epoch:
                    self.__ind_items_raw = 0
                else:
                    self.__useup = True
                    break
            # get items for current batch
            ind_end_items_raw = min( self.__ind_items_raw+self.__size_pool, len(self.__items_raw))
            items_raw = self.__items_raw[ self.__ind_items_raw:ind_end_items_raw ]
            self.__ind_items_raw = ind_end_items_raw

            items = self.__multiprocess.map(self.__fun_generator, items_raw)
            for item in items:
                self.__items.put(item)
                # put 时，如果已经满了会自动等待

    def __del__(self):
        pass
        #self.stop_thread( self.thread_tmp )


    def _async_raise(tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(thread):
        _async_raise(thread.ident, SystemExit)

if __name__=='__main__':
    def fun_generator(i):
        return i
    preloader = PreLoader( range(-20,23),fun_generator,size_cache=1000, size_pool=100, num_epoch=3 )
    while True:
        i = preloader.__next__()
        if i:
            print(i)
        else:
            break
