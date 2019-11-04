
import tensorflow as tf
import baselines.common.tf_util as U


def LeakyReLU_fn(alpha):
    def f(x):
        return LeakyReLU(x, alpha)
    return f

def LeakyReLU(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def variable2list( vars ):
    return [(v.name, str(v.shape)) for v in vars ]


#'''
class Saver:
    def __init__(self, var_list):
        self.__saver = tf.train.Saver(max_to_keep=1000000, var_list=var_list)

    def load_model(self, filename):
        self.__saver.restore(U.get_session(), filename)

    def save_model(self, filename):
        self.__saver.save(U.get_session(), filename)
#'''

__savers = []
def __get_saver(var_list_cur):
    global __savers
    for saver,var_list in __savers:
        if var_list is var_list_cur:
            return saver
    saver = tf.train.Saver(max_to_keep=1000000, var_list=var_list_cur)
    __savers.append( (saver, var_list_cur) )
    return saver

def load_model(filename, var_list=None):
    __get_saver(var_list).restore(U.get_session(), filename)

def save_model(filename, var_list=None):
    __get_saver(var_list).save(U.get_session(), filename, write_meta_graph=True)
