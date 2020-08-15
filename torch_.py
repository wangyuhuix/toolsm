import torch

from torch import tensor
import math


from torch import nn
import torch.nn.functional as F

def toTensor(a, dtype=torch.float32):
    return torch.tensor(a, device=torch.device('cuda'), dtype=dtype, requires_grad=False)

def toNumpy(a):
    return a.cpu().numpy()


# def toTensorSwiss(a, dtype=torch.float32 ):
#     return CovertSwiss(a, fn_convert=toTensor, dtype=dtype)
#
#
# def toNumpySwiss(a, dtype=torch.float32 ):
#     return CovertSwiss(a, fn_convert=toNumpy, dtype=dtype)

def CovertSwiss(a, fn_convert, **kwargs):
    if isinstance(a, tuple ):
        a = tuple( map( lambda x_:fn_convert(x_, **kwargs), a ) )
    elif isinstance( a, dict ) or isinstance(a, DotMap):
        a_values = map( lambda x_:fn_convert(x_, **kwargs), a.keys() )
        for k,v in zip(a.keys(), a_values):
            a[k] = v
    else:
        a = fn_convert(a, **kwargs)
    return a


class Pd(object):
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def log_prob(self, x):
        return - self.neglogp(x)
    def prob(self, x):
        return self.log_prob(x).exp()

class DiagNormal(Pd):
    def __init__(self, mean, var=None, logvar=None, std=None, logstd=None):
        self.mean = mean
        ls = [var, logvar,std, logstd]
        cnt_NotNone = 0
        for i in ls:
            if i is not None:
                cnt_NotNone += 1
                if cnt_NotNone >= 2:
                    raise Exception('Only one parameter of covariance should be specified')

        if cnt_NotNone == 0:
            raise Exception('Please specify variance')

        if var is not None:
            self.std =var.sqrt()
            self.logstd = self.std.log_row()
        elif logvar is not None:
            self.logstd = logvar.mul(0.5)
            self.std = self.logstd.exp()
        elif std is not None:
            self.std = std
            self.logstd = std.log_row()
        elif logstd is not None:
            self.logstd = logstd
            self.std = self.logstd.exp()

    def mode(self):
        return self.mean

    @property
    def variance(self):
        return self.std**2

    def neglogp(self, x):
        return 0.5 * torch.sum(((x - self.mean) / self.std)**2, dim=-1) \
               + 0.5 * math.log(2.0 * math.pi) * x.shape[-1] \
               + torch.sum(self.logstd, dim=-1)

    def kl(self, other):
        assert isinstance(other, DiagNormal)
        return torch.sum(other.logstd - self.logstd + (self.std ** 2 + (self.mean - other.mean)**2) / (2.0 * (other.std)**2) - 0.5, dim=-1)

    def entropy(self):
        return torch.sum(.5* self.logstd + .5 * torch.log(2.0 * math.pi * math.e), dim=-1)

    def sample(self, sample_shape=None):
        return tensor( self.rsample(sample_shape).data )

    def rsample(self, sample_shape=None):
        shape_new = self.mean.shape
        if sample_shape is not None:
            if isinstance(sample_shape, tuple):
                sample_shape = list( sample_shape )
            elif not isinstance( sample_shape, list ):
                sample_shape = [sample_shape]
            shape_new = sample_shape + list(self.mean.shape)
            shape_new = tuple(shape_new)
        return self.mean + self.std * self.mean.new(torch.Size(shape_new)).normal_()

import numpy as np
class FullyConnected_NN(nn.Module):
    def __init__(self, n_units, v_initial=None, ac_fn = F.relu):
        super().__init__()

        fc_all = []
        for ind_layer in range(len(n_units) -1):
            fc = nn.Linear(in_features=n_units[ind_layer], out_features=n_units[ind_layer + 1])
            setattr(self, f'fc_{ind_layer}', fc )
            fc_all.append( fc )
        self.fc_all = fc_all
        self.ac_fn = ac_fn

        if v_initial is not None:
            ind_all_lastlayer = None
            # Maybe not easy to understand, but it is right and please see my note
            for ind_layer in range( len(fc_all) ):
                fc = fc_all[ind_layer]
                n = fc.out_features
                ind_all = np.random.permutation(n)
                # ind_all = torch.randperm(n)

                ind_all_zero = ind_all[: n // 2]
                fc.bias.data[ind_all_zero] = 0.

                if ind_layer == 0:
                    fc.weight.data[ ind_all_zero, :] = 0.
                else:
                    if ind_layer == len(fc_all) - 1:#final layer
                        fc.bias.data.fill_(v_initial)
                    else:
                        ind_all_notzero = ind_all[n // 2:]
                        weight_notzero = fc.weight.data[ind_all_notzero, :].detach()

                    ind_all_zero = ind_all_lastlayer[n_lastlayer//2:]
                    fc.weight.data[ :, ind_all_zero ] = 0.

                    if ind_layer ==  len(fc_all) - 1:
                        pass
                    else:
                        pass
                        fc.weight.data[ind_all_notzero, : ] = weight_notzero

                ind_all_lastlayer = ind_all
                n_lastlayer = n


    def forward(self, x):
        for fc in self.fc_all[:-1]:
            x = self.ac_fn( fc(x) )
        x = self.fc_all[-1](x)
        return x




def tes_FullyConnected_NN():
    fullyconnected_nn = FullyConnected_NN(n_units=[10, 20])
    x = torch.randn( (2,10) )
    y = fullyconnected_nn(x)
    print(y)
    exit()



from torch.nn import Parameter, init
import math


def linear_multihead_op(x, weight, bias, n_head_in):
    '''

    :param x:
    :type x:
    :param weight:
    :type weight:
    :param bias:
    :type bias:
    :return: [batch, head, feature]
    :rtype:
    '''
    '''
        n: batch count
        h: head count
        f(g): (new) feature count
    '''
    if n_head_in is None:
        if x.dim() == 1:
            output = torch.einsum('f,hfg->hg', [x, weight])
        elif x.dim() == 2:
            output = torch.einsum('nf,hfg->nhg', [x, weight])
        else:
            raise NotImplementedError
    else:
        if x.dim() == 2:
            output = torch.einsum('hf,hfg->hg', [x, weight])
        elif x.dim() == 3:
            output = torch.einsum('nhf,hfg->nhg', [x, weight])
        else:
            raise NotImplementedError

    if bias is not None:
        output += bias
    return output


# Copy from torch.init.kaiming_uniform_
def kaiming_uniform_multihead(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = init._calculate_correct_fan(tensor[0], mode)# My modify, only caculate for one head
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Linear_MultiHead(nn.Module):
    def __init__(self, in_features, out_features, n_head_in, n_head_out, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_head_in = n_head_in
        self.n_head_out = n_head_out

        self.weight = Parameter(torch.Tensor(n_head_out, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(n_head_out, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

        kaiming_uniform_multihead(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = linear_multihead_op(x, self.weight, self.bias, self.n_head_in)
        return output


    def extra_repr(self):
        return 'in_features={}, out_features={}, n_head={}, bias={}'.format(
            self.in_features, self.out_features, self.n_head, self.bias is not None
        )

def tes_Linear_MultiHead():
    n_head_feature = 4
    n_head_feature_new = 5
    n_head = 3
    x = torch.randn(10, n_head, n_head_feature)

    # w = torch.randn(n_head, n_head_feature, n_head_feature_new)
    # y_einsum = linear_multihead_op(x, w)

    linear_multihead = Linear_MultiHead(n_head_feature, out_features=n_head_feature_new, n_head_in=n_head, n_head_out=n_head,
                                        bias=False)
    w = linear_multihead.weight
    y_einsum = linear_multihead(x)

    for ind_head in range(n_head):
        x_ = x[:, ind_head]
        w_ = w[ind_head]
        y_ = torch.matmul(x_, w_)
        print(y_einsum[:, ind_head] - y_)

    for ind in range( len(x) ):
        x_ = x[ind]
        y_ = linear_multihead( x_ )
        print( y_einsum[ind] - y_  )


    print( f'-------n_head_in=None' )

    x = torch.randn(10, n_head_feature)
    linear_multihead = Linear_MultiHead(n_head_feature, out_features=n_head_feature_new, n_head_in=None,
                                        n_head_out=n_head,
                                        bias=False)
    w = linear_multihead.weight
    y_einsum = linear_multihead(x)


    for ind in range(len(x)):
        x_ = x[ind]
        y_ = linear_multihead( x_ )
        # print( y_einsum[ind] - y_ )
        for ind_head in range(n_head):
            # x_head = x_[ ind_head ]
            w_head = w[ind_head]
            y_ = torch.matmul(x_, w_head)
            print(y_einsum[ind, ind_head] - y_)



            # x = torch.randn(10, n_head_feature)
    # y_einsum = linear_multihead(x)
    # for ind_head in range(n_head):
    #     x_ = x
    #     w_ = w[ind_head]
    #     y_ = torch.matmul(x_, w_)
    #     print(y_einsum[:, ind_head] - y_)

# tes_Linear_MultiHead()

class FullyConnected_MultiHead_NN(nn.Module):
    def __init__(self, n_units_shared, n_units_head, n_head, v_initial=None, ac_fn = F.relu):
        super().__init__()
        # units_n_head = units_n_head.copy()
        n_units_head.insert(0, n_units_shared[-1])

        fc_all = []
        for ind_layer in range(len(n_units_shared) -1):
            fc = nn.Linear(in_features=n_units_shared[ind_layer], out_features=n_units_shared[ind_layer + 1])
            setattr(self, f'fc_{ind_layer}', fc )
            fc_all.append( fc )

        for ind_layer in range(len(n_units_head) -1):
            if ind_layer == 0:
                n_head_in = None
            else:
                n_head_in = n_head

            fc = Linear_MultiHead(in_features=n_units_head[ind_layer], out_features=n_units_head[ind_layer + 1], n_head_in=n_head_in, n_head_out=n_head)
            # fc = nn.Linear(in_features=n_units_head[ind_layer], out_features=n_units_head[ind_layer + 1])#TOD: tmp
            setattr(self, f'fc_multihead_{ind_layer}', fc )
            fc_all.append( fc )

        self.fc_all = fc_all
        self.ac_fn = ac_fn

        if v_initial is not None:
            raise NotImplementedError


    def forward(self, x, head=None):
        input_ = x
        for fc in self.fc_all[:-1]:
            x = self.ac_fn( fc(x) )
        x = self.fc_all[-1](x)
        if head is not None:
            if input_.dim() == 1:
                x = x[head]
            elif input_.dim() == 2:
                x = x[ torch.arange( 0, len(input_), dtype=torch.long ), head ]
        return x

def tes_FullyConnected_MultiHead_NN():
    dnn = FullyConnected_MultiHead_NN(n_units_shared=[10, 20], n_units_head=[40], n_head=3)
    x = torch.randn((2, 10))
    y = dnn(x)
    for n in range( len(x) ):
        y_n = dnn( x[n] )
        print( y[n]-y_n )
    # exit()
# tes_FullyConnected_MultiHead_NN()








# This is based on Burda, Yuri, et al. "Exploration by Random Network Distillation." arXiv preprint arXiv:1810.12894 (2018).
from .learn import Buffer, Bundle
from dotmap import DotMap
from torch import optim
class InputTrainedJudger_MultiHead():
    def __init__(self, n_input, n_output, threshold_kwargs, train_kwargs, buffer_size=None, n_units=None, n_units_head=None, n_head=None, ac_fn_target='ELU', ac_fn='TanH'):

        if n_units is None:
            n_units = []

        if n_head is None:
            assert n_units_head is None or len(n_units_head) == 0
            kwargs = dict(
                n_units_=[n_input] + n_units + [n_output],
            )
            dnn_target = FullyConnected_NN( **kwargs, ac_fn=getattr(nn, ac_fn_target )() )
            dnn = FullyConnected_NN( **kwargs, ac_fn=getattr(nn, ac_fn )() )
        else:
            if n_units_head is None:
                n_units_head = []

            kwargs = dict(
                n_units_shared=[n_input] + n_units,
                n_units_head=n_units_head + [n_output],
                n_head=n_head
            )
            dnn_target = FullyConnected_MultiHead_NN( **kwargs, ac_fn=getattr(nn, ac_fn_target )() )
            dnn = FullyConnected_MultiHead_NN( **kwargs, ac_fn=getattr(nn, ac_fn )() )

        self.dnn = dnn
        self.dnn_target = dnn_target
        self.n_head = n_head
        self.threshold_kwargs = threshold_kwargs
        self.threshold_judger = 0

        self.buffer = Bundle( n=buffer_size )

        self.optimizer = optim.Adam(self.NN.parameters(), lr=1e-3)

        self.train_kwargs = train_kwargs

        self.loss_fn = nn.MSELoss()

        def loss_fn_judger(input, target):
            return (input - target).abs().mean(dim=-1, keepdim=True)

        self.loss_fn_judger = loss_fn_judger


    def add(self, x, head=None):
        if self.n_head is None:
            assert head is None
            y = self.dnn_target(x).detach()
            self.buffer.push( DotMap(x=x, y=y) )
        else:
            assert head is not None
            y = self.dnn_target(x, head).detach()
            self.buffer.push( DotMap(x=x, head=head, y=y))



    def train(self):
        # TODO: judge threshold by train loss

        kwargs = self.train_kwargs

        if 'loss_threshold' in kwargs:
            loss_threshold = kwargs['loss_threshold']
        else:
            loss_threshold = None

        for k in range( kwargs['n_iteration'] ):
            loss_sum = 0.

            for ind_batch, batch in enumerate( self.buffer.get_batch_enumerate( kwargs['n_batch'], random=True, fill_last=False)):
                if self.n_head is None:
                    model_output = self.dnn( batch.x )
                else:
                    model_output = self.dnn(batch.x, batch.head)

                loss = self.loss_fn(input=model_output, target=batch.y)
                loss_sum += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_final = loss_sum / (ind_batch + 1)

            if loss_threshold is not None:
                if loss_final <= loss_threshold:
                    break

        threshold_kwargs = self.threshold_kwargs
        with torch.no_grad():
            loss_all = []
            for ind_batch, batch in enumerate(
                    self.buffer.get_batch_enumerate(kwargs['n_batch'], random=False, fill_last=False)):
                if self.n_head is None:
                    model_output = self.dnn(batch.x)
                else:
                    model_output = self.dnn(batch.x, batch.head)

                loss = self.loss_fn_judger( model_output, batch.y )
                loss_all.append(loss)

            loss_all = torch.stack()

    def is_trained(self, x, head=None):
        if self.n_head is None:
            assert head is None
            y_target = self.dnn_target(x)
            y = self.dnn(x)
        else:
            assert head is not None
            y_target = self.dnn_target(x, head)
            y = self.dnn(x, head)

        l1loss_element = (y - y_target).abs().mean(dim=-1, keepdim=True)
        return l1loss_element <= self.threshold_judger


    def cuda(self):
        self.dnn_target.cuda()
        self.dnn.cuda()
        pass





if __name__ == '__main__':
    pass
    from torch.distributions.multivariate_normal import MultivariateNormal


    #-- test sample
    # mu = tensor([0.,0.])
    # sigma_v = tensor([1.,36.])
    #
    # dist = DiagNormal( mean=mu, var=sigma_v )
    # samples = dist.sample(2000)
    # import matplotlib.pyplot as plt
    # plt.scatter(samples[:,0], samples[:,1] )
    # plt.show()
    # exit()


    # mu = tensor([[1.0,2.0],[2.0,3.0]])
    # sigma_v = tensor([[1.0,2.0],[2.0,3.0]])
    # x = tensor([[0., 0], [0., 0]])
    # sigma = torch.stack( list( map( lambda v: v.diag(), torch.unbind(sigma_v,0) )  ) )
    #
    # probs = DiagNormal(mu, var=sigma_v).log_prob(x)
    # print(probs)
    #
    # probs = MultivariateNormal(mu, covariance_matrix=sigma).log_prob(x)
    # print(probs)
    #
    # probs = DiagNormal(mu, logvar=sigma_v).log_prob(x)
    # print(probs)
    #
    # sigma = torch.stack(list(map(lambda v: v.diag(), torch.unbind(sigma_v.exp(), 0))))
    # probs = MultivariateNormal(mu, covariance_matrix=sigma).log_prob(x)
    # print(probs)
    #
    # probs = DiagNormal(mu, std=sigma_v).log_prob(x)
    # print(probs)
    # sigma = torch.stack(list(map(lambda v: v.diag(), torch.unbind(sigma_v, 0))))
    # probs = MultivariateNormal(mu, scale_tril=sigma).log_prob(x)
    # print(probs)
    #
    # #--- test single distribution, multi x
    # mu = tensor([1.0,2.0])
    # sigma_v = tensor([1.0,2.0])
    # x = tensor([[0., 0], [1., 1]])
    # print( DiagNormal( mean=mu, var=sigma_v ).log_prob(x)  )
    # print( MultivariateNormal( loc=mu, covariance_matrix=sigma_v.diag() ).log_prob(x)  )

