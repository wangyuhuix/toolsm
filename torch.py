import torch

from torch import tensor
import math

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

    def sample(self,sample_shape=None):
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

if __name__ == '__main__':
    from torch.distributions.multivariate_normal import MultivariateNormal


    #-- test sample
    mu = tensor([0.,0.])
    sigma_v = tensor([1.,36.])

    dist = DiagNormal( mean=mu, var=sigma_v )
    samples = dist.sample(2000)
    import matplotlib.pyplot as plt
    plt.scatter(samples[:,0], samples[:,1] )
    plt.show()
    exit()


    mu = tensor([[1.0,2.0],[2.0,3.0]])
    sigma_v = tensor([[1.0,2.0],[2.0,3.0]])
    x = tensor([[0., 0], [0., 0]])
    sigma = torch.stack( list( map( lambda v: v.diag(), torch.unbind(sigma_v,0) )  ) )

    probs = DiagNormal(mu, var=sigma_v).log_prob(x)
    print(probs)

    probs = MultivariateNormal(mu, covariance_matrix=sigma).log_prob(x)
    print(probs)

    probs = DiagNormal(mu, logvar=sigma_v).log_prob(x)
    print(probs)

    sigma = torch.stack(list(map(lambda v: v.diag(), torch.unbind(sigma_v.exp(), 0))))
    probs = MultivariateNormal(mu, covariance_matrix=sigma).log_prob(x)
    print(probs)

    probs = DiagNormal(mu, std=sigma_v).log_prob(x)
    print(probs)
    sigma = torch.stack(list(map(lambda v: v.diag(), torch.unbind(sigma_v, 0))))
    probs = MultivariateNormal(mu, scale_tril=sigma).log_prob(x)
    print(probs)

    #--- test single distribution, multi x
    mu = tensor([1.0,2.0])
    sigma_v = tensor([1.0,2.0])
    x = tensor([[0., 0], [1., 1]])
    print( DiagNormal( mean=mu, var=sigma_v ).log_prob(x)  )
    print( MultivariateNormal( loc=mu, covariance_matrix=sigma_v.diag() ).log_prob(x)  )

