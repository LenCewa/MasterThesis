# My (intelligent) agent that explores the action space, i.e. set of all possible functions
import torch

class ANN(object):
    def __init__(self, in_dim, out_dim, intermediate_dim):
        '''
        Creates a neural network: https://github.com/BethanyL/DeepKoopman/blob/master/networkarch.py
        :param in_dim: {int} dimension of the state space
        :param out_dim: {int} dimension of the action space (latent variable)
        :param intermediate_dim: {list of int} dimensions of intermediate layers
        '''

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = torch.nn.Sequential()

        old_dim = in_dim
        for i, dim in enumerate(intermediate_dim):
            self.model.add_module("layer %d" % i, torch.nn.Linear(old_dim, dim))
            self.model.add_module("activation %d" % i, torch.nn.Tanh())
            old_dim = dim
        self.model.add_module("output", torch.nn.Linear(old_dim, out_dim))