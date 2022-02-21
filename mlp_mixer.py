import os
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        '''
        initialization
        :param in_dim: dimension of input of first FC layer
        :param hidden_dim: dimension of hidden layer (i.e. dim of out of first FC and dim of in of second FC)
        :param out_dim: dimension of output of second FC layer
        '''
        assert in_dim == out_dim # ? forse no? sì, perchè dal BLOCCO, input e output non devono cambiare, cfr. schema
        super(MLP, self).__init__()
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        def forward(self, x):
            return self.linear_gelu_stack(x)

class MixerLayer(nn.Module):
    '''
    Structure should be: 
    input (*) -> Layer Norm -> matrix[Patches x Channels] -> T -> matrix[Channels x Patches] -> MLP1 (weight sharing) ->
    -> matrix[Channels x Patches] -> T -> matrix[Patches x Channels] -> (*) Layer norm (**) -> MLP2 (weight sharing) ->
    -> (**) matrix[Patches x Channels]

    Note (*) and (**) represent skip connections
    '''
    def __init__(self, n_patches, n_channels, hidden_mlp1_dim, hidden_mlp2_dim):
        ''' Mixer layer, described in the class comment
        :param hyper_dict: has hyperparameters needed for the network architecture
            :param mlp1_lin1_dims: tuple describing the dimension of the first linear layer in the first MLP component
            :param mlp1_lin2_dims: tuple describing the dimension of the second linear layer in the first MLP component
            :param mlp2_lin1_dims: tuple describing the dimension of the first linear layer in the second MLP component
            :param mlp2_lin2_dims: tuple describing the dimension of the second linear layer in the second MLP component
        :param n_patches: number of rows of the input matrix i.e. number of patches 
        :param n_channels: number of columns of the input matrix i.e. number of channels
        '''
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_channels) # TODO: da capire la dimensione del layernorm, forse dim=n_channels? Sì, n_channels sarebbe la dimensione della rappresentazione dell'embedding del patch
        self.layer_norm2 = nn.LayerNorm(n_channels)

        # only two mlps because of weight sharing
        self.MLP1 = MLP(n_patches, hidden_mlp1_dim, n_patches) # teoricamente num patches, e hidden dimension
        self.MLP2 = MLP(n_channels, hidden_mlp2_dim, n_channels) # 'token mixing' teoricamente hidden dimension e ..


    def forward(self, x):
        '''
        :param x: has size [batch size, n_patches, n_channels]
        '''
        # layer norm, transp, mlp1, transp, layer norm, mlp2
        to_skip = self.layer_norm1(x)
        to_skip = to_skip.permute(0, 2, 1) # do the transposition, 0 remains there because it's the batch size
        to_skip = self.MLP1(to_skip) # weight sharing!!
        to_skip = to_skip.permute(0, 2, 1) # stesso permute di prima (transposition)
        x = x + to_skip # skip connections
        to_skip = self.layer_norm2(x)
        to_skip = self.MLP2(to_skip)
        return (x + to_skip)

#TODO: in articolo c'è il discorso su 1x1 convolution

class MLP_mixer(nn.Module):
    
