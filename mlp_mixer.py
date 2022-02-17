import os
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, params_lin1, params_lin2):
        '''
        initialization
        :param params_lin1: tuple in the form (in_features, out_features) of linear layer 1
        :param params_lin2: tuple in the form (in_features, out_features) of linear layer 2
        '''

        super(MLP, self).__init__()
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(params_lin1[0], params_lin1[1]),
            nn.GELU(),
            nn.Linear(params_lin2[0], params_lin2[1])
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
    def __init__(self, n_patches, n_channels, hyper_dict):
        ''' Mixer layer, described in the class comment
        :param hyper_dict: has hyperparameters needed for the network architecture
            :param mlp1_lin1_dims: tuple describing the dimension of the first linear layer in the first MLP component
            :param mlp1_lin2_dims: tuple describing the dimension of the second linear layer in the first MLP component
            :param mlp2_lin1_dims: tuple describing the dimension of the first linear layer in the second MLP component
            :param mlp2_lin2_dims: tuple describing the dimension of the second linear layer in the second MLP component
        :param n_patches: number of rows of the input matrix i.e. number of patches 
        :param n_channels: number of columns of the input matrix i.e. number of channels
        '''
        self.layer_norm1 = nn.LayerNorm(n_channels) # TODO: da capire la dimensione del layernorm, forse dim=n_channels?
        self.layer_norm2 = nn.LayerNorm(n_channels)
        self.MLP1 = MLP(hyper_dict['mlp1_lin1_dims'], hyper_dict['mlp1_lin2_dims'])
        self.MLP2 = MLP(hyper_dict['mlp2_lin1_dims'], hyper_dict['mlp2_lin2_dims'])


    def forward(x):
        pass