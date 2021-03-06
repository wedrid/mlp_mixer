import os
from unittest.mock import patch
import torch
from torch import nn
import einops

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
        #print("HELLOOOO")
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
    # Ci servono: 
        # - n_patches fully connected che genera rappresentazione latente
        #   le rappresentazioni latenti poi vanno in forma "maticiale"
        # - N mixer layers (N è un iperparametro)
        # - Global average pooling
        # - FC finale

    def __init__(self, img_h_w = 36, patch_dim = 12, n_channels = 16,  n_classes = 100, num_mixers_layers = 7, hidden_dim_mlp_token = 100, hidden_dim_mlp_channel = 100, debug = False):
        # :param patch_dims: dimensions of patch, will allow calculation of number of patches. Square image => square patches i.e. patch_dims = width = height
        # :param n_channels: aka patch latent representation dimention
        # :param hidden_dim_mlp_token: hidden dimension size for first mlp blocks (token mixing)
        # :param hidden_dim_mlp_channel: hidden dimension size for second mlp blocks (channel mixing)
        # :param n_classes: number of classes, defines size of last FC layer

        ''' 
        NOTE: si assumono immagini quadrate, se non sono quadrate, fare preprocessing aggiungendo padding
        '''
        self.debug = debug
        super().__init__()
        # 1) self.patch_embedding, in comune con ViT 
        # da (1) viene ricavato n_channels perchè è la dimensione del codice latente
        num_patches = (img_h_w // patch_dim) **2 # integer division, squared because image is a square duh
        if self.debug: 
            print(f"NUM PATCHES: {num_patches}")
        #self.embedder = PatchEmbedding((img_h_w, img_h_w), patch_size=patch_dim, embed_dim=n_channels)
        
        # linear embedder con conv e einops
        self.patch_embedder = nn.Conv2d(
            3,
            n_channels, #is dim
            kernel_size=patch_dim,
            stride=patch_dim,
        )
        
        self.mixerlayers = nn.ModuleList([MixerLayer(num_patches, n_channels, hidden_dim_mlp_token, hidden_dim_mlp_channel) for _ in range(num_mixers_layers)]) #TODO
        #self.mixerlayers = MixerLayer(num_patches, n_channels, hidden_dim_mlp_token, hidden_dim_mlp_channel)
        # in implementation reported in the paper, there is also a layernorm, which is not present in the macroscopic scheme
        self.pre_fc_layernorm = nn.LayerNorm(n_channels)
        self.fc_head = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        # 1 image tensor to patch embeddings tensor
        #out = self.embedder(x) # patch embeddings (batch size, n_patches, n_channels) check: [100, 9, 16]
        
        # embedding come in paper
        x = self.patch_embedder(
            x
        )  # (n_samples, hidden_dim, n_patches ** (1/2), n_patches ** (1/2))
        out = einops.rearrange(
            x, "n c h w -> n (h w) c"
        )  # (n_samples, n_patches, hidden_dim)
        
        # TODO: nel paper c'è scritto "per patch fully connected" ma non mi è molto chiaro
        if self.debug: 
            print(f"Embedding size: {out.size()}")
        # 2 pass through all the mixer layers
        for layer in self.mixerlayers:
            out = layer(out)
        #out = self.mixerlayers(out)
        #print(f"Before pooing size: {out.size()}")
        # 3 of the mixed layers, do global average pooling, which is simpy an average of the feature vectors
        out = out.mean(dim=1)
        if self.debug:
            print(f"After pooling size: {out.size()}")
        out = self.pre_fc_layernorm(out)
        out = self.fc_head(out)
        if self.debug: 
            print(f"After fc {out.size()}")
        return out

if __name__ == "__main__":
    import torch
    import torchvision
    import torchvision.transforms as transforms

    #hyperparameters
    batch_size = 100

    pad_totensor_transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()]) # does the padding, images 32x32 become 36x36 (symmetric increase) so that are divisible by three and patches are 12x12

    train_dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, transform=pad_totensor_transform, download=True)

    test_dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False, transform=pad_totensor_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # check patch embedding
    item = iter(train_loader).next()[0]
    print(f"Size of item: {item.size()}")

    embedder = PatchEmbedding(embed_dim=16, patch_size=12)
    embedding = embedder(item)
    print(f"Size of embedding: {embedding.size()}")
    mlpmixer = MLP_mixer()
    mlpmixer(item) 
