""" Patch and Positional embedding """

from torch import nn as nn
import torch
#import cv2


class PatchEmbedding(nn.Module):
    """ Patch grid and flattening (reshaping the input tensor), then linear embedding.
    Parameters
    ----------
    img_size : tuple
            Size of one single image
    in_chans : int
            Channels of on image (for an RGB image is 3)
    patch_size : int
            Size of a single patch of an image (assume is a square patch - one dimension is enough)
    embed_dim : int
            Embedding dimension
    Attributes
    ----------
    n_patches : int
            Number of patches of a single image
    linear_embedding : nn.Linear
            Linear embedding of the patches
    """

    def __init__(self, img_size, in_chans, patch_size, embed_dim):
        super().__init__()
        self.img_size = img_size  # tupla
        self.patch_size = patch_size
        self.in_chans = in_chans

        # check divisibility
        assert (self.img_size[0] % patch_size == 0), "H needs to be divisible by patch size"
        assert (self.img_size[1] % patch_size == 0), "W needs to be divisible by patch size"

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size * self.patch_size)  # H*W / P^2

        self.linear_embedding = nn.Linear((self.patch_size * self.patch_size) * self.in_chans, embed_dim)
        # bias=True by default

    def forward(self, x, plot=False):
        b, c, h, w = x.shape

        x = x.reshape(b, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        # (b, c, h/p, p, w/p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (b, h/p, w/p, c, p, p)
        x = x.flatten(1, 2)  # (b, h*w / p^2, c, p, p)
        # h*w / p^2 = number of patches, each patch is a matrix pxp with c channels
        
        x = x.flatten(2, 4)  # (b, h*w / p^2, c*p^2)

        out = self.linear_embedding(x)

        return out


class PositionEmbedding(nn.Module):
    """ Position learnable embedding.
    Parameters
    ----------
    img_size : tuple
            Size of one single image
    patch_size : int
            Size of a single patch of an image (assume is a square patch - one dimension is enough)
    embed_dim : int
            Embedding dimension
    Attributes
    ----------
    n_patches : int
            Number of patches of a single image
    position_embedding : nn.Parameter
            Initialization of the tensor for position embedding. The size here is (n_patches + 1) x embed_dim,
            in order to account for the cls_token
    """

    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.n_patches = (img_size[0] * img_size[1]) // (patch_size * patch_size)  # H*W / P^2
        # learnable 1D position embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(self.n_patches + 1, embed_dim))  # (n + 1, e)

    def forward(self, patch_embedded_x):
        out = patch_embedded_x + self.position_embedding
        return out

