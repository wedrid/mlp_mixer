""" ViT model and its modules """

from torch import nn as nn
import torch
from embeddings import PatchEmbedding, PositionEmbedding


# todo: questione pretraining / fine tuning per mlp - layer norm

# NB: dropout_value = 0.0 --> non viene applicato


class MultiHeadAttention(nn.Module):
    """ Multi-head Attention module.

    There are some units (heads) of self-attention. Each unit process a chunk (head_dim) of the embedded patches.

    Parameters
    ----------
    embed_dim : int
            Embedding dimension

    heads : int
            Number of units of self-attention

    Attributes
    ----------
    head_dim : int
            Dimension processed of a single unit of self-attention

    values, keys, queries : nn.Linear
            Linear projection of the input, to obtain

    fc_out : nn.Linear
            Final linear projection after concat the heads outputs

    """

    def __init__(self, embed_dim, heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (self.embed_dim // self.heads == self.head_dim), "Embedding size needs to be divisible by heads"
        # assert self.head_dim * self.heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)  # Wv*X
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)  # Wk*X
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)  # Wq*X
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)  # linear(concat), bias true di default

    def forward(self, x):
        # batch size = number of images
        # n_tokens = n_patches + 1
        batch_size, n_tokens, _ = x.shape  # linear embedding + positional embedding shape

        x_split = x.reshape(batch_size, n_tokens, self.heads, self.head_dim)  # b, n + 1, heads, e // heads
        x_split = x_split.permute(0, 2, 1, 3)  # b, heads, n_tokens, head_dim

        values = self.values(x_split)  # (b, heads, n_tokens, head_dim)
        keys = self.keys(x_split)  # (b, heads, n_tokens, head_dim)
        queries = self.queries(x_split)  # (b, heads, n_tokens, head_dim)

        keys_t = keys.transpose(-2, -1)  # (b, heads, head_dim, n_tokens)
        weights_mat = (queries @ keys_t)  # (b, heads, n_tokens, n_tokens)

        attention = torch.softmax(weights_mat / (self.head_dim ** (1 / 2)), dim=-1)  # (b, heads, n_tokens, n_tokens)

        av = attention @ values  # (b, heads, n_tokens, head_dim)

        # torno alla shape dell'input
        av = av.permute(0, 2, 1, 3)  # (b, n_tokens, heads, head_dim)
        av = av.reshape(batch_size, n_tokens, self.heads * self.head_dim)  # (b, n_tokens, heads * head_dim)
        # (b, n_tokens, heads * head_dim) = (b, n_tokens, embed_dim)

        out = self.fc_out(av)  # (b, n_tokens, embed_dim)

        return out, attention


class TransformerBlock(nn.Module):
    """ MultiHeadAttention, Normalization and MultiLayerPerceptron.

    Parameters
    ----------
    embed_dim : int
            Embedding dimension

    hidden_dim : int
            Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim)

    heads : int
            Number of units of self-attention

    dropout_value: float
             Amount of dropout to apply in the ff network

    Attributes
    ----------
    multi_head_attention : MultiHeadAttention
            Multi-head attention instance

    norm1, norm2 : nn.LayerNorm

    feed_forward : nn.Sequential
            Feed forward net

    """

    def __init__(self, embed_dim, heads, hidden_dim, dropout_value=0.0):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim=embed_dim, heads=heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        print("heads, hidden dim TB: ", heads, hidden_dim)

        # MLP
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        # x is like z0 in the paper
        mha_out, mha_atn_weights = self.multi_head_attention(self.norm1(x))
        z_ = mha_out + x  # zl'
        z = self.feed_forward(self.norm2(z_)) + z_  # zl

        return z, mha_atn_weights


class TransformerEncoder(nn.Module):
    """ Repeat the Transformer block n_layers times.

    Parameters
    ----------
    embed_dim : int
            Embedding dimension

    heads : int
            Number of units of self-attention

    n_layers : int
            Number of layers of Trasformer block

    hidden_dim: int
            Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim)

    dropout_value: float
             Amount of dropout to apply in the ff network

    Attributes
    ----------
    layers : nn.ModuleList
            List of Transformer blocks

    """

    def __init__(self, embed_dim, heads, n_layers, hidden_dim, dropout_value=0.0):
        super().__init__()
        self.heads = heads
        print("heads, hidden dim TE: ", heads, hidden_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim=embed_dim, heads=self.heads, hidden_dim=hidden_dim,
                                 dropout_value=dropout_value)
                for _ in range(n_layers)
            ]
        )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n_tokens, _ = x.shape  # (b, n_tokens, embed_dim)
        attention_weights = []
        for l, layer in enumerate(self.layers):
            x, atn = layer(x)
            attention_weights.append(atn)

        return x, attention_weights


class ViT(nn.Module):
    """ Repeat the Transformer block n_layers times.

    Parameters
    ----------
    img_size : int
            Image dimension (the image is supposed to be a square image)

    embed_dim : int
            Embedding dimension

    num_channels : int
            Number of image channels

    num_heads : int
            Number of units of self-attention

    num_layers : int
            Number of layers of Trasformer block

    num_classes : int
            Number of classes of the dataset

    patch_size : int
            Patch dimension

    hidden_dim : int
            Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim)

    dropout_value: float
             Amount of dropout to apply in the ff network and on the input encoding (after positional encoding)

    Attributes
    ----------
    cls_token: nn.Parameter
            Classification token

    patch_embedding, position_embedding, transformer, mlp_head, dropout: layers of the model

    """

    def __init__(self, img_size, embed_dim, num_channels, num_heads, num_layers, num_classes,
                 patch_size, hidden_dim, dropout_value=0.0):
        super(ViT, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = num_channels
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # layers
        self.patch_embedding = PatchEmbedding(img_size=(img_size, img_size), in_chans=num_channels,
                                              patch_size=patch_size, embed_dim=embed_dim)

        self.position_embedding = PositionEmbedding(img_size=(img_size, img_size), patch_size=patch_size,
                                                    embed_dim=embed_dim)

        self.transformer = TransformerEncoder(embed_dim=embed_dim, heads=num_heads, n_layers=num_layers,
                                              hidden_dim=hidden_dim, dropout_value=dropout_value)

        # one hidden layer with tanh as non-linearity at pre-training, one single linear layer at fine tuning
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Tanh(),
        #     nn.Linear(embed_dim, num_classes)
        # )

        self.mlp_head = nn.Sequential(  # funziona meglio
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout_value)

        # qui ho assunto di usare lo stesso valore di dropout sia per la FF che per l'MLP head ...

    def forward(self, x):
        b, c, h, w = x.shape

        # patch embedding
        patch_embedding = self.patch_embedding(x)  # plot=False di default
        # print("patch embedding shape: ", patch_embedding.shape)

        # cls_token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # print("cls_tokens shape: ", cls_tokens.shape)
        cls_patches = torch.cat([cls_tokens, patch_embedding], dim=1)  # (b, n + 1, e)
        # print("cls_patches shape: ", cls_patches.shape)

        # positional embedding
        pp_encoded = self.position_embedding(cls_patches)  # (b, n + 1, e)
        # print("pp_encoded shape: ", pp_encoded.shape)

        # dropout after positional embedding (as written in the paper)
        pp_encoded = self.dropout(pp_encoded)

        # transformer encoder
        pp_transformed, atn_weights = self.transformer(pp_encoded)  # (b, n + 1, e)
        # print("pp_transformed shape: ", pp_transformed.shape)

        # classification head
        cls_pred = pp_transformed[:, 0, :]  # (b, e)
        # print("cls_pred shape: ", cls_pred.shape)

        out = self.mlp_head(cls_pred)  # (b, num_classes)

        return out, atn_weights


if __name__ == '__main__':
    import cv2

    # test values
    b = 1
    c = 3
    img_size = (200, 200)  # h, w
    patch_size = 50
    embed_dim = 60

    image = cv2.imread("./imgs/prova_patch.jpg")  # 200, 200, 3
    ima = torch.from_numpy(image)
    ima = torch.unsqueeze(ima, 0)
    # cv2.imshow("Test Image", image)
    # cv2.waitKey(0)

    im = ima.permute(0, 3, 1, 2).float()
    # opencv return a BGR image - to get an RGB image we extract the channels and stack them in the 'right' order
    b_ch = im[:, 0, :, :]  # blue channel
    g_ch = im[:, 1, :, :]  # green channel
    r_ch = im[:, 2, :, :]  # red channel

    im_rgb = torch.stack([r_ch, g_ch, b_ch], dim=1)

    vit = ViT(img_size=img_size[0], embed_dim=embed_dim, num_channels=c, num_heads=3, num_layers=2, num_classes=4,
              patch_size=patch_size, hidden_dim=embed_dim)
    out, atn_weights_list = vit(im_rgb)
