import math
import torch
from torch import nn
import torch.nn.functional as F


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class Transformer(nn.Module):
    def __init__(self, catFeatures=None, numBlocks=2, dimToken=64, dimout=1,
                 numHeads=8, attDropout=0.2, ffnDropout=0.2, resDropout=0,
                 headActivation=nn.ReLU,
                 activation=ReGLU,
                 ffnNorm = nn.LayerNorm,
                 headNorm = nn.LayerNorm,
                 attNorm = nn.LayerNorm,
                 dimHidden=512):
        super(Transformer, self).__init__()
        self.CategoricalEmbedding = Embedding(catFeatures + 1, dimToken)
        self.ClassToken = ClassToken(dimToken)
        self.layers = nn.ModuleList([])
        for layer_idx in range(numBlocks):
            layer = nn.ModuleDict({
                'attention': nn.MultiheadAttention(dimToken, numHeads,
                                                   dropout=attDropout, bias=True,
                                                   batch_first=True),
                'ffn': FeedForwardBlock(dimToken, dimHidden,
                                        biasFirst=True, biasSecond=True, dropout=ffnDropout,
                                        activation=activation),
                'attentionResDropout': nn.Dropout(resDropout),
                'ffnResDropout': nn.Dropout(resDropout),
                'ffnNorm': ffnNorm(dimToken)
            })
            if layer_idx != 0:
                layer['attentionNorm'] = attNorm(dimToken)
            self.layers.append(layer)

        self.head = Head(dimToken, bias=True, activation=headActivation,
                         norm=headNorm, dimOut=dimout)

    def forward(self,input):
        mask = torch.where(input == 0, True, False)
        x = self.CategoricalEmbedding(input)
        x = self.ClassToken(x)
        mask = torch.cat([mask, torch.zeros((x.shape[0], 1), device=mask.device,
                                           dtype=mask.dtype)], dim=1)
        for layer_idx, layer in enumerate(self.layers):
            xResidual = self.startResidual(layer, 'attention', x)

            if layer_idx==len(layer):
                dims = xResidual.shape
                xResidual, attnWeights = layer['attention'](
                    xResidual[:,-1].view((dims[0], 0, dims[2])),
                    xResidual,
                    xResidual, mask
                )
                x = x[:,-1].view(dims[0], 0, dims[2])
            else:
                xResidual, _ = layer['attention'](xResidual,
                                                  xResidual,
                                                  xResidual,
                                                  mask)
            x = self.endResidual(layer, 'attention', x, xResidual)

            xResidual = self.startResidual(layer, 'ffn', x)
            xResidual = layer['ffn'](xResidual)

            x = self.endResidual(layer, 'ffn', x, xResidual)

        x = self.head(x).squeeze()
        return x

    def startResidual(self, layer, stage, x):
        xResidual = x
        normKey = f'{stage}Norm'
        if normKey in layer:
            xResidual = layer[normKey](xResidual)
        return xResidual

    def endResidual(self, layer, stage, x, xResidual):
        xResidual = layer[f'{stage}ResDropout'](xResidual)
        x = x + xResidual
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dimToken=64, dimHidden=512, biasFirst=True, biasSecond=True,
                 dropout=0.0, activation=nn.ReLU):
        super(FeedForwardBlock, self).__init__()
        self.linear_first = nn.Linear(dimToken, dimHidden * 2, biasFirst)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.linear_second = nn.Linear(dimHidden, dimToken, biasSecond)

    def forward(self, input):
        x = self.linear_first(input)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_second(x)
        return x


class Head(nn.Module):

    def __init__(self, dim_in, bias, activation, norm, dimOut):
        super(Head, self).__init__()
        self.normalization = norm(dim_in)
        self.activation = activation()
        self.linear = nn.Linear(dim_in, dimOut, bias)

    def forward(self, input):
        x = input[:, -1]  # ?
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


class Embedding(nn.Module):

    def __init__(self, numEmbeddings, embeddingDim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(numEmbeddings, embeddingDim, padding_idx=1)

    def forward(self, input):
        x = self.embedding(input + 1)
        return x


# adds a class token embedding
class ClassToken(nn.Module):
    def __init__(self, dimToken):
        super(ClassToken, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(dimToken, 1))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def expand(self,*dims):
        if not dims:
            return self.weight
        new_dims = (1,) * (len(dims) -1)
        return self.weight.view(*new_dims, -1).expand(*dims, -1)

    def forward(self, input):
        return torch.cat([input, self.expand(len(input), 1)], dim=1)

