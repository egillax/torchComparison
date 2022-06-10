from torch import nn


class ResNet(nn.Module):
    def __init__(self, catFeatures=None, sizeEmbedding=128, sizeHidden=128, numLayers=4,
                 hiddenFactor=2, activation=nn.ReLU(), normalization=nn.BatchNorm1d, hiddenDropout=None,
                 residualDropout=None, dimOut=1, num_numerical_features=0):
        super(ResNet, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=catFeatures + 1, embedding_dim=sizeEmbedding,
                                      padding_idx=1)
        self.first_layer = nn.Linear(sizeEmbedding + num_numerical_features, sizeHidden)

        res_hidden = sizeHidden * hiddenFactor

        self.layers = nn.ModuleList(ResidualLayer(sizeHidden, res_hidden, normalization,
                                                  activation, hiddenDropout, residualDropout)
                                    for _ in range(numLayers))
        self.last_norm = normalization(sizeHidden)
        self.head = nn.Linear(sizeHidden, dimOut)
        self.last_act = activation

    def forward(self, input):
        cat_input = self.embedding(input+1)
        x = cat_input
        x = self.first_layer(x)

        for layer in self.layers:
            x = layer(x)
        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
      
      
class ResidualLayer(nn.Module):
    def __init__(self, size_hidden, res_hidden, normalization, activation,
                 hiddenDropout=None, residualDropout=None):
        super(ResidualLayer, self).__init__()
        self.norm = normalization(size_hidden)
        self.linear0 = nn.Linear(size_hidden, res_hidden)
        self.linear1 = nn.Linear(res_hidden, size_hidden)

        self.activation = activation
        self.hidden_dropout = hiddenDropout
        self.residual_dropout = residualDropout
        if hiddenDropout:
            self.hidden_dropout = nn.Dropout(p=hiddenDropout)
        if residualDropout:
            self.residual_dropout = nn.Dropout(p=residualDropout)

    def forward(self, input):
        z = input
        z = self.norm(z)
        z = self.linear0(z)
        z = self.activation(z)
        if self.hidden_dropout:
            z = self.hidden_dropout(z)
        z = self.linear1(z)
        if self.residual_dropout:
            z = self.residual_dropout(z)
        z = z + input
        return z
