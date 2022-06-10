import numpy as np
import torch
from torch.utils.data import TensorDataset

from Estimator import Estimator
from ResNet import ResNet
from Transformer import Transformer


def get_params(model):
    if model.__name__ == 'ResNet':
        modelParams = {'numLayers': 8,
                       'sizeHidden': 512,
                       'hiddenFactor': 2,
                       'residualDropout': 0.0,
                       'hiddenDropout': 0.0,
                       'sizeEmbedding': 256,
                       'catFeatures': columns}
        fitParams = {'epochs': 20,
                     'learningRate': 3e-4,
                     'weightDecay': 0,
                     'batch_size': 2056}  # 1024
    else:

        modelParams = {'numBlocks': 3,
                       'numHeads': 8,
                       'dimToken': 64,
                       'dimHidden': 512,
                       'attDropout': 0.2,
                       'resDropout': 0,
                       'catFeatures': columns}

        fitParams = {'epochs': 10,
                     'learningRate': 3e-4,
                     'weightDecay': 0,
                     'batch_size': 16,
                     }

    return modelParams, fitParams


data = torch.as_tensor(np.load('data.npy'), dtype=torch.long)
targets = torch.as_tensor(np.load('targets.npy'), dtype=torch.float32).squeeze()
columns = 33420

baseModel = Transformer
modelParams, fitParams = get_params(baseModel)

estimator = Estimator(model=baseModel,
                      model_parameters=modelParams,
                      fit_parameters=fitParams,
                      device='cuda:0')

dataset = TensorDataset(data, targets)

estimator.fit(dataset)

# peak memory 1337 MB, time 0.914, loss 0.048
# without embedding, 997 MB, 0.48 sec, loss: 0.296, 9,114,625 parameters

# Transformer: 19,5 sec per epoch,  2,487,937 parameters. peak memory 839 MiB, loss: 0.178
