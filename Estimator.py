import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm


class Estimator:
    """
    A class that wraps around pytorch models. Using this class I can quickly add pytorch models without
    having to write much code.
    """

    def __init__(self, model, model_parameters, fit_parameters,
                 optimizer=torch.optim.Adam, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion=nn.BCEWithLogitsLoss(), device='cpu'):
        """

        Parameters
        ----------
        model : nn.Module                   A pytorch model with a forward or __call__ method
        model_parameters : dict             The parameters to pass on to the pytorch model
        fit_parameters : dict               The parameters for the estimator
        optimizer :                         A pytorch optimizer, defaults to AdamW
        scheduler :                         A pytorch learning rate scheduler, default is reduce on plateau
        criterion :                         A pytorch loss function, default is BCEWithLogitsLoss
        device :                            Device to use, either 'cpu' or 'cuda:x' where x is number of gpu
        """
        self.model = model(**model_parameters)
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.epochs = fit_parameters.get('epochs', 5)
        self.learning_rate = fit_parameters.get('lr', 3e-4)
        self.weight_decay = fit_parameters.get('weight_decay', 1e-5)
        self.results_dir = pathlib.Path(fit_parameters.get('results_dir', './results'))

        self.prefix = fit_parameters.get('prefix', 'Model')
        self.previous_epochs = fit_parameters.get('previous_epochs', 0)

        self.device = device
        self.model.to(device)
      
        self.optimizer = optimizer(params=self.model.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        self.criterion = criterion
        self.criterion.to(device)

        self.batch_size = fit_parameters['batch_size']

    def fit(self, dataset):
        """
        Function that fit's a model to data loaded with a pytorch dataloader. It uses early stopping with data
        loaded with test_dataloader

        Parameters
        ----------
        dataset :        A pytorch dataset
        test_dataset :   the validation set
        trial :          optuna trial instance, used for pruning

        Returns
        -------
        self :              Returns itself so I can chain together operations like fit().score()

        """
        sampler = BatchSampler(RandomSampler(data_source=dataset), batch_size=self.batch_size, drop_last=False)
        dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=None)

        times = []
        for epoch in range(self.epochs):
            start = time.time()
            loss = self.fit_epoch(dataloader)
            delta = time.time() - start

            current_epoch = epoch + 1 + self.previous_epochs
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f'Epochs: {current_epoch} | Train loss: {loss:.3f}'
                f'LR: {lr} | Time: {round(delta, 3)} seconds')
            times.append(delta)
        print(f'Average time per epoch: {round(np.mean(times), 3)} seconds')
        return self

    def fit_epoch(self, dataloader):
        """
        Fit's one epoch. An epoch is one round through the data you have available.

        Parameters
        ----------
        dataloader :        A pytorch dataloader

        Returns
        -------

        """
        batch_loss = torch.empty(len(dataloader))
        self.model.train()
        for batch_num, (batch, target) in enumerate(tqdm(dataloader)):
            batch = self._batch_to_device(batch)
            target = self._batch_to_device(target)
            y_pred = self.model(batch)
            loss = self.criterion(y_pred, target)
            batch_loss[batch_num] = loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return batch_loss.mean().item()

    @property
    def num_parameters(self):
        return sum([p.shape.numel() for p in self.model.parameters()])

    def _batch_to_device(self, batch):
        """
        Sends data in batch to device. If batch is a list it goes recursively through each element in it's list and
        sends it to the device.

        Parameters
        ----------
        batch :         The batch data. Can be a tensor, a list or a pytorch geometric Batch

        Returns
        -------

        """
        if isinstance(batch, torch.Tensor):  # or isinstance(batch, Batch):
            batch = batch.to(self.device)
        else:
            for ix, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    b = b.to(self.device)
                elif isinstance(b, list):
                    b = self._batch_to_device(b)
                else:
                    Warning('Unsupported type found in batch')
                batch[ix] = b
        return batch

  
    
