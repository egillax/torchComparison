library(torch)
library(reticulate)
source('ResNet.R')
source('Transformer.R')
source('Estimator.R')

torch::torch_manual_seed(42)
features <- 33420
rows <-11250

data <- torch_randint(0, features, size=c(rows, 200), dtype=torch_long())
targets <- torch_randint(0, 2, size=c(rows), dtype=torch_float32())
# 
# save to use same data in python
# numpy <- reticulate::import('numpy')
# numpy$save('data.npy', r_to_py(as.matrix(data)))
# numpy$save('targets.npy', r_to_py(as.matrix(targets)) )
# data <- torch_tensor(numpy$load('data.npy'), dtype=torch_float32())
# targets <- torch_tensor(numpy$load('targets.npy'), dtype=torch_float32())$squeeze()

modelParamsResNet <- list('numLayers'=8L,
                        'sizeHidden'=512L,
                        'hiddenFactor'=2L,
                        'residualDropout'=0.0,
                        'hiddenDropout'=0.0,
                        'sizeEmbedding'=256L,
                        'catFeatures'=features)

modelParamsTransformer <- list('numBlocks'=3L,
                               'numHeads'=8,
                               'dimToken'=64L,
                               'dimHidden'=512L,
                               'attDropout'=0.2,
                               'ffnDropout'=0.1,
                               'resDropout'=0,
                               'catFeatures'=features)

fitParamsResNet <- list('epochs'=20,
                        'learningRate'=3e-4,
                        'weightDecay'=0,
                        'batchSize'=2056,
                        'posWeight'=1)
fitParamsTransformer <- list('epochs'=10,
                             'learningRate'=3e-4,
                             'weightDecay'=0,
                             'batchSize'=16,
                             'posWeight'=1)

baseModel <- Transformer
modelParams <- modelParamsTransformer
fitParams <- fitParamsTransformer


estimator <- Estimator$new(baseModel = baseModel,
                           modelParameters = modelParams,
                           fitParameters = fitParams,
                           device='cuda:0')

dataset <- torch::tensor_dataset(data, targets)

estimator$fit(dataset)

# torch::torch_manual_seed(42)
# model <- do.call(ResNet, modelParameters)$to(device='cpu')
# 
# batch <- data[1:64,]$to(device='cpu')
# y <- targets[1:64]$to(device='cpu')
# 
# optimizer <- torch::optim_adam(params = model$parameters,
#                                lr = 3e-4,
#                                weight_decay = 1e-6)
# 
# criterion <- torch::nn_bce_with_logits_loss()
# 
# 
# numIterations <- 100
# updateIterations <- 10
# losses <- torch::torch_empty(updateIterations)
# model$train()
# ix <- 1
# 
# for (i in 1:numIterations) {
#   out <- model(batch)
#   loss <- criterion(out, y)
#   losses[ix] <- loss$detach()
#   loss$backward()
#   optimizer$step()
#   optimizer$zero_grad()
#   ix <- ix + 1
#   if (i %% updateIterations == 0) {
#     avgLoss <- losses$mean()$item()
#     print(paste0('Training Loss: ', avgLoss))
#     ix <- 1
#   }
# 
# }

# peak memory use 2989 MB, Avg time per epoch 1.356, final loss 0.692
# without embedding layer: 1877 MB, 1.064 sec, loss: 0.251, 9,114,625


# Transformer: 1.067 mins per epoch, 2,487,937 parameters
# peak_memory: 2765 MiB


