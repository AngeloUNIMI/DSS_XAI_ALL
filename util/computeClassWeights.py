import os
import pickle
import torch
from util import getClassCount


def computeClassWeights(dataset_sizes, dataloader_train, dataloader_val, cuda, dirResults):

    datasetSizeAll = dataset_sizes['train'] + dataset_sizes['val']
    classCountAll = getClassCount(dataloader_train) + getClassCount(dataloader_val)
    # if no count, put 1
    numSub = 10
    for listc, tt in enumerate(classCountAll):
        if tt < numSub:
            classCountAll[listc] = numSub
    weightsBCE = torch.FloatTensor(datasetSizeAll / classCountAll)

    # cuda
    if cuda:
        weightsBCE = weightsBCE.to('cuda')

    return weightsBCE
