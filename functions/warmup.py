import torch
import time
import copy
import numpy as np
import os
import torch.nn as nn
from util import pause
from util import getClassCount
from util import normImageCustom
from util import imshow
from util import visImage
from util import print_pers

import utils


# training with validation
def warmup(model, optimizer, dataloader_train, num_epochs_warmup, cuda):

    criterion = nn.CrossEntropyLoss()

    model.train()  # Set model to training mode

    # choose dataloader
    dataloaders_chosen = dataloader_train

    for epochs in range(0, num_epochs_warmup):

        # Iterate over data.
        for batch_num, (inputs, label) in enumerate(dataloaders_chosen):

            # cuda
            if cuda:
                inputs = inputs.to('cuda')
                label = label.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs, _, _ = model(inputs)
                if cuda:
                    outputs = outputs.to('cuda')

                # softmax
                _, preds = torch.max(outputs, 1)

                label.type(torch.int64)

                loss = criterion(outputs, label)

                # backward + optimize
                loss.backward()
                optimizer.step()

    return model