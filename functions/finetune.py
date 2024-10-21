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
from util.computeClassWeights import computeClassWeights
from util import print_pers

import utils


# training with validation
def finetune(model, criterion_metric, metric_switch,
                    optimizer, scheduler,
                    num_epochs, dataset_sizes, dataloader_train, dataloader_val,
                    batch_sizeP, modelName,
                    dirResults, iteration, r_orth, fileResultNameFull, log, cuda):

    #init time
    since = time.time()

    # init best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_val_loss = 1e6

    # class weights computed on train and val together
    print('\tGetting class weights...')
    weightsBCE = computeClassWeights(dataset_sizes, dataloader_train, dataloader_val, cuda, dirResults)

    criterion = nn.CrossEntropyLoss(weight=weightsBCE)

    # compute num batches
    numBatches = {}
    numBatches['train'] = np.round(dataset_sizes['train'] / batch_sizeP)
    numBatches['val'] = np.round(dataset_sizes['val'] / batch_sizeP)

    #print(classCountAll)
    #print(weightsBCE)

    # loop on epochs
    for epoch in range(num_epochs):

        # display
        if log:
            print_pers('\tEpoch {}/{}'.format(epoch+1, num_epochs), fileResultNameFull)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # init losses and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # choose dataloader
            if phase == 'train':
                dataloaders_chosen = dataloader_train
            if phase == 'val':
                dataloaders_chosen = dataloader_val

            # Iterate over data.
            for batch_num, (inputs, label) in enumerate(dataloaders_chosen):

                # get size of current batch
                sizeCurrentBatch = inputs.size(0)

                ##################
                #if batch_num > 10:
                    #break
                ##################

                # cuda
                if cuda:
                    inputs = inputs.to('cuda')
                    label = label.to('cuda')

                # display
                #if batch_num % 100 == 0:
                    #print_pers("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches[phase])), fileResultNameFull)

                # indexes
                #indStart = batch_num * batch_sizeP
                #indEnd = indStart + sizeCurrentBatch

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _, features_flat = model(inputs)
                    if cuda:
                        outputs = outputs.to('cuda')

                    # softmax
                    _, preds = torch.max(outputs, 1)

                    label.type(torch.int64)

                    #print(label)
                    #print(label.size())
                    #print(outputs)
                    #print(outputs.size())

                    # loss = criterion(outputs, label)

                    #####
                    if modelName == 'resnet18':
                        diff = utils.orth_dist(model.layer2[0].downsample[0].weight) + \
                               utils.orth_dist(model.layer3[0].downsample[0].weight) + \
                               utils.orth_dist(model.layer4[0].downsample[0].weight)
                        diff += utils.deconv_orth_dist(model.layer1[0].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer1[1].conv1.weight, stride=1)
                        diff += utils.deconv_orth_dist(model.layer2[0].conv1.weight, stride=2) + \
                                utils.deconv_orth_dist(model.layer2[1].conv1.weight, stride=1)
                        diff += utils.deconv_orth_dist(model.layer3[0].conv1.weight, stride=2) + \
                                utils.deconv_orth_dist(model.layer3[1].conv1.weight, stride=1)
                        diff += utils.deconv_orth_dist(model.layer4[0].conv1.weight, stride=2) + \
                                utils.deconv_orth_dist(model.layer4[1].conv1.weight, stride=1)
                    if modelName == 'resnet34':
                        diff = utils.orth_dist(model.layer2[0].downsample[0].weight) + \
                               utils.orth_dist(model.layer3[0].downsample[0].weight) + \
                               utils.orth_dist(model.layer4[0].downsample[0].weight)
                        diff += utils.deconv_orth_dist(model.layer1[0].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer1[1].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer1[2].conv1.weight, stride=1)
                        diff += utils.deconv_orth_dist(model.layer2[0].conv1.weight, stride=2) + \
                                utils.deconv_orth_dist(model.layer2[1].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer2[2].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer2[3].conv1.weight, stride=1)
                        diff += utils.deconv_orth_dist(model.layer3[0].conv1.weight, stride=2) + \
                                utils.deconv_orth_dist(model.layer3[1].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer3[2].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer3[3].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer3[4].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer3[5].conv1.weight, stride=1)
                        diff += utils.deconv_orth_dist(model.layer4[0].conv1.weight, stride=2) + \
                                utils.deconv_orth_dist(model.layer4[1].conv1.weight, stride=1) + \
                                utils.deconv_orth_dist(model.layer4[2].conv1.weight, stride=1)
                    #####

                    # print(label)
                    # print(label.size())
                    # print(outputs)
                    # print(outputs.size())

                    # loss = criterion(outputs, label)
                    if metric_switch:
                        # loss = criterion(outputs, label) + criterion_metric(outputs, label) + r_orth * diff
                        loss = criterion(outputs, label) + criterion_metric(features_flat, label) + r_orth * diff
                    else:
                        loss = criterion(outputs, label) + r_orth * diff

                    #print(loss)
                    #pause()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                with torch.no_grad():
                    # running_loss += loss.item() * inputs.size(0)
                    running_loss += loss.detach() * inputs.size(0)
                    running_corrects += torch.sum(preds == label.data.int())

            # update schedulers
            if phase == 'train':
                for schedulerSingle in scheduler:
                    schedulerSingle.step()

            # compute epochs losses
            with torch.no_grad():
                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_loss = running_loss.item() / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / (dataset_sizes[phase])

            # display
            if log:
                print_pers('\t\t{} Loss: {:.4f} Acc: {:.4f};'.format(phase, epoch_loss, epoch_acc), fileResultNameFull)

            # if greater val accuracy, deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc == best_acc:
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        # save model at epoch
        # if epoch % 10 == 0:
            # fileNameSave = 'modelsave_{0}_epoch_{1}.pt'.format(iteration+1, epoch)
            #torch.save(model.state_dict(), os.path.join(dirResults, fileNameSave))

        # del
        # del inputs, label
        # torch.cuda.empty_cache()

    # time
    time_elapsed = time.time() - since
    print_pers('\tTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), fileResultNameFull)
    print_pers('\tBest val Acc: {:4f}'.format(best_acc), fileResultNameFull)

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save final
    fileNameSaveFinal = 'modelsave_{0}_final.pt'.format(iteration+1)
    # torch.save(model.state_dict(), os.path.join(dirResults, fileNameSaveFinal))

    # del
    torch.cuda.empty_cache()

    # del
    del inputs, label
    del outputs, loss, preds

    return model