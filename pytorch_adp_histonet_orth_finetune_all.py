# --------------------------
# IMPORT
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import warnings
# warnings.filterwarnings("ignore")
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
# import splitfolders
import random
from random import seed
# from random import random
from datetime import datetime
import pickle
import PIL
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
import copy
import cv2

# --------------------------
# PRIV FUNCTIONS
import util
import functions
from modelGeno.resnet_geno import resnet18_orth_mtl
from modelGeno.resnet_geno import resnet34_orth_mtl
from functions.DatasetFromSubset import DatasetFromSubset
from xai_metric.Explanation_generator import Explanation_generator as eg

# tsne
from sklearn.manifold import TSNE

# --------------------------
# CLASSES
from classes.classesADP import classesADP
from classes.classesALL import classesALL


def count_parameters(model):
    all_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, train_params


# --------------------------
# MAIN
if __name__ == '__main__':

    # params
    plotta = False
    log = True
    extOrig = 'png'
    extNew = 'png'
    num_iterations = 5  # 10
    # nFolds = 10
    batch_sizeP = 8  #32
    batch_sizeP_norm = 8
    batch_size_test = 80
    numWorkersP = 0
    n_neighborsP = 1
    fontSize = 22
    padSize = 30
    ms = 10
    num_epochs_train = 2  # 90

    sizeALL = 260
    sizeNulls = 130  # 130
    classesCNN = 4  # 3

    # switches
    metric_switch = True
    gcam = True
    tsne = True


    # ------------------------------------------------------------------- db info
    dirWorkspace = 'D:/Workspace/DB HEM - Public (test) (Unsharp)/ALL_IDB/'
    dirPretrainedModels = './pretrained_nets/'
    dbName = 'ALL_IDB2_Unsharpened_7.1_add'


    # ------------------------------------------------------------------- Enable CUDA
    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.cuda.DoubleTensor if cuda else torch.cuda.DoubleTensor
    if cuda:
        torch.cuda.empty_cache()
    print("Cuda is {0}".format(cuda))
    #util.pause()


    # ------------------------------------------------------------------- dirs
    dirDbOrig = dirWorkspace + dbName + '/'
    dirDbTest = dirWorkspace + dbName + '/datastore/'
    dirOutTrainTest = dirWorkspace + dbName + '/datastore_trainTest/'
    os.makedirs(dirDbTest, exist_ok=True)
    # os.makedirs(dirOutTrainTest, exist_ok=True)

    # dirPatches
    dirPatches = 'D:/Workspace/DB HEM - Public (test)/patches_256_overlap_0.25_toll_5_original/'
    csvFile = 'ALL_IDB1_patches_256_overlap_0.25_toll_5.csv'
    csvFile_full = os.path.join(dirPatches, csvFile)


    # ------------------------------------------------------------------- transform db
    # util.dbToDataStore(dirDbOrig, dirDbTest, extOrig, extNew, log)  # class numbers are increased +1

    # add patches with nothing to the 0 class
    # randomly sample 130 images
    """
    classVec, fileNameVec, columnNames = util.getAllClassesVec(classesALL, csvFile_full, log)
    count = 0
    for num, (classVec_single) in enumerate(classVec):
        if np.sum(classVec_single) == 0 and count < sizeNulls:
            # shutil.copy2(os.path.join(dirPatches, fileNameVec[num]), os.path.join(dirDbTest, '0'))
            img = plt.imread(os.path.join(dirPatches, fileNameVec[num]))
            pre, ext = os.path.splitext(fileNameVec[num])
            plt.imsave(os.path.join(dirDbTest, '0', pre + '.' + extNew), img, format=extNew)
            count = count + 1
    """

    # ------------------------------------------------------------------- define all models we want to try
    modelNamesAll = list()
    modelNamesAll.append({'name': 'resnet18', 'sizeFeatures': 512})
    # modelNamesAll.append({'name': 'resnet34', 'sizeFeatures': 512})

    # - ADP or IMAGENet
    trainModes = []
    trainModes.append('adp')
    # trainModes.append('imagenet_adp')
    for trainMode in trainModes:


        # ------------------------------------------------------------------- loop on models
        for i, (modelData) in enumerate(modelNamesAll):

            # dir results
            dirResult = './results/' + trainMode + '/' + modelData['name'] + '/'
            if not os.path.exists(dirResult):
                os.makedirs(dirResult)

            # result file
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            fileResultName = current_time + '.txt'
            fileResultNameFull = os.path.join(dirResult, fileResultName)
            fileResult = open(fileResultNameFull, "x")
            fileResult.close()

            # display
            if log:
                print()
                util.print_pers("Model: {0}".format(modelData['name']), fileResultNameFull)

            # indexes train test
            kf_train_test = KFold(n_splits=num_iterations, shuffle=True, random_state=42)
            # all_indexes = np.arange(260)  # 260 images in all_idb2
            all_indexes = np.arange(sizeALL+sizeNulls)  # 260 images in all_idb2
            # kf.get_n_splits(all_indexes)

            # ------------------------------------------------------------------- loop on iterations
            # init
            dataset_sizes = {}
            accuracyALL = np.zeros(num_iterations)
            CM_all = np.zeros((classesCNN, classesCNN))
            CM_perc_all = np.zeros((classesCNN, classesCNN))
            # for r in range(0, num_iterations): #(nFolds-1)
            # noinspection PyUnboundLocalVariable
            for r, (train_index, test_index) in enumerate(kf_train_test.split(all_indexes)):
                # print(r)
                # print(train_index)
                # print(test_index)

                # indexes train val
                kf_train_val = KFold(n_splits=num_iterations-1, shuffle=True)
                train_temp, val_temp = next(kf_train_val.split(train_index))
                train_index_2 = train_index[train_temp]
                val_index = train_index[val_temp]

                # display
                if log:
                    util.print_pers("", fileResultNameFull)
                    util.print_pers("Iteration n. {0}".format(r + 1), fileResultNameFull)
                    util.print_pers("", fileResultNameFull)

                # load model
                if modelData['name'] == 'resnet18':
                    currentModel = resnet18_orth_mtl(pretrained=False)
                if modelData['name'] == 'resnet34':
                    currentModel = resnet34_orth_mtl(pretrained=False)

                # change last layer, so we can load results
                # Multi-task learning
                # fc1
                new_classifier1 = nn.Linear(modelData['sizeFeatures'], classesADP[0]['numClasses'])
                currentModel.fc1 = new_classifier1
                # fc2
                new_classifier2 = nn.Linear(modelData['sizeFeatures'], classesADP[1]['numClasses'])
                currentModel.fc2 = new_classifier2
                # fc3
                new_classifier3 = nn.Linear(modelData['sizeFeatures'], classesADP[2]['numClasses'])
                currentModel.fc3 = new_classifier3

                dirPretrainedModel = dirPretrainedModels + trainMode + '/' + modelData['name'] + '/'
                currentModel.load_state_dict(torch.load(os.path.join(dirPretrainedModel, 'modelsave_1_final.pt')))

                del currentModel.fc1
                del currentModel.fc2
                del currentModel.fc3

                # block parameters
                for param in currentModel.parameters():
                    param.requires_grad = False  # frozen for warmup
                # change last layer
                new_fc = nn.Linear(modelData['sizeFeatures'], classesCNN)
                currentModel.fc = new_fc
                imageSize = 224

                """
                all_params_histotnet, train_params_histotnet = count_parameters(currentModel)
                print(all_params_histotnet, train_params_histotnet)
                modelC = models.resnet34()
                all_params_resnet, train_params_resnet = count_parameters(modelC)
                print(all_params_resnet, train_params_resnet)
                print()
                """

                #currentModel.double()
                # cuda
                if cuda:
                    currentModel.to('cuda')
                # log
                if log:
                    print(currentModel)

                """
                # split into classes
                # first delete dir
                if os.path.exists(dirOutTrainTest):
                    shutil.rmtree(dirOutTrainTest)
                # create
                if not os.path.exists(dirOutTrainTest):
                    os.makedirs(dirOutTrainTest)
                splitfolders.ratio(dirDbTest, output=dirOutTrainTest, seed=random.random(), ratio=(.6, .2, .2))
                util.print_pers("", fileResultNameFull)
                """

                # preprocess
                transform = {
                    'train':
                        transforms.Compose([
                            transforms.CenterCrop(size=(256, 256)),
                            transforms.Resize(size=(imageSize, imageSize), interpolation=transforms.InterpolationMode.BILINEAR),
                            #transforms.RandomRotation(45),
                            transforms.ToTensor()
                        ]),
                    'val':
                        transforms.Compose([  # [1]
                            transforms.CenterCrop(size=(256, 256)),
                            transforms.Resize(size=(imageSize, imageSize), interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.ToTensor()
                        ])
                }

                # ------------------------------------------------------------------- TRAIN
                # load data
                all_idb2 = datasets.ImageFolder(dirDbTest)
                # train
                all_idb2_train_subset = torch.utils.data.Subset(all_idb2, train_index_2)
                all_idb2_train = DatasetFromSubset(all_idb2_train_subset, transform['train'])
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP_norm, shuffle=True,
                                                                    num_workers=numWorkersP, pin_memory=True)
                util.print_pers("\tClassi: {0}".format(all_idb2.classes), fileResultNameFull)
                dataset_sizes['train'] = len(all_idb2_train)
                util.print_pers("\tDimensione dataset train: {0}".format(dataset_sizes['train']), fileResultNameFull)

                # val
                all_idb2_val_subset = torch.utils.data.Subset(all_idb2, val_index)
                all_idb2_val = DatasetFromSubset(all_idb2_val_subset, transform['val'])
                all_idb2_val_loader = torch.utils.data.DataLoader(all_idb2_val,
                                                                  batch_size=batch_sizeP_norm, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)
                util.print_pers("\tClassi: {0}".format(all_idb2.classes), fileResultNameFull)
                dataset_sizes['val'] = len(all_idb2_val)
                util.print_pers("\tDimensione dataset val: {0}".format(dataset_sizes['val']), fileResultNameFull)
                print()

                # mean, std
                print("Normalization...")
                # save norm
                fileNameSaveNorm = {}
                fileSaveNorm = {}
                meanNorm = {}
                stdNorm = {}
                dataloaders_all = list()
                dataloaders_all.append(all_idb2_train_loader)
                # dataloaders_all.append(all_idb2_val_loader)
                # dataset_sizes_all = dataset_sizes['train']+dataset_sizes['val']
                dataset_sizes_all = dataset_sizes['train']
                fileNameSaveNorm = os.path.join(dirResult, 'norm.dat')

                """
                # if file exist, load
                if os.path.isfile(fileNameSaveNorm):
                    # read
                    fileSaveNorm = open(fileNameSaveNorm, 'rb')
                    meanNorm, stdNorm = pickle.load(fileSaveNorm)
                    fileSaveNorm.close()
                # else, compute normalization
                else:
                """

                # compute norm for all channels together
                meanNorm, stdNorm = util.computeMeanStd(dataloaders_all, dataset_sizes_all, batch_sizeP_norm, cuda)
                # save
                fileSaveNorm = open(fileNameSaveNorm, 'wb')
                pickle.dump([meanNorm, stdNorm], fileSaveNorm)
                fileSaveNorm.close()

                # update transforms
                # train
                transform['train'] = transforms.Compose([
                    transforms.CenterCrop(size=(256, 256)),
                    transforms.Resize(size=(imageSize, imageSize), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                # val
                transform['val'] = transforms.Compose([
                    transforms.CenterCrop(size=(256, 256)),
                    transforms.Resize(size=(imageSize, imageSize), interpolation=transforms.InterpolationMode.BILINEAR),
                    #
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(90),
                    #
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                print()

                # update data loaders
                # train
                all_idb2_train = DatasetFromSubset(all_idb2_train_subset, transform['train'])
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP, shuffle=True,
                                                                    num_workers=numWorkersP, pin_memory=True)
                # val
                all_idb2_val = DatasetFromSubset(all_idb2_val_subset, transform['val'])
                all_idb2_val_loader = torch.utils.data.DataLoader(all_idb2_val,
                                                                  batch_size=batch_sizeP, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)

                # optim
                optimizer_ft = optim.SGD(currentModel.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)

                # metric learning
                from pytorch_metric_learning import miners, losses
                # miner = miners.MultiSimilarityMiner()
                miner = []
                criterion_metric = losses.TripletMarginLoss()

                # warm-up
                num_epochs_warmup = 2
                util.print_pers("Warm-up", fileResultNameFull)
                print()
                currentModel = functions.warmup(currentModel, optimizer_ft, all_idb2_train_loader, num_epochs_warmup, cuda)

                # re-enable grad
                for param in currentModel.parameters():
                    param.requires_grad = True  # re-enable gradients

                # optim + sched
                # optimizer_ft = optim.SGD(currentModel.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
                optimizer_ft = optim.SGD(currentModel.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
                exp_lr_scheduler = list()
                exp_lr_scheduler.append(lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5))

                # train
                util.print_pers("Training", fileResultNameFull)
                # train net
                r_orth = 0.1
                currentModel = functions.train_model_val(currentModel, miner, criterion_metric, metric_switch,
                                                         optimizer_ft, exp_lr_scheduler,
                                                         num_epochs_train, dataset_sizes, all_idb2_train_loader, all_idb2_val_loader,
                                                         batch_sizeP, modelData['name'],
                                                         dirResult, r, r_orth, fileResultNameFull, log, cuda)

                # visualize some outputs
                #functions.visualize_model(currentModel, all_idb2_val_loader, cuda, columnNames, num_images=6)
                #util.pause()
                print()

                # I need to compute outputs for training data
                # train loader with NO SHUFFLING
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True)
                # eval
                currentModel.eval()
                # zero the parameter gradients
                optimizer_ft.zero_grad()
                torch.no_grad()
                util.print_pers('Computing outputs for training data...', fileResultNameFull)
                # init
                outputsALL_train = torch.zeros(dataset_sizes['train'], classesCNN)
                labelALL_train = torch.zeros(dataset_sizes['train'])
                numBatches = {}
                numBatches['train'] = np.round(dataset_sizes['train'] / batch_sizeP)
                # loop on images
                for batch_num, (inputs, label) in enumerate(all_idb2_train_loader):                    # get size of current batch
                    sizeCurrentBatch = label.size(0)
                    if batch_num % 100 == 0:
                        print("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches['train'])))
                    # stack
                    indStart = batch_num * batch_sizeP
                    indEnd = indStart + sizeCurrentBatch
                    # extract features
                    if cuda:
                        inputs = inputs.to('cuda')
                        label = label.to('cuda')
                    # predict
                    with torch.set_grad_enabled(False):
                        outputs, _, _ = currentModel(inputs)
                        if cuda:
                            outputs = outputs.to('cuda')
                    outputsALL_train[indStart:indEnd, :] = outputs
                    labelALL_train[indStart:indEnd] = label
                print()


                # ------------------------------------------------------------------- TEST
                # torch.cuda.empty_cache()

                # display
                if log:
                    util.print_pers("Testing", fileResultNameFull)

                # eval
                currentModel.eval()
                # zero the parameter gradients
                optimizer_ft.zero_grad()
                torch.no_grad()

                # test transform
                transform['test'] = transforms.Compose([
                    transforms.CenterCrop(size=(256, 256)),
                    transforms.Resize(size=(imageSize, imageSize), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                # transform PLAIN for visualization
                transform_PLAIN = transforms.Compose([
                    transforms.CenterCrop(size=(256, 256)),
                    transforms.Resize(size=(imageSize, imageSize), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ])
                # load data
                all_idb2_test_subset = torch.utils.data.Subset(all_idb2, test_index)
                all_idb2_test = DatasetFromSubset(all_idb2_test_subset, transform['test'])
                all_idb2_test_loader = torch.utils.data.DataLoader(all_idb2_test,
                                                                  batch_size=batch_size_test, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)
                # load data with no normalization (for visualization)
                all_idb2_test_PLAIN = DatasetFromSubset(all_idb2_test_subset, transform_PLAIN)
                all_idb2_test_loader_PLAIN = torch.utils.data.DataLoader(all_idb2_test_PLAIN,
                                                                  batch_size=batch_size_test, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)

                dataset_sizes['test'] = len(all_idb2_test)
                util.print_pers("\tDimensione dataset test: {0}".format(dataset_sizes['test']), fileResultNameFull)
                numBatches['test'] = np.round(dataset_sizes['test'] / batch_size_test)

                # loop on images
                # init
                predALL_test = torch.zeros(dataset_sizes['test'])
                predALL_test_majority = torch.zeros(dataset_sizes['test'])
                labelsALL_test = torch.zeros(dataset_sizes['test'])
                for batch_num, (inputs, label) in enumerate(all_idb2_test_loader):

                    # gen = iter(all_idb2_test_loader_PLAIN)
                    # inputs_test_PLAIN, _ = next(gen)
                    ##################
                    #if batch_num > 10:
                        #break
                    ##################

                    # get size of current batch
                    sizeCurrentBatch = label.size(0)

                    if batch_num % 100 == 0:
                        print("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches['test'])))

                    if plotta:
                        util.visImage(inputs)
                        util.print_pers("\tClasse: {0}".format(label), fileResultNameFull)
                        # util.pause()

                    # stack
                    indStart = batch_num * batch_sizeP
                    indEnd = indStart + sizeCurrentBatch

                    # extract features
                    if cuda:
                        inputs = inputs.to('cuda')
                        label = label.to('cuda')

                    # predict
                    with torch.set_grad_enabled(False):
                        outputs, _, _ = currentModel(inputs)
                        if cuda:
                            outputs = outputs.to('cuda')

                        # softmax
                        _, preds = torch.max(outputs, 1)

                        predALL_test[indStart:indEnd] = preds
                        labelsALL_test[indStart:indEnd] = label

                print()
                # ----------------------------------------------------------------------
                # distance plots
                util.print_pers('Generating distance plots...', fileResultNameFull)
                from matplotlib.backends.backend_pdf import PdfPages
                # create a PdfPages object
                pdf = PdfPages(os.path.join(dirResult, 'distances_{}.pdf'.format(r+1)))
                # batch size 80 so all test images in the batch
                # look for closest samples, distance function on the 'outputs' vector
                # distanza tutti-tutti
                num_closest = 3
                from scipy.spatial import distance_matrix
                # dm = distance_matrix(outputs.cpu(), outputs.cpu())
                dm = distance_matrix(outputs.cpu(), outputsALL_train)  # look for closest in training output
                for i in range(0, dataset_sizes['test']):  #i-th sample in testing db
                # for i in [10, 50, 70]:  # short version
                    # display
                    if i % 10 == 0:
                        print("\tImg n. {0} / {1}".format(i, int(dataset_sizes['test'])))
                    #
                    distances = dm[i]  # row with distances of element i to others
                    closest = np.argsort(distances)
                    # we need the first 'num_closest' elements
                    # also, first element is distance to itself (0), so we remove it
                    # closest = closest[0:num_closest+1]
                    closest = closest[0:num_closest] # against training data, no need to remove

                    # output  for i-th sample as majority voting of closest samples
                    votes = []
                    for close in closest:
                        votes.append(int(labelALL_train[close].item()))
                    output_majority = util.majority_vote(votes)
                    predALL_test_majority[i] = output_majority

                    filenameImg_full = all_idb2.imgs[test_index[i]][0]
                    filenameImg = os.path.basename(filenameImg_full)
                    # from torchvision.utils import save_image
                    imgs_rgb = []
                    imgs_gcam_normal = []
                    imgs_gcam_metric = []
                    titles_rgb = []
                    titles_gcam = []
                    egx = eg()
                    imgs_rgb.append(plt.imread(filenameImg_full))
                    # titles.append(filenameImg + ' \n Output: ' + str(int(preds[i].item())) + '; Label: ' + str(int(labelsALL_test[i].item())))
                    titles_rgb.append(filenameImg + ' \n Output: ' + str(output_majority) + '; Label: ' + str(int(labelsALL_test[i].item())))
                    for i_close, (close) in enumerate(closest):
                        # filenameImgClose = os.path.basename(all_idb2.imgs[test_index[close]][0])
                        filenameImgCloseFull = all_idb2.imgs[train_index_2[close]][0]
                        filenameImgClose = os.path.basename(filenameImgCloseFull)
                        imgs_rgb.append(plt.imread(filenameImgCloseFull))
                        dist_str = "Dist: {:.2f}".format(distances[close])
                        titles_rgb.append(filenameImgClose + ' \n Label: ' + str(int(labelALL_train[close].item())) + '; ' + dist_str)

                        # ---
                        inputs_1, image_1, inputs_2, image_2 = egx.get_input_from_path(filenameImg_full, filenameImgCloseFull, (224,224), meanNorm, stdNorm)
                        embed_1, map_1, embed_2, map_2, fc = egx.get_embed(currentModel, inputs_1=inputs_1, inputs_2=inputs_2)

                        # generate Grad-CAM
                        map_1.retain_grad()
                        map_2.retain_grad()

                        product_vector = torch.mul(embed_1, embed_2)
                        product = torch.sum(product_vector)
                        product.backward(torch.tensor(1.).to('cpu'), retain_graph=True)

                        gradcam_1 = egx.RGradCAM(map_1)
                        gradcam_2 = egx.RGradCAM(map_2)

                        image_overlay_1 = image_1 * 0.7 + egx.imshow_convert(gradcam_1) / 255.0 * 0.3
                        image_overlay_2 = image_2 * 0.7 + egx.imshow_convert(gradcam_2) / 255.0 * 0.3

                        # append
                        imgs_gcam_normal.append(image_overlay_1)
                        imgs_gcam_normal.append(image_overlay_2)

                        # --------------------------------------------------------------------------------
                        '''
                            Generate overall activation map using activation decomposition,
                            or the rectified Grad-CAM (RGrad-CAM)
                            They are equivalent in certain situation, see the paper for details
                        '''

                        # compute the overall activation map with decomposition (no bias term)
                        egx.Decomposition = egx.Overall_map(map_1=map_1, map_2=map_2, fc_1=fc, fc_2=fc, mode='GMP')

                        decom_1 = cv2.resize(np.sum(egx.Decomposition, axis=(2, 3)), (224, 224))
                        decom_1 = decom_1 / np.max(decom_1)
                        decom_2 = cv2.resize(np.sum(egx.Decomposition, axis=(0, 1)), (224, 224))
                        decom_2 = decom_2 / np.max(decom_2)

                        image_overlay_1 = image_1 * 0.7 + egx.imshow_convert(decom_1) / 255.0 * 0.3
                        image_overlay_2 = image_2 * 0.7 + egx.imshow_convert(decom_2) / 255.0 * 0.3

                        # append
                        imgs_gcam_metric.append(image_overlay_1)
                        titles_gcam.append(filenameImg + ' \n Output: ' + str(output_majority) + '; Label: ' + str(int(labelsALL_test[i].item())))
                        imgs_gcam_metric.append(image_overlay_2)
                        titles_gcam.append(filenameImgClose + ' \n Label: ' + str(int(labelALL_train[close].item())) + '; ' + dist_str)

                        """
                        plt.figure()
                        plt.suptitle('Activation Decomposition (Overall map)')
                        plt.subplot(1, 2, 2)
                        plt.imshow(image_overlay_1)
                        plt.subplot(2, 2, 2)
                        plt.imshow(image_overlay_2)
                        plt.show()
                        """

                        del inputs_1, image_1, inputs_2, image_2
                        del embed_1, map_1, embed_2, map_2, fc
                        # ---

                    # img = inputs_PLAIN[i]
                    # pred_img = preds[i]
                    # save_image(img, 'img.png')
                    util.show_images_rgb(imgs_rgb, 3, titles_rgb, pdf)
                    util.show_images_gcam(imgs_gcam_normal, 3, titles_gcam, pdf)
                    util.show_images_gcam(imgs_gcam_metric, 3, titles_gcam, pdf)
                    # print()
                    # loop on closests

                # remember to close the object to ensure writing multiple plots
                pdf.close()
                print()

                # end for x,y

                # confusion matrix
                # CM = confusion_matrix(labelsALL_test, predALL_test)
                CM = confusion_matrix(labelsALL_test, predALL_test_majority)
                CM_perc = CM / dataset_sizes['test']  # perc
                accuracyResult = util.accuracy(CM)
                CM_all = CM_all + CM
                CM_perc_all = CM_perc_all + CM_perc

                # print(output_test)
                util.print_pers("\tConfusion Matrix (%):", fileResultNameFull)
                util.print_pers("\t\t{0}".format(CM_perc * 100), fileResultNameFull)
                util.print_pers("\tAccuracy (%): {0:.2f}".format(accuracyResult * 100), fileResultNameFull)

                # assign
                accuracyALL[r] = accuracyResult

                # newline
                util.print_pers("", fileResultNameFull)

                # save iter
                fileSaveIter = open(os.path.join(dirResult, 'results_{0}.dat'.format(r+1)), 'wb')
                pickle.dump([accuracyResult], fileSaveIter)
                fileSaveIter.close()
                # fileSaveModelIter = open(os.path.join(dirResult, 'model_{0}.dat'.format(r+1)), 'wb')
                # pickle.dump([currentModel], fileSaveModelIter)
                # fileSaveModelIter.close()

                if tsne or gcam:
                    #  create test loader for tsne/gcam
                    # train
                    all_idb2_train = DatasetFromSubset(all_idb2_train_subset,
                                                      transforms.Compose([
                                                          transforms.CenterCrop(256),
                                                          transforms.Resize(imageSize, interpolation=transforms.InterpolationMode.BILINEAR),
                                                          transforms.ToTensor()]))
                    all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                      batch_size=1, shuffle=False,
                                                                      num_workers=numWorkersP, pin_memory=True)
                    # test
                    all_idb2_test = DatasetFromSubset(all_idb2_test_subset,
                                                      transforms.Compose([
                                                          transforms.CenterCrop(256),
                                                          transforms.Resize(imageSize, interpolation=transforms.InterpolationMode.BILINEAR),
                                                          transforms.ToTensor()]))
                    all_idb2_test_loader = torch.utils.data.DataLoader(all_idb2_test,
                                                                      batch_size=1, shuffle=False,
                                                                      num_workers=numWorkersP, pin_memory=True)

                if tsne:
                    # -------------------------------------------------------------- TSNE
                    if log:
                        util.print_pers("Computing T-SNE...", fileResultNameFull)
                    currentModelTSNE = copy.deepcopy(currentModel)
                    # change last layer
                    new_fc = nn.Identity(modelData['sizeFeatures'], classesCNN)
                    # currentModelTSNE.nncf_module.fc = new_fc
                    currentModelTSNE.fc = new_fc
                    currentModelTSNE.to('cpu')
                    dirTSNE = os.path.join(dirResult, 'tsne')
                    os.makedirs(dirTSNE, exist_ok=True)

                    # init
                    featuresALL_train_TSNE = torch.zeros(dataset_sizes['train'], modelData['sizeFeatures'])
                    featuresALL_test_TSNE = torch.zeros(dataset_sizes['test'], modelData['sizeFeatures'])
                    labelsALL_train_TSNE = torch.zeros(dataset_sizes['train'])
                    labelsALL_test_TSNE = torch.zeros(dataset_sizes['train'])
                    outputsALL_test_TSNE = torch.zeros(dataset_sizes['test'])
                    colorsALL_test_TSNE = []

                    # 1. first TSNE: with labels of testing data.
                    # evaluate accuracy of training
                    # (standard tsne)
                    # loop on test data
                    util.print_pers("\t 1st T-SNE...", fileResultNameFull)
                    for img_num, (torch_img, label) in enumerate(all_idb2_test_loader):

                        torch_img.to('cpu')
                        label.to('cpu')

                        if img_num % 10 == 0:
                            print("\t\tImg n. {0} / {1}".format(img_num, int(dataset_sizes['test'])))

                        normed_torch_img = transforms.Normalize([meanNorm, meanNorm, meanNorm], [stdNorm, stdNorm, stdNorm])(torch_img)[None]
                        normed_torch_img = normed_torch_img.squeeze().unsqueeze(0)
                        # tsne
                        features_TSNE, _, _ = currentModelTSNE(normed_torch_img)
                        featuresALL_test_TSNE[img_num, :] = features_TSNE.detach().cpu()

                        # colors
                        if label.item() == 0:
                            labelsALL_test_TSNE[img_num] = 0
                            colorsALL_test_TSNE.append('y')
                        if label.item() == 1:
                            labelsALL_test_TSNE[img_num] = 1
                            colorsALL_test_TSNE.append('b')
                        if label.item() == 2:
                            labelsALL_test_TSNE[img_num] = 2
                            colorsALL_test_TSNE.append('r')
                        if label.item() == 3:
                            labelsALL_test_TSNE[img_num] = 3
                            colorsALL_test_TSNE.append('g')

                    # tsne
                    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(predALL_test_TSNE, labelsALL_test_TSNE)
                    X_embedded = TSNE(n_components=2, init='pca').fit_transform(featuresALL_test_TSNE, labelsALL_test_TSNE)

                    # t-sne for all, standard
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=ms, c=colorsALL_test_TSNE, cmap=plt.cm.Spectral)
                    ax.axis("tight")
                    ax.set_xlabel('Dim. 1')
                    ax.set_ylabel('Dim. 2')
                    legend_elements = [Line2D([0], [0], marker='o', color='r', label='Lymphoblast (1)',
                                              markerfacecolor='r', markersize=5),
                                       Line2D([0], [0], marker='o', color='g', label='Lymphoblast (2)',
                                              markerfacecolor='g', markersize=5),
                                       Line2D([0], [0], marker='o', color='b', label='Normal',
                                              markerfacecolor='b', markersize=5),
                                       Line2D([0], [0], marker='o', color='y', label='Nothing',
                                              markerfacecolor='y', markersize=5),
                                       ]
                    ax.legend(handles=legend_elements, loc='best')
                    # plt.show()
                    fig.savefig(os.path.join(dirTSNE, 'tsne_standard_{0}.png'.format(r+1)))
                    fig.savefig(os.path.join(dirTSNE, 'tsne_standard_{0}.eps'.format(r+1)))
                    plt.close(fig)
                    print()

                    # 2. second TSNE: with labels of training data
                    # and with outputs of testing data
                    # can be used as decision support
                    util.print_pers("\t 2nd T-SNE...", fileResultNameFull)

                    # init
                    featuresALL_train_TSNE = torch.zeros(dataset_sizes['train'], modelData['sizeFeatures'])
                    featuresALL_test_TSNE = torch.zeros(dataset_sizes['test'], modelData['sizeFeatures'])
                    labelsALL_train_TSNE = torch.zeros(dataset_sizes['train'])
                    # labelsALL_test_TSNE = torch.zeros(dataset_sizes['train'])
                    outputsALL_test_TSNE = torch.zeros(dataset_sizes['test'])
                    colorsALL_TSNE = []
                    marker_type_ALL_TSNE = []

                    # loop on train data
                    for img_num, (torch_img, label) in enumerate(all_idb2_train_loader):

                        torch_img.to('cpu')
                        label.to('cpu')

                        if img_num % 10 == 0:
                            print("\t\tTrain Img n. {0} / {1}".format(img_num, int(dataset_sizes['train'])))

                        normed_torch_img = transforms.Normalize([meanNorm, meanNorm, meanNorm], [stdNorm, stdNorm, stdNorm])(torch_img)[None]
                        normed_torch_img = normed_torch_img.squeeze().unsqueeze(0)
                        # tsne
                        features_TSNE, _, _ = currentModelTSNE(normed_torch_img)
                        featuresALL_train_TSNE[img_num, :] = features_TSNE.detach().cpu()

                        # colors
                        if label.item() == 0:
                            labelsALL_train_TSNE[img_num] = 0
                            colorsALL_TSNE.append('y')
                            marker_type_ALL_TSNE.append('o')
                        if label.item() == 1:
                            labelsALL_train_TSNE[img_num] = 1
                            colorsALL_TSNE.append('b')
                            marker_type_ALL_TSNE.append('o')
                        if label.item() == 2:
                            labelsALL_train_TSNE[img_num] = 2
                            colorsALL_TSNE.append('r')
                            marker_type_ALL_TSNE.append('o')
                        if label.item() == 3:
                            labelsALL_train_TSNE[img_num] = 3
                            colorsALL_TSNE.append('g')
                            marker_type_ALL_TSNE.append('o')

                    # loop on test data
                    for img_num, (torch_img, label) in enumerate(all_idb2_test_loader):

                        torch_img.to('cpu')
                        label.to('cpu')

                        if img_num % 10 == 0:
                            print("\t\tTest Img n. {0} / {1}".format(img_num, int(dataset_sizes['test'])))

                        normed_torch_img = transforms.Normalize([meanNorm, meanNorm, meanNorm], [stdNorm, stdNorm, stdNorm])(torch_img)[None]
                        normed_torch_img = normed_torch_img.squeeze().unsqueeze(0)
                        # tsne
                        features_TSNE, _, _ = currentModelTSNE(normed_torch_img)
                        featuresALL_test_TSNE[img_num, :] = features_TSNE.detach().cpu()

                        # colors
                        # if predALL_test[img_num].item() == 0:
                        if predALL_test_majority[img_num].item() == 0:
                            outputsALL_test_TSNE[img_num] = 0
                            colorsALL_TSNE.append('y')
                            marker_type_ALL_TSNE.append('x')
                        # if predALL_test[img_num].item() == 1:
                        if predALL_test_majority[img_num].item() == 1:
                            outputsALL_test_TSNE[img_num] = 1
                            colorsALL_TSNE.append('b')
                            marker_type_ALL_TSNE.append('x')
                        # if predALL_test[img_num].item() == 2:
                        if predALL_test_majority[img_num].item() == 2:
                            outputsALL_test_TSNE[img_num] = 2
                            colorsALL_TSNE.append('r')
                            marker_type_ALL_TSNE.append('x')
                        # if predALL_test[img_num].item() == 3:
                        if predALL_test_majority[img_num].item() == 3:
                            outputsALL_test_TSNE[img_num] = 3
                            colorsALL_TSNE.append('g')
                            marker_type_ALL_TSNE.append('x')

                    # tsne
                    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(predALL_test_TSNE, labelsALL_test_TSNE)
                    X_embedded = TSNE(n_components=2, init='pca').fit_transform(
                        torch.cat((featuresALL_train_TSNE, featuresALL_test_TSNE)),
                        torch.cat((labelsALL_train_TSNE, outputsALL_test_TSNE)))

                    """
                    # t-sne for all
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.scatter(X_embedded[0:dataset_sizes['train'], 0], X_embedded[0:dataset_sizes['train'], 1],
                               c=colorsALL_TSNE[0:dataset_sizes['train']], marker='o', cmap=plt.cm.Spectral)
                    ax.scatter(X_embedded[dataset_sizes['train']:, 0], X_embedded[dataset_sizes['train']:, 1],
                               c=colorsALL_TSNE[dataset_sizes['train']:], marker='x', cmap=plt.cm.Spectral)
                    ax.axis("tight")
                    ax.set_xlabel('Dim. 1')
                    ax.set_ylabel('Dim. 2')
                    legend_elements_1 = [Line2D([0], [0], marker='o', color='r', label='Lymphoblast',
                                              markerfacecolor='r', markersize=5),
                                       Line2D([0], [0], marker='o', color='b', label='Normal',
                                              markerfacecolor='b', markersize=5),
                                       Line2D([0], [0], marker='o', color='y', label='Nothing',
                                              markerfacecolor='y', markersize=5)]
                    legend_elements_2 = [Line2D([0], [0], marker='o', color='k', label='Train',
                                              markerfacecolor='k', markersize=5),
                                         Line2D([0], [0], marker='x', color='k', label='Test',
                                              markerfacecolor='k', markersize=5)]
                    first_legend = plt.legend(handles=legend_elements_1, loc='upper right')
                    ax = plt.gca().add_artist(first_legend)
                    plt.legend(handles=legend_elements_2, loc='lower right')
                    # plt.show()
                    fig.savefig(os.path.join(dirTSNE, 'tsne_train_labels_and_test_output{0}.png'.format(r+1)))
                    # fig.savefig(os.path.join(dirTSNE, 'tsne.eps'))
                    plt.close(fig)
                    print()
                    """

                    # 3. t-sne for each image
                    util.print_pers('Generating T-SNE plots...', fileResultNameFull)
                    # create a PdfPages object
                    pdf = PdfPages(os.path.join(dirTSNE, 'distances_tsne_{}.pdf'.format(r+1)))

                    # loop on images
                    for i in range(0, dataset_sizes['test']):  #i-th sample in testing db
                    # for i in range(0, 3):  # short version
                        # display
                        if i % 10 == 0:
                            print("\tImg n. {0} / {1}".format(i, int(dataset_sizes['test'])))
                        #
                        # create figure
                        fig = plt.figure()
                        ax = fig.add_subplot()
                        # plot 2nd tsne
                        ax.scatter(X_embedded[0:dataset_sizes['train'], 0], X_embedded[0:dataset_sizes['train'], 1],
                                   edgecolors=colorsALL_TSNE[0:dataset_sizes['train']], facecolors='none',
                                   s=ms, marker='o')
                        ax.scatter(X_embedded[dataset_sizes['train']:, 0], X_embedded[dataset_sizes['train']:, 1],
                                   c=colorsALL_TSNE[dataset_sizes['train']:],
                                   s=ms, marker='x')
                        ax.axis("tight")
                        ax.set_xlabel('Dim. 1')
                        ax.set_ylabel('Dim. 2')

                        distances = dm[i]  # row with distances of element i to others
                        closest = np.argsort(distances)
                        # we need the first 'num_closest' elements
                        closest = closest[0:num_closest]

                        filenameImg = os.path.basename(all_idb2.imgs[test_index[i]][0])
                        label_img = labelsALL_test_TSNE[i]
                        output_img = outputsALL_test_TSNE[i]
                        plt.title(filenameImg + ' Output: ' + str(int(output_img)) + '; Label: ' + str(int(label_img)))

                        # black circle
                        ax.scatter(X_embedded[dataset_sizes['train']+i, 0],
                                   X_embedded[dataset_sizes['train']+i, 1],
                                   s=160, marker='s', facecolors='none', edgecolors='k')

                        for close in closest:
                            ax.scatter(X_embedded[close, 0], X_embedded[close, 1],
                                       s=160, facecolors='none', edgecolors='g')

                        # legend
                        legend_elements_1 = [Line2D([0], [0], marker='o', color='r', label='Lymphoblast (1)',
                                                    markerfacecolor='r', markersize=5),
                                             Line2D([0], [0], marker='o', color='g', label='Lymphoblast (2)',
                                                    markerfacecolor='g', markersize=5),
                                             Line2D([0], [0], marker='o', color='b', label='Normal',
                                                    markerfacecolor='b', markersize=5),
                                             Line2D([0], [0], marker='o', color='y', label='Nothing',
                                                    markerfacecolor='y', markersize=5)]
                        legend_elements_2 = [Line2D([0], [0], marker='o', color='k', label='Train',
                                                    markerfacecolor='k', markersize=5),
                                             Line2D([0], [0], marker='x', color='k', label='Test',
                                                    markerfacecolor='k', markersize=5)]
                        legend_elements_3 = [Line2D([0], [0], marker='s', label=filenameImg,
                                                  markerfacecolor='none', markeredgecolor='k'),
                                           Line2D([0], [0], marker='o', label='Closest (in training data)',
                                                  markerfacecolor='none', markeredgecolor='g')]
                        first_legend = plt.legend(handles=legend_elements_1, loc='upper right')
                        ax_1 = plt.gca().add_artist(first_legend)
                        second_legend = plt.legend(handles=legend_elements_2, loc='lower right')
                        ax_2 = plt.gca().add_artist(second_legend)
                        plt.legend(handles=legend_elements_3, loc='upper left')
                        # plt.show()
                        # fig.savefig(os.path.join(dirTSNE, 'tsne_iteration_{0}_img_{1}.png'.format(r + 1, i)))
                        pdf.savefig(fig)
                        plt.close(fig)


                    # close pdf
                    pdf.close()
                    print()

                    #
                    # util.pause()

                    # del
                    if cuda:
                        del currentModelTSNE

                # del
                if cuda:
                    del currentModel
                    del all_idb2_train, all_idb2_train_loader
                    del all_idb2_val, all_idb2_val_loader
                    del all_idb2_test, all_idb2_test_loader
                    del inputs, label
                    del outputs, preds
                    del optimizer_ft, exp_lr_scheduler
                    torch.cuda.empty_cache()

            # end loop on iterations

            # average accuracy
            meanAccuracy = np.mean(accuracyALL)
            stdAccuracy = np.std(accuracyALL)
            meanCM = CM_all / num_iterations
            meanCM_perc = CM_perc_all / num_iterations

            # display
            util.print_pers("", fileResultNameFull)
            util.print_pers("Mean classification accuracy over {0} iterations (%); {1:.2f}".format(num_iterations, meanAccuracy * 100),
                            fileResultNameFull)
            util.print_pers("Std classification accuracy over {0} iterations (%); {1:.2f}".format(num_iterations, stdAccuracy * 100),
                            fileResultNameFull)
            util.print_pers("\tMean Confusion Matrix over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0}".format(meanCM_perc * 100), fileResultNameFull)
            util.print_pers("\tTP (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[1, 1] * 100), fileResultNameFull)
            util.print_pers("\tTN (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[0, 0] * 100), fileResultNameFull)
            util.print_pers("\tFP (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[0, 1] * 100), fileResultNameFull)
            util.print_pers("\tFN (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[1, 0] * 100), fileResultNameFull)

            #close
            fileResult.close()

            # save
            fileSaveFinal = open(os.path.join(dirResult, 'resultsFinal.dat'), 'wb')
            pickle.dump([meanAccuracy], fileSaveFinal)
            fileSaveFinal.close()

            # del
            torch.cuda.empty_cache()
