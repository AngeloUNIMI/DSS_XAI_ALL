import os,sys
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Explanation_generator:

    def __init__(self):
        self.Decomposition = None

    # read image, subtract bias, convert to rgb for imshow
    def read(self, path, mean, std):
        image = cv2.imread(path).astype(float)
        image_show = image[:,:,::-1]/255.
        image[:, :, 0] = (image[:, :, 0] - mean*255) / (std*255)
        image[:, :, 1] = (image[:, :, 1] - mean*255) / (std*255)
        image[:, :, 2] = (image[:, :, 2] - mean*255) / (std*255)
        return image, image_show

    # if you wanna load data from CUB or car dataset, you may download the dataset and write a simple dataloader.
    def get_input_from_path(self, path_1, path_2, size, mean, std):
        '''
            load two images from paths
        '''
        inputs_1, image_1 = self.read(path_1, mean, std)
        inputs_2, image_2 = self.read(path_2, mean, std)

        image_1 = cv2.resize(image_1, (size[1], size[0]))
        image_2 = cv2.resize(image_2, (size[1], size[0]))

        inputs_1 = cv2.resize(inputs_1, (size[1], size[0]))
        inputs_2 = cv2.resize(inputs_2, (size[1], size[0]))

        inputs_1 = np.transpose(inputs_1, (2, 0, 1))
        inputs_2 = np.transpose(inputs_2, (2, 0, 1))

        # wrap them in Variable
        inputs_1 = Variable(torch.from_numpy(np.expand_dims(inputs_1.astype(np.float32), axis=0))).cuda()
        inputs_2 = Variable(torch.from_numpy(np.expand_dims(inputs_2.astype(np.float32), axis=0))).cuda()

        return inputs_1, image_1, inputs_2, image_2

    def imshow_convert(self, raw):
        '''
            convert the heatmap for imshow
        '''
        heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET))
        return heatmap

    def GradCAM(self, map, size=(224, 224)):
        # gradient = map.grad.cpu().numpy()
        # map = map.detach().cpu().numpy()
        gradient = map.grad.numpy()
        map = map.detach().numpy()

        # compute the average value
        weights = np.mean(gradient[0], axis=(1, 2), keepdims=True)
        grad_CAM_map = np.sum(np.tile(weights, [1, map.shape[-2], map.shape[-1]]) * map[0], axis=0)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1], size[0]))
        return cam

    def RGradCAM(self, map, size = (224, 224)):
        # rectified Grad-CAM, one variant
        # gradient = map.grad.cpu().numpy()
        # map = map.detach().cpu().numpy()
        gradient = map.grad.numpy()
        map = map.detach().numpy()

        # remove the heuristic GAP step
        weights = gradient[0]
        grad_CAM_map = np.sum(weights * map[0], axis = 0)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1], size[0]))
        return cam

    def Overall_map(self, map_1, map_2, fc_1 = None, fc_2 = None, size = (224, 224), mode = 'GMP'):
        '''
            Only for GMP architecture, you may check the code of other applications
            for the implementation of GAP and flattened feature.
        '''
        if mode == 'GMP':
            # map_1 = np.transpose(map_1.detach().cpu().numpy(),(0,2,3,1))
            # map_2 = np.transpose(map_2.detach().cpu().numpy(),(0,2,3,1))
            map_1 = np.transpose(map_1.detach().numpy(),(0,2,3,1))
            map_2 = np.transpose(map_2.detach().numpy(),(0,2,3,1))

            # generate the maximum mask
            for k in range(map_1.shape[-1]):
                map_1[:,:,:,k] = map_1[:,:,:,k] * (map_1[:,:,:,k] == np.max(map_1[:,:,:,k]))
            for k in range(map_2.shape[-1]):
                map_2[:,:,:,k] = map_2[:,:,:,k] * (map_2[:,:,:,k] == np.max(map_2[:,:,:,k]))

            map_1_reshape = np.reshape(map_1, [-1, map_1.shape[-1]])
            map_2_reshape = np.reshape(map_2, [-1, map_2.shape[-1]])

            # compute the equivalent feature for each position
            map_1_embed = np.matmul(map_1_reshape, np.transpose(fc_1.weight.data.numpy())) #+ fc_1.bias.data.numpy() / map_1_reshape.shape[0]
            map_2_embed = np.matmul(map_2_reshape, np.transpose(fc_2.weight.data.numpy())) #+ fc_2.bias.data.numpy() / map_2_reshape.shape[0]
            map_1_embed = np.reshape(map_1_embed, [map_1.shape[1], map_1.shape[2],-1])
            map_2_embed = np.reshape(map_2_embed, [map_2.shape[1], map_2.shape[2],-1])

            Decomposition = np.zeros([map_1.shape[1],map_1.shape[2],map_2.shape[1],map_2.shape[2]])
            for i in range(map_1.shape[1]):
                for j in range(map_1.shape[2]):
                    for x in range(map_2.shape[1]):
                        for y in range(map_2.shape[2]):
                            Decomposition[i,j,x,y] = np.sum(map_1_embed[i,j]*map_2_embed[x,y])
            Decomposition = Decomposition / np.max(Decomposition)
            Decomposition = np.maximum(Decomposition, 0)
            return Decomposition

    def get_embed(self, model, inputs_1, inputs_2):

        device = 'cpu'

        model = model.to(device)
        model = model.eval()

        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)

        embed_1, map_1, _ = model(inputs_1)
        embed_2, map_2, _ = model(inputs_2)

        """
        embed_1 = embed_1.to('cpu')
        map_1 = map_1.to('cpu')
        embed_2 = embed_2.to('cpu')
        map_2 = map_2.to('cpu')
        """

        fc = model.fc
        fc = fc.cpu()

        return embed_1, map_1, embed_2, map_2, fc
