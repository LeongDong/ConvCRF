import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import ceil, sqrt

class ConvCRF(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_size = 3
        self.iterations = 10
        self.ratio = 0.5
        self.shift = self.kernel_size // 2
        self.modes = ['combine']#['position','combine']
        self.channels = [3]#[2, 3]
        self.thetas = [self.register_parameter('{}_theta'.format(mode), nn.Parameter(torch.ones(1,c,1,1).cuda()*1.)) for c, mode in zip(self.channels, self.modes)]
        self.weights = [self.register_parameter('{}_weight'.format(mode), nn.Parameter(torch.Tensor([1.]).cuda())) for mode in self.modes]

    def mesh(self, image_size):
        '''
        image_size:(W,H)
        return: meshgrid with size (2,W,H)
        (0,0) (0,1) ... (0, N-1)
        (1,0) (1,1) ... (1, N-1)
        .
        .
        (M-1,0) (M-1,1)...(M-1,N-1)
        '''
        coord_range = [range(s) for s in image_size]
        mesh = np.array(np.meshgrid(*coord_range, indexing='ij'),dtype=np.float32)
        mesh = torch.from_numpy(mesh).cuda()
        return mesh
    def features(self, img, mode):
        '''
        img: gray image
        mode: combine(gray image with position, 3 channels);
              position(Only position, 2 channels)
        :return: feature map with 2 or 3 channels
        '''
        if mode == 'combine':
            return torch.cat([torch.stack(img.shape[0] * [self.mesh(img.shape[-2:None])]),img * 255],dim=1)
            #cat[(batchsize,2,W,H),(batchsize,1,H,W)]=>(batchsize,3,H,W)
        elif mode == 'position':
            return torch.stack(img.shape[0] * [self.mesh(img.shape[-2:None])])
            #batchsize*(2,H,W)=>(batchsize,2,H,W)
        else:
            print("Choose mode from 'combine' or 'position'")
            return None

    def convFilter(self,feature,image_size,theta):
        '''
        feature: position,(Batch_size,2,H,W) or combine, (Batch_size,3,H,W)
        theta: Gaussian parameter for adjusting the scaling among features
        return: Gaussian convolutional filter (Batch_size,1,kernel height, kernel width,H,W)
        '''
        gaussian_filter = feature.data.new(feature.shape[0],self.kernel_size,self.kernel_size,*image_size).fill_(0).cuda()

        def _get_ind(dz):
            if(dz < 0):
                return 0, -dz
            elif(dz > 0):
                return dz, 0
            else:
                return 0, 0

        def _negative(dz):
            if(dz == 0):
                return None
            else:
                return -dz

        for dx in range(-self.shift, self.shift + 1):
            for dy in range(-self.shift, self.shift + 1):
                dx1, dx2 = _get_ind(dx)
                dy1, dy2 = _get_ind(dy)
                feature_1 = feature[:,:,dx1:_negative(dx2),dy1:_negative(dy2)]
                feature_2 = feature[:,:,dx2:_negative(dx1),dy2:_negative(dy1)]

                difference = (feature_1 - feature_2) ** 2
                expon = torch.exp(-0.5 * torch.sum(difference * theta, dim=1))

                gaussian_filter[:,dx + self.shift, dy + self.shift, dx2:_negative(dx1), dy2:_negative(dy1)] = expon

        gaussian_filter = gaussian_filter.view(feature.shape[0],1,self.kernel_size,self.kernel_size,*image_size)
        return gaussian_filter #gaussian_filter (Batch_size,1,kernel_size,kernel_size,H,W)
        '''
        gaussian_filter[:,0,0,1:None,1:None] = torch.exp(-0.5*torch.sum((feature[:,:,1:None,1:None] - feature[:,:,0:-1,0:-1])**2*theta,dim=1))
        gaussian_filter[:,0,1,1:None,0:None] = torch.exp(-0.5*torch.sum((feature[:,:,1:None,0:None] - feature[:,:,1:None,1:None])**2*theta,dim=1))
        gaussian_filter[:,0,2,1:None,0:-1] = torch.exp(-0.5*torch.sum((feature[:,:,1:None,0:-1] - feature[:,:,0:-1,1:None])**2*theta,dim=1))
        gaussian_filter[:,1,0,0:None,1:None] = torch.exp(-0.5*torch.sum((feature[:,:,0:None,1:None] - feature[:,:,0:None,0:-1])**2*theta,dim=1))
        gaussian_filter[:,1,1,0:None,0:None] = torch.exp(-0.5*torch.sum((feature[:,:,0:None,0:None] - feature[:,:,0:None,0:None])**2*theta,dim=1))
        gaussian_filter[:,1,2,0:None,0:-1] = torch.exp(-0.5*torch.sum((feature[:,:,0:None,0:-1] - feature[:,:,0:None,1:None])**2*theta,dim=1))
        gaussian_filter[:,2,0,0:-1,1:None] = torch.exp(-0.5*torch.sum((feature[:,:,1:None,0:-1] - feature[:,:,0:-1,1:None])**2*theta,dim=1))
        gaussian_filter[:,2,1,0:-1,0:None] = torch.exp(-0.5*torch.sum((feature[:,:,1:None,0:None] - feature[:,:,0:-1,0:None])**2*theta,dim=1))
        gaussian_filter[:,2,2,0:-1,0:-1] = torch.exp(-0.5*torch.sum((feature[:,:,1:None,1:None] - feature[:,:,0:-1,0:-1])))
        '''

    def convCRF(self, prediction, gaussian_filter):
        '''

        prediction: The output of segmentation network (Batchsize, c=1, H, W)
        gaussian_filter: Gaussian Convolutional Filter (Batchsize,1,kernel_size,kernel_size,H,W)
        return Convolutional Result (After Message Passing) (Batchsize,1,H,W)
        '''
        Batchsize, c, H, W = prediction.shape
        prediction_unfold = F.unfold(input=prediction,kernel_size=self.kernel_size,dilation=1,padding=self.shift,stride=1)
        #prediction_unfold: (Batchsize,c*kernel_size*kernel_size,H*W) The channel of prediction is equal to the channel of gaussian filter (is 1).
        prediction_unfold = prediction_unfold.view(Batchsize,c,self.kernel_size,self.kernel_size,H,W)

        product = prediction_unfold * gaussian_filter
        product = product.view(Batchsize,c,self.kernel_size * self.kernel_size,H,W)
        #product: (Batchsize,c,kernelsize^2,H,W)
        mespass = torch.sum(product,dim=2)

        return mespass #mespass:(Batchsize,c,H,W)

    def forward(self, image, unary):
        '''

        image: Batchsize*1*H*W
        unary: Batchsize*1*H*W
        return Final Prediction Map: Batchsize*1*H*W
        '''
        image_size = image.shape[-2:None]
        batch_size = image.shape[0]
        norm_gaussian_kernels = []

        # def maxTensor(tensorA, tensorB):
        #     tensorMid = tensorA - tensorB
        #     return torch.div((torch.abs(tensorMid) - tensorMid), 2) + tensorA

        feats = [self.features(image,mode=mode) for mode in self.modes]
        gaussian_kernels = [self.convFilter(feat, image_size, eval('self.{}_theta'.format(mode))) for i,(feat,mode) in enumerate(zip(feats,self.modes))]
        for i, gaussian_kernel in enumerate(gaussian_kernels):
            gaussian_kernel = gaussian_kernel.view(batch_size, 1, self.kernel_size * self.kernel_size, *image_size)
            gaussian_kernel = F.softmax(gaussian_kernel, dim=2)
            gaussian_kernel = gaussian_kernel.view(batch_size, 1, self.kernel_size, self.kernel_size, *image_size)
            norm_gaussian_kernels.append(gaussian_kernel)

        prediction = unary
        for iter in range(self.iterations):
            message = torch.zeros(batch_size,1,*image_size).cuda()
            for i, (kernel, mode) in enumerate(zip(norm_gaussian_kernels,self.modes)):
                iter_message = self.convCRF(prediction,kernel)
                message = message + (iter_message * eval('self.{}_weight'.format(mode)))

            prediction = self.ratio * unary + (1 - self.ratio) * message
            #prediction = maxTensor(prediction, unary)

        return prediction