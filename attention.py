from tqdm import tqdm
import torch
import numpy as np
import cv2
import torch.nn as nn
from torchvision.utils import save_image
import os
import torch.optim as optim
from torch.autograd import Variable

def attention(models=None,args=None,dataloader=None,train=False):


    if train:
        # for n, layer in models[4].named_children():
        #     # layer.required_grad = False
        #     if n != 'fc1':
        #         for p in layer.parameters():
        #             p.required_grad = False
        fc_optim = optim.Adam(models[4].parameters(), lr=args.lr, betas=(0.5, 0.999))
        for idx,datas in tqdm(enumerate(dataloader),total=len(dataloader)):
            use_cuda = torch.cuda.is_available()
            data = datas[0].cuda() if use_cuda else datas[0]
            label = datas[1].cuda() if use_cuda else datas[1]
            encoder_out = models[0](data)  # out6= 512,4,4
            finalconv_name = 'down4'
            # hook
            feature_blobs = [] #(1,1024,8,8)

            def hook_feature(module, input, output):
                feature_blobs.append(output)

            models[4]._modules.get(finalconv_name).register_forward_hook(hook_feature)
            weight_softmax = list(models[4].parameters())[-1] #2x1024

            classifier_out = models[4](encoder_out[-1])
            feature_blobs = feature_blobs[0].view(feature_blobs[0].size(1),-1)
            cam = torch.matmul(weight_softmax,feature_blobs)
            N_cam = cam[1,:].view(8,8)
            AN_cam = cam[0,:].view(8,8)
            ones = Variable(torch.zeros(N_cam.shape).cuda(),requires_grad=False)
            l = torch.mean(ones-N_cam+AN_cam)
            models[4].zero_grad()
            l.backward()
            print(3)



            if label == 1:
                save_image(save_img, os.path.join(args.train_attention_dir, args.folder_name,
                                                   '{0:04d}_{1:03d}_normal.png'.format(args.start_epoch, idx)), normalize=True)
            else:
                save_image(save_img, os.path.join(args.train_attention_dir, args.folder_name,
                                                   '{0:04d}_{1:03d}_anomaly.png'.format(args.start_epoch, idx)), normalize=True)
    else:
        for idx,datas in tqdm(enumerate(dataloader),total=len(dataloader)):
            use_cuda = torch.cuda.is_available()
            data = datas[0].cuda() if use_cuda else datas[0]
            label = datas[1].cuda() if use_cuda else datas[1]
            encoder_out = models[0](data)  # out6= 512,4,4
            finalconv_name = 'down4'
            # hook
            feature_blobs = [] #(1,1024,8,8)

            def hook_feature(module, input, output):
                feature_blobs.append(output)

            models[4]._modules.get(finalconv_name).register_forward_hook(hook_feature)
            weight_softmax = list(models[4].parameters())[-1] #2x1024

            classifier_out = models[4](encoder_out[-1])
            feature_blobs = feature_blobs[0].view(feature_blobs[0].size(1),-1)
            cam = torch.matmul(weight_softmax,feature_blobs).cpu().data.numpy()
            save_img = create_cam(data = data.cpu(),cam=cam)
            if label == 1:
                save_image(save_img, os.path.join(args.train_attention_dir, args.folder_name,
                                                   '{0:04d}_{1:03d}_normal.png'.format(args.start_epoch, idx)), normalize=True)
            else:
                save_image(save_img, os.path.join(args.train_attention_dir, args.folder_name,
                                                   '{0:04d}_{1:03d}_anomaly.png'.format(args.start_epoch, idx)), normalize=True)

def create_cam(data=None,cam=None):
    img = data.numpy().squeeze(0).transpose(1,2,0)
    cam = cam.reshape(-1, 8, 8)

    N_cam = cam[1,:,:]
    AN_cam = cam[0,:,:]
    N_cam = N_cam - np.min(N_cam)
    N_cam = N_cam / np.max(N_cam)
    AN_cam = AN_cam - np.min(AN_cam)
    AN_cam = AN_cam / np.max(AN_cam)
    img = img - np.min(img)
    img = img / np.max(img)

    N_cam = np.uint8(255 * N_cam)
    AN_cam = np.uint8(255 * AN_cam)
    img = np.uint8(255*img)

    N_cam = cv2.resize(N_cam, (128, 128))
    N_cam = cv2.applyColorMap(cv2.resize(N_cam, (128, 128)), cv2.COLORMAP_JET)
    AN_cam = cv2.resize(AN_cam, (128, 128))
    AN_cam = cv2.applyColorMap(cv2.resize(AN_cam, (128, 128)), cv2.COLORMAP_JET)

    N_img = N_cam *0.3 + img*0.5
    AN_img = AN_cam * 0.3 +img*0.5

    N_img = N_img.transpose(2,0,1)[np.newaxis,:]
    AN_img = AN_img.transpose(2,0,1)[np.newaxis,:]
    img = img.transpose(2, 0, 1)[np.newaxis, :]

    N_img = torch.tensor(N_img)
    AN_img = torch.tensor(AN_img)
    img = torch.tensor(img)

    return torch.cat([img, N_img, AN_img], dim=0)





