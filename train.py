import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import wandb
from tqdm import tqdm
import cv2
import numpy as np
import gc
def train(args=None,models=None,dataloader=None,epoch=None,optimizer=None,train=True): #0:encoder, 1:decoder, 2:normal_D, 3:anomaly_D

    L1 = nn.L1Loss(reduction='mean').cuda()
    MSE = nn.MSELoss(reduction='mean').cuda()
    ce = torch.nn.BCELoss(reduction='mean').cuda()

    loss_name = ['train_L1_loss', 'train_ABC_loss'] if train else ['valid_L1_loss', 'valid_ABC_loss']
                 #'AN_adv_loss_G', 'AN_adv_loss_D','classifier_loss','classifier_acc','attention_loss']
    loss_dict = {loss_name[0]:[],loss_name[1]:[]}#,loss_name[5]:[],loss_name[6]:[],loss_name[7]:[],loss_name[8]:[],loss_name[9]:[],loss_name[10]:[]}
    metrics = {}

    # feature_blobs = [] # (1,1024,8,8)
    # def hook_feature(module, input, output):
    #     feature_blobs.append(output)
    # models[4]._modules.get('down4').register_forward_hook(hook_feature)

    for idx,datas in tqdm(enumerate(dataloader),total=len(dataloader)):
        use_cuda = torch.cuda.is_available()
        data = datas[0].cuda() if use_cuda else datas[0]
        label = datas[1].cuda() if use_cuda else datas[1]
        normal_number = data[label==1].shape[0]
        anomaly_number = data[label==0].shape[0]

        if normal_number !=0:
            #update Discriminator
            encoder_out_N = models[0](data[label == 1])  # out6= 512,4,4
            decoder_out_N = models[1](*encoder_out_N)

            #update Normal data Generator
            difference = L1(data[label==1], decoder_out_N)

            if train:
                loss_dict['train_L1_loss'].append(difference.item())

                N_total_loss = difference * args.L1_loss
                models[0].zero_grad()
                models[1].zero_grad()
                N_total_loss.backward()
                optimizer[0].step()
                optimizer[1].step()
            else:
                loss_dict['valid_L1_loss'].append(difference.item())


        if anomaly_number !=0:
            #update Anomaly data
            encoder_out_AN = models[0](data[label == 0])  # out6= 512,4,4
            decoder_out_AN = models[1](*encoder_out_AN)
            difference = L1(data[label==0], decoder_out_AN)
            difference = -torch.log(1 - torch.exp(-1 * difference)) #0.13밑으로는 내려가지 않음(difference가 0~2라)
            if train:
                loss_dict['train_ABC_loss'].append(difference.item())
                AN_total_loss = difference * args.ABC_loss
                models[0].zero_grad()
                models[1].zero_grad()
                AN_total_loss.backward()
                optimizer[0].step()
                optimizer[1].step()
            else:
                loss_dict['valid_ABC_loss'].append(difference.item())

        # total_loss = N_total_loss + AN_total_loss
        # models[0].zero_grad()
        # models[1].zero_grad()
        # total_loss.backward()
        # optimizer[0].step()
        # optimizer[1].step()
        '''
        #update Discriminator
        if label == 1: #normal
            real_logit = models[2](data)
            fake_logit = models[2](decoder_out) #(1,1,14,14)
            Normal_D_loss =(
                    ce(real_logit,Variable(torch.ones(real_logit.shape).cuda(),requires_grad=False)) +\
                                ce(fake_logit,Variable(torch.zeros(fake_logit.shape).cuda(),requires_grad=False))
            )*0.5
            loss_dict['N_adv_loss_D'].append(Normal_D_loss.item())
            Normal_D_loss =  Normal_D_loss *args.N_adv_loss_D
            models[2].zero_grad()
            Normal_D_loss.backward(retain_graph=True)
            optimizer[2].step()
        
        elif label ==0: #anomaly
            real_logit = models[3](data)
            fake_logit = models[3](decoder_out)  # (1,1,14,14)
            Anomaly_D_loss = (
                    ce(real_logit, Variable(torch.ones(real_logit.shape).cuda(), requires_grad=False)) + \
                                ce(fake_logit, Variable(torch.zeros(fake_logit.shape).cuda(), requires_grad=False))
            ) * 0.5
            loss_dict['AN_adv_loss_D'].append(Anomaly_D_loss.item())
            Anomaly_D_loss = Anomaly_D_loss * args.AN_adv_loss_D
            models[3].zero_grad()
            Anomaly_D_loss.backward(retain_graph=True)
            optimizer[3].step()
        
        #update Generator
        difference = L1(data,decoder_out)
        if label == 1: #normal
            recon_latent = models[0](decoder_out)[3]
            latent_difference = L1(encoder_out[3], recon_latent)
            patch_loss = compute_patch(real=data,fake=decoder_out,args=args)
            fake_logit = models[2](decoder_out)
            Normal_G_loss = ce(fake_logit, Variable(torch.ones(fake_logit.shape).cuda(), requires_grad=False))

            loss_dict['L1_loss'].append(difference.item())
            loss_dict['Latent_loss'].append(latent_difference.item())
            loss_dict['N_adv_loss_G'].append(Normal_G_loss.item())
            loss_dict['patch_loss'].append(patch_loss.item())
            N_total_loss = difference * args.L1_loss + latent_difference * args.Latent_loss + patch_loss * args.patch_loss + Normal_G_loss * args.N_adv_loss_G

            models[0].zero_grad()
            models[1].zero_grad()
            N_total_loss.backward()
            optimizer[0].step()
            optimizer[1].step()
        
        elif label == 0:#anomaly
            difference = -torch.log(1 - torch.exp(-1 * difference))
            fake_logit = models[3](decoder_out)
            Anomaly_G_loss = ce(fake_logit, Variable(torch.zeros(fake_logit.shape).cuda(), requires_grad=False))

            loss_dict['ABC_loss'].append(difference.item())
            loss_dict['AN_adv_loss_G'].append(Anomaly_G_loss.item())
            AN_total_loss = difference * args.ABC_loss + Anomaly_G_loss*args.AN_adv_loss_G
            models[0].zero_grad()
            models[1].zero_grad()
            AN_total_loss.backward()
            optimizer[0].step()
            optimizer[1].step()
        
        # encoder_out = models[0](data)
        # classifier_out = models[4](encoder_out[-1].detach())
        # pred = torch.argmax(classifier_out[0], dim=1)
        # target = Variable(torch.zeros(classifier_out[0].shape).cuda(), requires_grad=False)
        # target[:, label] = 1.0
        # classifier_loss = ce(classifier_out[0], target)
        #
        # weight_softmax = list(models[4].parameters())[-1]
        #
        # cam = torch.matmul(weight_softmax, classifier_out[1])
        # N_cam = torch.sigmoid(cam[1, :].view(8, 8))
        # AN_cam = torch.sigmoid(cam[0, :].view(8, 8))#nn.functional.sigmoid
        # ones = Variable(torch.ones(N_cam.shape).cuda(), requires_grad=False)
        # attention_loss = torch.mean(ones - N_cam + AN_cam) *2
        #
        # loss_dict['classifier_loss'].append(classifier_loss.item())
        # loss_dict['attention_loss'].append(attention_loss.item())
        # loss_dict['classifier_acc'].append(100) if pred == label else loss_dict['classifier_acc'].append(0)
        #
        # models[4].zero_grad()
        #
        # if label == 1:
        #     classifier_loss.backward(retain_graph=True)
        #     attention_loss.backward()
        # else:
        #     classifier_loss.backward()
        # optimizer[4].step()
        '''
        if train:
            if idx % args.train_print_freq == 0:
                # real_fake_attention = create_cam(real=data.cpu(),fake=decoder_out.detach().cpu(),cam=cam.detach().cpu(),epoch=epoch,idx=idx)

                if normal_number !=0:
                    real_fake_N = torch.cat((data[label==1],decoder_out_N),dim=0)
                    save_image(real_fake_N, os.path.join(args.sample_dir,args.folder_name,
                                                    '{0:04d}_{1:03d}_normal.png'.format(epoch, idx)),normalize=True)
                if anomaly_number !=0:
                    real_fake_AN = torch.cat((data[label == 0], decoder_out_AN), dim=0)
                    save_image(real_fake_AN, os.path.join(args.sample_dir, args.folder_name,
                                                       '{0:04d}_{1:03d}_anomaly.png'.format(epoch, idx)), normalize=True)
        else:
            if idx % args.valid_print_freq == 0:
                # real_fake_attention = create_cam(real=data.cpu(),fake=decoder_out.detach().cpu(),cam=cam.detach().cpu(),epoch=epoch,idx=idx)

                if normal_number != 0:
                    real_fake_N = torch.cat((data[label == 1], decoder_out_N), dim=0)
                    save_image(real_fake_N, os.path.join(args.valid_dir, args.folder_name,
                                                         '{0:04d}_{1:03d}_normal.png'.format(epoch, idx)), normalize=True)
                if anomaly_number != 0:
                    real_fake_AN = torch.cat((data[label == 0], decoder_out_AN), dim=0)
                    save_image(real_fake_AN, os.path.join(args.valid_dir, args.folder_name,
                                                          '{0:04d}_{1:03d}_anomaly.png'.format(epoch, idx)), normalize=True)

        for n in loss_name:
            try:
                tmp = sum(loss_dict[n]) / len(loss_dict[n])
                loss_dict[n] = []
                loss_dict[n].append(tmp)
            except:
                pass
        torch.cuda.empty_cache()
    for n in loss_name:
        try:
            avg = sum(loss_dict[n]) / len(loss_dict[n])
            metrics[n] = avg
        except:
            pass
    return metrics

def compute_patch(real=None,fake=None,args=None):
    N_error = []
    L1 = nn.L1Loss(reduction='mean').cuda()
    for i in range(((args.img_size[0] - args.patch_size) // args.stride) + 1):  # ex) Output = ((Input - Kernel + (2*padding)) / S) + 1
        for j in range(((args.img_size[1] - args.patch_size) // args.stride) + 1):
            a = real[:,:, i * args.stride: (i * args.stride) + args.patch_size,
                                j * args.stride: (j * args.stride) + args.patch_size]
            real_N_patch = real[:,:, i * args.stride: (i * args.stride) + args.patch_size,
                                j * args.stride: (j * args.stride) + args.patch_size]
            reconstructed_N_patch = fake[:,:, i * args.stride: (i * args.stride) + args.patch_size,
                                         j * args.stride: (j * args.stride) + args.patch_size]
            reconstructed_N_patch_loss = L1(real_N_patch, reconstructed_N_patch)
            N_error.append(reconstructed_N_patch_loss)
    N_error = torch.tensor(N_error)
    N_error = torch.sort(N_error,descending=True)
    error_patchs = N_error.values[:args.error_patch]
    mean = error_patchs.mean()
    return mean.cuda()

def create_cam(real=None,fake=None,cam=None,epoch=None,idx=None):
    real = real.numpy().squeeze(0).transpose(1,2,0)
    fake = fake.numpy().squeeze(0).transpose(1,2,0)
    cam = cam.numpy()
    cam = cam.reshape(-1, 8, 8)

    N_cam = cam[1,:,:]
    AN_cam = cam[0,:,:]
    N_cam = N_cam - np.min(N_cam)
    N_cam = N_cam / np.max(N_cam)
    AN_cam = AN_cam - np.min(AN_cam)
    AN_cam = AN_cam / np.max(AN_cam)
    real = real - np.min(real)
    real = real / np.max(real)
    fake = fake - np.min(fake)
    fake = fake / np.max(fake)

    N_cam = np.uint8(255 * N_cam)
    AN_cam = np.uint8(255 * AN_cam)
    real = np.uint8(255*real)
    fake = np.uint8(255 * fake)

    N_cam = cv2.resize(N_cam, (128, 128))
    N_cam = cv2.applyColorMap(cv2.resize(N_cam, (128, 128)), cv2.COLORMAP_JET)
    N_cam = N_cam[:,:,::-1]
    AN_cam = cv2.resize(AN_cam, (128, 128))
    AN_cam = cv2.applyColorMap(cv2.resize(AN_cam, (128, 128)), cv2.COLORMAP_JET)
    AN_cam = AN_cam[:, :, ::-1]

    N_img = N_cam *0.3 + real*0.5
    AN_img = AN_cam * 0.3 +real*0.5

    # cv2.imwrite('N_cam{}_{}.png'.format(epoch,idx), N_img)
    # cv2.imwrite('AN_cam{}_{}.png'.format(epoch,idx), AN_img)

    # N_img = N_img[:,:,::-1].copy()
    # AN_img = AN_img[:,:,::-1].copy()


    N_img = N_img.transpose(2,0,1)[np.newaxis,:]
    AN_img = AN_img.transpose(2,0,1)[np.newaxis,:]
    real = real.transpose(2, 0, 1)[np.newaxis, :]
    fake = fake.transpose(2, 0, 1)[np.newaxis, :]

    N_img = torch.tensor(N_img)
    AN_img = torch.tensor(AN_img)
    real = torch.tensor(real)
    fake = torch.tensor(fake)

    return torch.cat([real,fake, N_img, AN_img], dim=0)


