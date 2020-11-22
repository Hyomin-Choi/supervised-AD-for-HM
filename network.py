import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class encoder(nn.Module):
    def __init__(self,args):
        super(encoder, self).__init__()
        self.down1 = down(args.img_ch,args.ch_g,args.k_size,2,1,bias=False,BN=False)
        self.down2 = down(args.ch_g,args.ch_g*2,args.k_size,2,1,bias=False,BN=True)
        self.down3 = down(args.ch_g*2, args.ch_g * 4, args.k_size, 2,1, bias=False, BN=True)
        self.down4 = down(args.ch_g * 4, args.ch_g * 8, args.k_size, 2,1, bias=False, BN=True)
        # self.down5 = down(args.ch_g * 4, args.ch_g * 4, args.k_size, 2, 1, bias=False, BN=True)
        # self.down6 = down(args.ch_g * 4, args.ch_g * 8, args.k_size, 2, 1, bias=False, BN=True)
    def forward(self,input):
        out1 = self.down1(input) #out1:64X64,64
        out2 = self.down2(out1) # out2: 32x32,128
        out3 = self.down3(out2) # out3:16x16, 256
        out4 = self.down4(out3) #out4: 8x8, 512
        # out5 = self.down5(out4) #out5:8x8, 256
        # out6 = self.down6(out5) #out6:4x4, 512
        return out1,out2,out3,out4 #, out5, out6



class decoder(nn.Module):
    def __init__(self,args):
        super(decoder,self).__init__()
        self.up1 = up(args.ch_g*8,args.ch_g*4,args.k_size,2,1,bias=False,BN=True)
        self.up2 = up(args.ch_g * 8, args.ch_g*2, args.k_size, 2,1, bias=False, BN=True)
        self.up3 = up(args.ch_g * 4, args.ch_g, args.k_size, 2,1, bias=False, BN=True)
        self.up4 = up(args.ch_g * 2, args.img_ch, args.k_size, 2, 1, bias=False, BN=False, last=True)
        # self.up5 = up(args.ch_g * 4, args.ch_g, args.k_size, 2, 1, bias=False, BN=True)
        # self.up6 = up(args.ch_g*2 , args.img_ch, args.k_size, 2,1, bias=False, BN=False,last=True)
    def forward(self,*input):
        out = self.up1(input[3])
        out = torch.cat([out,input[2]],dim=1)

        out = self.up2(out)
        out = torch.cat([out, input[1]],dim=1)

        out = self.up3(out)
        out = torch.cat([out, input[0]],dim=1)

        out = self.up4(out)
        # out = self.up4(out)
        # out = torch.cat([out, input[1]], dim=1)
        #
        # out = self.up5(out)
        # out = torch.cat([out, input[0]], dim=1)
        #
        # out = self.up6(out)
        return out
class down(nn.Module):
    def __init__(self,in_c,out_c,k_size,s_size,p_size,bias=False,BN=True):
        super(down,self).__init__()
        conv = []
        conv.append(nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=k_size,stride=s_size,padding=p_size,bias=False))
        if BN:
            conv.append(nn.BatchNorm2d(out_c))
        conv.append(nn.Tanh())
        # conv.append(nn.LeakyReLU(0.2,True))
        self.conv = nn.Sequential(*conv)

    def forward(self,x):
        return self.conv(x)

class up(nn.Module):
    def __init__(self, in_c, out_c, k_size, s_size, p_size, bias=False, BN=True,last=False):
        super(up,self).__init__()
        deconv = []
        deconv.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=s_size, padding=p_size,
                              bias=False))
        if BN:
            deconv.append(nn.BatchNorm2d(out_c))
        deconv.append(nn.Tanh())
        # if last:
        #     deconv.append(nn.Tanh())
        # else:
        #     deconv.append(nn.ReLU(True))
        self.deconv = nn.Sequential(*deconv)
    def forward(self, x):
        return self.deconv(x)


class N_discriminator(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,args):
        super(N_discriminator, self).__init__()
        model = [nn.Conv2d(in_channels=args.img_ch,out_channels=args.ch_d,kernel_size=4,stride=2,padding=1,bias=False),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=args.ch_d,out_channels=args.ch_d*2,kernel_size=4,stride=2,padding=1,bias=False),
                 nn.BatchNorm2d(args.ch_d*2),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=args.ch_d*2, out_channels=args.ch_d * 4, kernel_size=4, stride=2, padding=1,bias=False),
                 nn.BatchNorm2d(args.ch_d * 4),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=args.ch_d*4,out_channels=args.ch_d*4,kernel_size=4,stride=1,padding=1,bias=False),
                 nn.BatchNorm2d(args.ch_d*4),
                 nn.LeakyReLU(0.2,True),
                 nn.Conv2d(in_channels=args.ch_d*4,out_channels=1,kernel_size=4,stride=1,padding=1,bias=False),
                 nn.Sigmoid()
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)

        return output

class AN_discriminator(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,args):
        super(AN_discriminator, self).__init__()
        model = [nn.Conv2d(in_channels=args.img_ch,out_channels=args.ch_d,kernel_size=4,stride=2,padding=1,bias=False),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=args.ch_d,out_channels=args.ch_d*2,kernel_size=4,stride=2,padding=1,bias=False),
                 nn.BatchNorm2d(args.ch_d*2),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=args.ch_d*2, out_channels=args.ch_d * 4, kernel_size=4, stride=2, padding=1,bias=False),
                 nn.BatchNorm2d(args.ch_d * 4),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=args.ch_d*4,out_channels=args.ch_d*4,kernel_size=4,stride=1,padding=1,bias=False),
                 nn.BatchNorm2d(args.ch_d*4),
                 nn.LeakyReLU(0.2,True),
                 nn.Conv2d(in_channels=args.ch_d*4,out_channels=1,kernel_size=4,stride=1,padding=1,bias=False),
                 nn.Sigmoid()
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)

        return output
class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        self.down1 = down(args.ch_g * 8, args.ch_g * 8, args.k_size, 1, 1, bias=False, BN=True)
        self.down2 = down(args.ch_g * 8, args.ch_g * 16, args.k_size, 1, 1, bias=False, BN=True)
        self.down3 = down(args.ch_g *16,args.ch_g*16,args.k_size,1,1,bias=False,BN=True)
        self.down4 = down(args.ch_g*16,args.ch_g*16,args.k_size,1,1,bias=False,BN=True)
        self.gap = nn.AvgPool2d(kernel_size=(8,8),stride=1)
        self.fc1 = nn.Linear(args.ch_g*16,2,bias=False)

    def forward(self,input):
        output= self.down1(self.pad(input))
        output = self.down2(self.pad(output))
        output = self.down3(self.pad(output))
        output = self.down4(self.pad(output))
        last_conv = output.view(output.shape[1],-1)
        output = self.gap(output)
        output = output.view(output.shape[0],-1)
        output = self.fc1(output)
        output = nn.Softmax(dim=-1)(output)
        return output,last_conv

    def pad(self,x):
        x = F.pad(x,(0,1,0,1))
        return x

class conventional_AE(nn.Module):
    def __init__(self,args):
        super(conventional_AE,self).__init__()
        self.down1 = down(args.img_ch, args.ch_g, args.k_size, 2, 1, bias=False, BN=False)
        self.down2 = down(args.ch_g, args.ch_g * 2, args.k_size, 2, 1, bias=False, BN=True)
        self.down3 = down(args.ch_g * 2, args.ch_g * 4, args.k_size, 2, 1, bias=False, BN=True)
        self.down4 = down(args.ch_g * 4, args.ch_g * 8, args.k_size, 2, 1, bias=False, BN=True)
        self.up1 = up(args.ch_g * 8, args.ch_g * 4, args.k_size, 2, 1, bias=False, BN=True)
        self.up2 = up(args.ch_g * 4, args.ch_g * 2, args.k_size, 2, 1, bias=False, BN=True)
        self.up3 = up(args.ch_g * 2, args.ch_g, args.k_size, 2, 1, bias=False, BN=True)
        self.up4 = up(args.ch_g , args.img_ch, args.k_size, 2, 1, bias=False, BN=False, last=True)
        self.model = nn.Sequential(self.down1,self.down2,self.down3,self.down4,self.up1,self.up2,self.up3,self.up4)
    def forward(self,input):
        output = self.model(input)
        return output


def create_model(args):
    # netEncoder= encoder(args)
    # netDecoder = decoder(args)
    CAE = conventional_AE(args)
    # net_N_Dis = N_discriminator(args)
    # net_AN_Dis = AN_discriminator(args)
    # classifier = Classifier(args)


    # netEncoder = netEncoder.cuda()
    # netDecoder = netDecoder.cuda()
    CAE = CAE.cuda()
    # net_N_Dis = net_N_Dis.cuda()
    # net_AN_Dis = net_AN_Dis.cuda()
    # classifier = classifier.cuda()

    # init_net(netEncoder)
    # init_net(netDecoder)
    init_net(CAE)
    # init_net(net_N_Dis)
    # init_net(net_AN_Dis)
    # init_net(classifier)

    return [CAE] #[netEncoder,netDecoder]#,net_N_Dis]#,net_AN_Dis,classifier]
