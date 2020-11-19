import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt

def test(args=None,models=None,dataloader=None):
    with torch.no_grad():
        total_L1_loss = []
        L1 = nn.L1Loss(reduction='mean').cuda()
        MSE = nn.MSELoss(reduction='mean').cuda()
        difference_list = []
        N_difference_list = []
        AN_difference_list = []
        label_list =[]
        # anomaly_count = dataloader.sampler.target.count(0)
        # normal_count = dataloader.sampler.target.count(1)
        for idx,datas in enumerate(dataloader):
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                data = datas[0].cuda()
                label = datas[1].cuda()
            CAE_out = models[0](data)
            # encoder_out = netEncoder(data)  # out4= 512,16,16
            # decoder_out = netDecoder(*encoder_out)
            difference = L1(data, CAE_out)

            difference_list.append(difference.item())
            if label ==0:
                label_list.append(1)
                AN_difference_list.append(difference)
            else:
                label_list.append(0)
                N_difference_list.append(difference)
            real_fake = torch.cat([data, CAE_out], dim=0)
            if label == 1:
                save_image(real_fake, os.path.join(args.result_dir, args.folder_name,
                                                   '{0:04d}_{1:03d}_normal.png'.format(args.start_epoch, idx)), normalize=True)
            else:
                save_image(real_fake, os.path.join(args.result_dir, args.folder_name,
                                                   '{0:04d}_{1:03d}_anomaly.png'.format(args.start_epoch, idx)),normalize=True)
        print(difference_list[-1])
        label = np.array(label_list)
        pred = np.array(difference_list)
        fpr1, tpr1, thresholds1 = roc_curve(label, pred, drop_intermediate=False)
        print('auc:{}'.format(auc(fpr1, tpr1)))
        plt.plot(fpr1, tpr1, 'o-', ms=2, label="AUC:{}".format(auc(fpr1, tpr1)))
        plt.legend()
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        plt.xlabel('False Positive rate(100-Specificity)')
        plt.ylabel('True Positive rate(Sensitivity')
        plt.title('test ROC Curve')

        plt.savefig(os.path.join(args.result_dir, args.folder_name) + '/{}_ROC_curve.png'.format(args.start_epoch))
        plt.show()
        # plt.xlim(0,2)
        sns.set_style("darkgrid")
        sns.distplot(AN_difference_list,label='Abnormal Scores', rug=False,hist=False)
        sns.distplot(N_difference_list, label='Normal Scores',  rug=False,hist=False)
        # sns.kdeplot(AN_difference_list,label = 'Abnormal Scores',shade=True)#,shade=True)
        # sns.kdeplot(N_difference_list, label='Normal Scores', shade=True)
        plt.legend()
        plt.savefig(os.path.join(args.result_dir, args.folder_name) + '/{}_distplot.png'.format(args.start_epoch))
        plt.show()






'''
file_anormal.close()
normal_label = np.zeros(len(normal))
anormal_label = np.ones(len(anormal))
normal = np.array(normal)
anormal = np.array(anormal)
label = np.concatenate((normal_label,anormal_label))
pred = np.concatenate((normal,anormal))


fpr1, tpr1, thresholds1 = roc_curve(label,pred,drop_intermediate=False)
print('auc:{}'.format(auc(fpr1, tpr1)))
plt.plot(fpr1, tpr1, 'o-', ms=2, label="AUC:{}".format(auc(fpr1,tpr1)))
plt.legend()
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive rate(100-Specificity)')
plt.ylabel('True Positive rate(Sensitivity')
plt.title('test ROC Curve')
plt.show()
'''