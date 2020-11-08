import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#checkpoint_dir  result_dir sample_dir log_dir
def create_folder(args):
    #checkpoint_dir
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.isdir(args.checkpoint_dir+'/'+args.folder_name):
        os.mkdir(args.checkpoint_dir+'/'+args.folder_name)

    # result_dir
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.isdir(args.result_dir + '/' + args.folder_name):
        os.mkdir(args.result_dir+'/'+args.folder_name)

    #sample_dir
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)
    if not os.path.isdir(args.sample_dir+'/'+args.folder_name):
        os.mkdir(args.sample_dir+'/'+args.folder_name)

    # log_dir
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.isdir(args.log_dir + '/' + args.folder_name):
        os.mkdir(args.log_dir+'/'+args.folder_name)

    # # train_attention_dir
    # if not os.path.isdir(args.train_attention_dir):
    #     os.mkdir(args.train_attention_dir)
    # if not os.path.isdir(args.train_attention_dir + '/' + args.folder_name):
    #     os.mkdir(args.train_attention_dir + '/' + args.folder_name)
    #
    # # test_attention_dir
    # if not os.path.isdir(args.test_attention_dir):
    #     os.mkdir(args.test_attention_dir)
    # if not os.path.isdir(args.test_attention_dir + '/' + args.folder_name):
    #     os.mkdir(args.test_attention_dir + '/' + args.folder_name)



def dict_hyperparameter(args=None):
    params = {}
    for k in args.__dict__:
        params.setdefault(k,args.__dict__[k])
    return params

def create_optimier(model=None,args=None):
    optimizer_Encoder = optim.Adam(model[0].parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_Decoder = optim.Adam(model[1].parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_N_Dis = optim.Adam(model[2].parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optimizer_AN_Dis = optim.Adam(model[3].parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optimizer_classifier = optim.Adam(model[4].parameters(),lr=args.lr,betas=(0.5, 0.999))
    optimizer = [optimizer_Encoder, optimizer_Decoder, optimizer_N_Dis]#, optimizer_AN_Dis,optimizer_classifier]

    scheduler_Encoder = StepLR(optimizer_Encoder, step_size=args.decay_epoch, gamma=0.5)
    scheduler_Decoder = StepLR(optimizer_Decoder, step_size=args.decay_epoch, gamma=0.5)
    scheduler_N_Dis = StepLR(optimizer_N_Dis, step_size=args.decay_epoch, gamma=0.5)
    # scheduler_AN_Dis = StepLR(optimizer_AN_Dis, step_size=args.decay_epoch, gamma=0.5)
    # scheduler_classifier = StepLR(optimizer_classifier, step_size=args.decay_epoch, gamma=0.5)
    scheduler = [scheduler_Encoder, scheduler_Decoder,scheduler_N_Dis]#,scheduler_AN_Dis,scheduler_classifier]

    return optimizer,scheduler

def save_hyper_params(args=None):
    if not os.path.isfile(args.log_dir + '/' + args.folder_name +'/hyper_params.txt'):
        with open(args.log_dir + '/' + args.folder_name +'/hyper_params.txt','w') as f:
            for k in args.__dict__:
                f.write("{0}:{1}".format(k,args.__dict__[k]))
                f.write('\n')

