import argparse
from network import *
from create_data import *
from train import *
from save_or_load import *
from utils import *
from test import *
from attention import *
import wandb
desc = "project anomaly detection for grad_cam"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--flag', type=tuple, default=(False,True), help='train and test')
parser.add_argument('--resume', type=bool, default=True, help='load model')
parser.add_argument('--dataroot', type=str, default='E:\eccvw\GAN_based_Anomaly_Detection\Final_model\\548_500_defect\defect_data_2\\', help='dataset_name')
parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
parser.add_argument('--start_epoch', type=int, default=200, help='start epoch')
parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
parser.add_argument('--train_print_freq', type=int, default=5000, help='The number of image_print_freq') #### 1000
parser.add_argument('--valid_print_freq', type=int, default=1200, help='The number of image_print_freq') #### 1000
parser.add_argument('--save_freq', type=int, default=10, help='The number of ckpt_save_freq') #### 1000
#parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch') ###### 10

parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
parser.add_argument('--L1_loss', type=float, default=1.0, help='Weight about L1_weight')
parser.add_argument('--ABC_loss', type=float, default=1.0, help='Weight about ABC loss')
parser.add_argument('--N_adv_loss_G', type=float, default=1.0, help='Weight about normal adversarial loss for G')
parser.add_argument('--N_adv_loss_D', type=float, default=1.0, help='Weight about normal adverarial loss for D')
parser.add_argument('--Latent_loss', type=float, default=1.0, help='Weight about Latent loss')
# parser.add_argument('--patch_loss', type=float, default=1.5, help='Weight about patch loss')
# parser.add_argument('--AN_adv_loss_G', type=float, default=1.0, help='Weight about anomaly adversarial loss for G')
# parser.add_argument('--AN_adv_loss_D', type=float, default=0.5, help='Weight about anomaly adversarial loss for D')

parser.add_argument('--ch_d', type=int, default=64, help='base channel number per layer') # discriminator channel
parser.add_argument('--ch_g', type=int, default=64, help='base channel number per layer') # generator channel

parser.add_argument('--img_size', type=tuple, default=(128,128), help='The size of image')
parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
parser.add_argument('--k_size', type=int, default=4, help='kernel size')
parser.add_argument('--patch_size', type=int, default=32, help='The size of image patch')   # 1~9
parser.add_argument('--stride', type=int, default=16, help='The size of sliding patch stride size')
parser.add_argument('--error_patch', type=int, default=10, help='The selected number of patch of loss')
# parser.add_argument('--augment_flag', type=str2bool, default=False, help='Image augmentation use or not') #### True

parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                    help='Directory name to save the checkpoints')
parser.add_argument('--result_dir', type=str, default='results',
                    help='Directory name to save the generated images')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='Directory name to save training logs')
parser.add_argument('--sample_dir', type=str, default='samples',
                    help='Directory name to save the samples on training')
parser.add_argument('--valid_dir', type=str, default='valid',
                    help='Directory name to save the samples on validation')
parser.add_argument('--folder_name', type=str, default='all_data',
                    help='Directory name to save the samples on training')

args = parser.parse_args()
hyperparameter = dict_hyperparameter(args)
wandb.init(config=hyperparameter,project=desc,name=0,id='all_data',resume=args.resume)


def main():
    models = create_model(args)
    wandb.watch(models[0])
    wandb.watch(models[1])
    wandb.watch(models[2])
    # wandb.watch(models[3])
    # wandb.watch(models[4])
    if args.resume:
        load_model(args,models=models)
        optimizer, schedular = create_optimier(model=models, args=args)
    if args.flag[0]:
        #train
        train_dataloader = load_data(args, train_flag=True)
        valid_dataloader = load_data(args,valid_flag=True)
        optimizer,schedular = create_optimier(model=models,args=args)
        for epoch in range(args.start_epoch,args.epoch):
             print('\nEpoch: [%d | %d]' % (epoch + 1, args.epoch))
             train_metrics= train(args,models=models,dataloader=train_dataloader,epoch=epoch,optimizer=optimizer)
             test_metrics = train(args,models=models,dataloader=valid_dataloader,epoch=epoch,optimizer=optimizer,train=False)
             train_metrics.update(test_metrics)
             wandb.log(train_metrics)
             del train_metrics,test_metrics
             if epoch % args.save_freq ==0 or epoch == args.epoch-1:
                 for idx in range(len(models)):
                     save_checkpoint({
                         'epoch': epoch + 1,
                         'state_dict': models[idx].state_dict(),
                     }, checkpoint=os.path.join(args.checkpoint_dir, args.folder_name),
                         filename='{0}_model_{1:04d}.pth.tar'.format(models[idx].__class__.__name__,epoch+1))

    if args.flag[1]:
        test_dataloader = load_data(args,test_flag=True)
        test(args,netEncoder=models[0],netDecoder=models[1],netDiscriminator=None,dataloader=test_dataloader)

    wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
if __name__ == '__main__':
    create_folder(args)
    save_hyper_params(args)
    main()