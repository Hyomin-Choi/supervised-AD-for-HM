import os
import torch
def save_checkpoint(state, checkpoint,filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
def load_model(args=None,models=None):
    for idx in range(len(models)):
        checkpoint = torch.load(os.path.join(args.checkpoint_dir,args.folder_name,\
                                             '{0}_model_{1:04d}.pth.tar'.format(models[idx].__class__.__name__,args.start_epoch)))
        models[idx].load_state_dict((checkpoint['state_dict']))
        #args.start_epoch = checkpoint['epoch']