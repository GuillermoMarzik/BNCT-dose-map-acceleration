import torch

dataset = {
        'name' : 'full_104_106',
        'sim_path' : './data/full_104_106',
        'ct_path' : './data/ct',
        'summary_path': './data/full_104_106/summary',
        'shuffle' : True,
        'tally' : -1,
        } 

training = {
        'num_epochs' : 100,
        'lr' : 0.0002,
        'lr_scheduler': False,
        'lr_half' : 10,
        'beta1' : 0.5,
        'save_period': 25,
        'batch_size': 32,
        'validation_split': 0.7,
        'loss': 'MSE_pond',
        }

network = {
        'num_layers' : 3, # number of layers for constant dimension blocks
        'mod_type' : 'Unet_skip',
        'input_shape' : 14,
        'output_shape' : 5,
        }

model_name = 'unet_tcia_104_106_lossMSEpond_modSkip_dosiscomp'
save_path = './models'
logs_path = './logs'
predictions_path = './predictions'
sim_input = True
upstream = True
glob = False

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# TODO: fix to handle more than one GPU scenario
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
