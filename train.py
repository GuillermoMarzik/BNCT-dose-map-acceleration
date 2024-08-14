import sys

sys.path.insert(1, './configs')
sys.path.insert(1, './src')

import importlib
import argparse
import model 
import torch
import os
import utils as u
from trainer import Trainer

argParser = argparse.ArgumentParser()
argParser.add_argument("config_name", help="Name of the loaded configuration file")
args = argParser.parse_args()
config_name = list(vars(args).values())[0]
c = importlib.import_module(config_name)

if c.upstream == True or c.glob == True:
    from trainer import Trainer
    import utils as u
else:
    from trainer_dose import Trainer
    import map_utils as u


## Model params
# Number of consecutive same dimension layers
num_layers = c.network['num_layers']
mod_type = c.network['mod_type']
if mod_type == 'Unet':
    mod = model.Unet(num_layers, c.network['input_shape'], c.network['output_shape'])
elif mod_type == 'Unet_skip':
    mod = model.Unet_skip(num_layers, c.network['input_shape'], c.network['output_shape'])
elif mod_type == 'Unet_res':
    mod = model.Unet_res(c.network['input_shape'], c.network['output_shape'])
elif mod_type == 'Unet_glob':
    mod = model.Global_Unet(num_layers, c.network['input_shape1'], c.network['input_shape2'], c.network['output_shape1'], c.network['output_shape2'])
mod.to(c.device)

## Dataset generation
if c.upstream == True or c.glob == True:
    train_dataloader, val_dataloader, test_dataloader = u.gen_dataloaders(c.dataset['sim_path'], c.dataset['ct_path'], c.dataset['name'], 
                                                                        c.dataset['summary_path'], c.training['validation_split'],
                                                                        c.training['batch_size'], c.dataset['tally'] )
else:
    train_dataloader, val_dataloader, test_dataloader = u.gen_dataloaders(c.dataset['pred_path'], c.training['validation_split'], c.training['batch_size'])


## Train routine
loss = c.training['loss']
optimizer = torch.optim.Adam(params = mod.parameters(), lr = c.training['lr'], betas = (c.training['beta1'], 0.999))
if c.upstream == True or c.glob == True:
    if not os.path.exists(c.save_path + f'/{c.model_name}'):
        print('Creating directory for model weights: ', c.save_path + f'/{c.model_name}_upstream')
        os.mkdir(c.save_path + f'/{c.model_name}')
    if not os.path.exists(c.logs_path + f'/{c.model_name}'):
        print('Creating directory for training logs: ', c.logs_path + f'/{c.model_name}_upstream')
        os.mkdir(c.logs_path + f'/{c.model_name}')
else:
    if not os.path.exists(c.save_path + f'/{c.model_name}'):
        print('Creating directory for model weights: ', c.save_path + f'/{c.model_name}_downstream')
        os.mkdir(c.save_path + f'/{c.model_name}')
    if not os.path.exists(c.logs_path + f'/{c.model_name}'):
        print('Creating directory for training logs: ', c.logs_path + f'/{c.model_name}_downstream')
        os.mkdir(c.logs_path + f'/{c.model_name}')
print('The model will be trained in: ', c.device)
t = Trainer(mod, loss, optimizer, c.training['num_epochs'], train_dataloader, val_dataloader, c.device, c.save_path, c.logs_path, c.training['save_period'],
            c.model_name, c.training['lr_scheduler'], c.training['lr_half'])

# Start training
t.train_loop()
