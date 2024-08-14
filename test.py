import sys

sys.path.insert(1, './configs')
sys.path.insert(1, './src')

import importlib
import argparse
import model
import torch
import os
import glob
import utils as u
from eval import Evaluator

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

weigths_list = glob.glob(os.path.join(c.save_path+f'/{c.model_name}', "*"))
mod.load_state_dict(torch.load(weigths_list[-1]))

## Dataset generation
if c.upstream == True or c.glob == True:
    train_dataloader, val_dataloader, test_dataloader = u.gen_dataloaders(c.dataset['sim_path'], c.dataset['ct_path'], c.dataset['name'], 
                                                                        c.dataset['summary_path'], 0.99,
                                                                        1, c.dataset['tally'] )
else:
    train_dataloader, val_dataloader, test_dataloader = u.gen_dataloaders(c.dataset['pred_path'], 0.99, 1)


# Iterators definition
train_iterator = iter(train_dataloader)
val_iterator = iter(val_dataloader)
test_iterator = iter(test_dataloader)

# Evaluation routine
if c.upstream == True or c.glob == True:
    if not os.path.exists(c.predictions_path + f'/{c.model_name}'):
        print('Creating directory for model predictions: ', c.predictions_path + f'/{c.model_name}')
        os.mkdir(c.predictions_path + f'/{c.model_name}')
else:
    if not os.path.exists(c.predictions_path + f'/{c.model_name}'):
        print('Creating directory for model predictions: ', c.predictions_path + f'/{c.model_name}')
        os.mkdir(c.predictions_path + f'/{c.model_name}')
e = Evaluator(mod, train_iterator, val_iterator, test_iterator, c.device, c.predictions_path, c.model_name)

# Make predictions
e.make_predictions()

