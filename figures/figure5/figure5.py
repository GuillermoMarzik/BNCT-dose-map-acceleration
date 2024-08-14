import sys

sys.path.insert(1, '../../src')
sys.path.insert(1, '../../configs')

import importlib
import model 
import torch
import os
import utils as u
from trainer import Trainer
import glob
import numpy as np
from dataset import BNCTDataset
import matplotlib.pyplot as plt

c = importlib.import_module("unet_tcia_104_106_lossMSEpond_modSkip_dosiscomp")

## Model params
# Number of consecutive same dimension layers
num_layers = c.network['num_layers']
mod_type = c.network['mod_type']
input_shape = c.network['input_shape'] # agregadas 21/11
output_shape = c.network['output_shape']
if mod_type == 'Unet':
    mod = model.Unet(num_layers, input_shape, output_shape)
elif mod_type == 'Unet_skip':
    mod = model.Unet_skip(num_layers, input_shape, output_shape)
elif mod_type == 'Unet_res':
    mod = model.Unet_res(input_shape, output_shape)
mod.to(c.device)

# Load model weigths
weigths_list = glob.glob(os.path.join('../../models'+f'/{c.model_name}', "*"))
mod.load_state_dict(torch.load(weigths_list[-1]))

with open(f'../../data/full_104_106/summary/high.txt', 'rt') as f:
            lines = f.readlines()
max_val_h_boro = float(lines[1].split(',')[0])
max_val_h_hydrogen = float(lines[1].split(',')[4])
max_val_h_nitrogen = float(lines[1].split(',')[8])
max_val_h_brainf = float(lines[1].split(',')[20])
max_val_h_bonef = float(lines[1].split(',')[24])
max_val_h = [max_val_h_boro,max_val_h_hydrogen,max_val_h_nitrogen,max_val_h_brainf,max_val_h_bonef]

with open(f'../../data/full_104_106/summary/low.txt', 'rt') as f:
            lines = f.readlines()
max_val_l_boro = float(lines[1].split(',')[0])
max_val_l_hydrogen = float(lines[1].split(',')[4])
max_val_l_nitrogen = float(lines[1].split(',')[8])
max_val_l_brainf = float(lines[1].split(',')[20])
max_val_l_bonef = float(lines[1].split(',')[24])
max_val_l = [max_val_l_boro,max_val_l_hydrogen,max_val_l_nitrogen,max_val_l_brainf,max_val_l_bonef]

pg, ctg, _ = u.load_pred(max_val_h, max_val_l, c.model_name)

out = pg[1]
pred = pg[2]
inp = pg[0]
inp = inp/out.max()
pred = pred/out.max()
out = out/out.max()
idx = np.where(out>0.1) # discard very low values
diff = abs(out[idx]-pred[idx])
dif = abs(out-pred)
dif2 = abs(out-inp)
mse = ((out[idx] - pred[idx])**2).mean(axis=None)
mse2 = ((out[idx] - inp[idx])**2).mean(axis=None)
indices_list = u.get_indices_by_ranges(out[idx].flatten(), 10)
out_flat = out[idx].flatten()
inp_flat = inp[idx].flatten()
pred_flat = pred[idx].flatten()
stds_pred_low = []
stds_pred_high = []
stds_inp_low = []
stds_inp_high = []
for i in range(len(indices_list)):
    stds_pred_low.append(np.percentile(out_flat[indices_list[i]] - pred_flat[indices_list[i]],10))
    stds_pred_high.append(np.percentile(out_flat[indices_list[i]] - pred_flat[indices_list[i]],90))
    stds_inp_low.append(np.percentile(out_flat[indices_list[i]] - inp_flat[indices_list[i]],10))
    stds_inp_high.append(np.percentile(out_flat[indices_list[i]] - inp_flat[indices_list[i]],90))
y = np.linspace(0.1,1,num=10)
up_bound_pred = y - abs(np.array(stds_pred_high))
low_bound_pred = y + abs(np.array(stds_pred_low))
up_bound_inp = y - abs(np.array(stds_inp_high))
low_bound_inp = y + abs(np.array(stds_inp_low))
    
fig, ax = plt.subplots()
ax.fill_between(y, low_bound_inp, up_bound_inp, color='blue', alpha=0.3, label=r'$Input\;simulation$')
ax.fill_between(y, low_bound_pred, up_bound_pred, color='red', alpha=0.3, label=r'$Neural\;network\;prediction$')
ax.set_xlabel(r'$Normalized\;ground\;truth\;dose$')
ax.set_ylabel(r'$Normalized\;evaluated\;dose$')
ax.plot(y,y, ls='--', c='black', label=f'$y = x$')
plt.legend()
plt.tight_layout()
plt.show()

