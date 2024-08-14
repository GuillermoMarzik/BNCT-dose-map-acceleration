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
mae_prom = np.mean(dif**2)
stds = [np.std(i) for i in dif]
std_prom = np.mean(stds)
max_err = [np.max(i) for i in dif]
max_err_idx = [np.where(dif[i]==max_err[i]) for i in range(len(dif))]
max_prom = np.max(max_err)
min_err = [np.min(i) for i in dif]
min_prom = np.mean(min_err)
abs_err = [(100/(24**3)) * (np.linalg.norm(dif[i])/np.max(out[i,...])) for i in range(len(dif))]
abs_err = np.array(abs_err)
abs_err = abs_err[np.isfinite(abs_err)]
abs_err = np.mean(abs_err)
pred_data = [mse, std_prom, max_prom, min_prom, abs_err]
dif_inp = abs(inp-out)
mae_prom_inp = np.mean(dif_inp**2)
stds_inp = [np.std(i) for i in dif_inp]
std_prom_inp = np.sqrt(np.sum(stds_inp)/len(dif_inp))
max_err_inp = [np.max(i) for i in dif_inp]
max_prom_inp = np.max(max_err_inp)
min_err_inp = [np.min(i) for i in dif_inp]
min_prom_inp = np.mean(min_err_inp)
abs_err_inp = [(100/(24**3)) * (np.linalg.norm(dif_inp[i])/np.max(out[i,...])) for i in range(len(dif_inp))]
abs_err_inp = np.array(abs_err_inp)
abs_err_inp = abs_err_inp[np.isfinite(abs_err_inp)]
abs_err_inp = np.mean(abs_err_inp)
inp_data = [mse2, std_prom_inp, max_prom_inp, min_prom_inp, abs_err_inp]

print('Neural network statistics')
print(f'Normalized average error: {round(pred_data[0],5)}')
print(f'Normalized average standard deviation: {round(pred_data[1],5)}')
print(f'Normalized average maximum error: {round(pred_data[2],5)}')
print(f'Normalized average minimum error: {round(pred_data[3],5)}')
print(f'Average absolute error: {round(pred_data[4],5)}')
print('Low quality simulation statistics')
print(f'Normalized average error: {round(inp_data[0],5)}')
print(f'Normalized average standard deviation: {round(inp_data[1],5)}')
print(f'Normalized average maximum error: {round(inp_data[2],5)}')
print(f'Normalized average minimum error: {round(inp_data[3],5)}')
print(f'Average absolute error: {round(inp_data[4],5)}')
