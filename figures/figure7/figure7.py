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

pg, ctg, pg_raw = u.load_pred(max_val_h, max_val_l, c.model_name)
del pg

worst_idx = 496 
avg_idx = 1059
best_idx = 1615

p_worst = pg_raw[:,worst_idx,...]
p_avg = pg_raw[:,avg_idx,...]
p_best = pg_raw[:,best_idx,...]

ctg_worst = ctg[worst_idx,...]
ctg_avg = ctg[avg_idx,...]
ctg_best = ctg[best_idx,...]

p_gen = [p_worst,p_avg,p_best]
ctg_gen = [ctg_worst, ctg_avg, ctg_best]

map_dose_l_tumor_peor, map_dose_l_sano_peor = u.mapa_dosis_tumor_sano(p_gen[0][0],'low',ctg_gen[0])
map_dose_h_tumor_peor, map_dose_h_sano_peor = u.mapa_dosis_tumor_sano(p_gen[0][1],'high',ctg_gen[0])
map_dose_pred_tumor_peor, map_dose_pred_sano_peor = u.mapa_dosis_tumor_sano(p_gen[0][2],'high',ctg_gen[0])
p_tumor_peor = np.stack((map_dose_l_tumor_peor[...,::-1],map_dose_h_tumor_peor[...,::-1],map_dose_pred_tumor_peor[...,::-1],ctg_gen[0][0,...,::-1]))
p_sano_peor = np.stack((map_dose_l_sano_peor[...,::-1],map_dose_h_sano_peor[...,::-1],map_dose_h_sano_peor[...,::-1],ctg_gen[0][0,...,::-1]))

map_dose_l_tumor_avg, map_dose_l_sano_avg = u.mapa_dosis_tumor_sano(p_gen[1][0],'low',ctg_gen[1])
map_dose_h_tumor_avg, map_dose_h_sano_avg = u.mapa_dosis_tumor_sano(p_gen[1][1],'high',ctg_gen[1])
map_dose_pred_tumor_avg, map_dose_pred_sano_avg = u.mapa_dosis_tumor_sano(p_gen[1][2],'high',ctg_gen[1])
p_tumor_avg = np.stack((map_dose_l_tumor_avg[...,::-1],map_dose_h_tumor_avg[...,::-1],map_dose_pred_tumor_avg[...,::-1],ctg_gen[1][0,...,::-1]))
p_sano_avg = np.stack((map_dose_l_sano_avg[...,::-1],map_dose_h_sano_avg[...,::-1],map_dose_h_sano_avg[...,::-1],ctg_gen[1][0,...,::-1]))

map_dose_l_tumor_best, map_dose_l_sano_best = u.mapa_dosis_tumor_sano(p_gen[2][0],'low',ctg_gen[2])
map_dose_h_tumor_best, map_dose_h_sano_best = u.mapa_dosis_tumor_sano(p_gen[2][1],'high',ctg_gen[2])
map_dose_pred_tumor_best, map_dose_pred_sano_best = u.mapa_dosis_tumor_sano(p_gen[2][2],'high',ctg_gen[2])
p_tumor_best = np.stack((map_dose_l_tumor_best[...,::-1],map_dose_h_tumor_best[...,::-1],map_dose_pred_tumor_best[...,::-1],ctg_gen[2][0,...,::-1]))
p_sano_best = np.stack((map_dose_l_sano_best[...,::-1],map_dose_h_sano_best[...,::-1],map_dose_h_sano_best[...,::-1],ctg_gen[2][0,...,::-1]))

p_tumor = [p_tumor_peor,p_tumor_avg,p_tumor_best]
p_sano = [p_sano_peor,p_sano_avg,p_sano_best]
perf_x = [12,12,13]
perf_y = [15,10,13]

for i in range(3):
    u.plot_cortes2D_tumor_sano(p_tumor[i],p_sano[i],perf_x[i],perf_y[i])
