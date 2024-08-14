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
import matplotlib.cm as cm
import plotly.graph_objects as go

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

worst_idx = 496 
avg_idx = 1059
best_idx = 1615

p_worst = pg[:,worst_idx,...]
p_avg = pg[:,avg_idx,...]
p_best = pg[:,best_idx,...]

ctg_worst = ctg[worst_idx,...]
ctg_avg = ctg[avg_idx,...]
ctg_best = ctg[best_idx,...]

p_gen = [p_worst,p_avg,p_best]
ctg_gen = [ctg_worst, ctg_avg, ctg_best]

# 3D dose map plots
for j in range(3):
    for i in range(3):
        X, Y, Z = np.mgrid[0:24:24j, 0:24:24j, 24:0:24j]
        # dose normalization
        dosis_norm = p_gen[j][i]/p_gen[j][i].max()
    
        # exclude extremely low dose values that could hinder the visualization
        dosis_norm[dosis_norm < 0.001] = 0
        # log scale transformation
        dosis_norm = 2+np.log10(1e-5+abs(dosis_norm.min())+dosis_norm)
        dosis_norm = dosis_norm/(dosis_norm.max()+0.0000001)
        # exclude extremely low dose values that could hinder the visualization
        dosis_norm[dosis_norm < 0.001] = 0
    
        # ct normalization
        ct = ctg_gen[j][0]/ctg_gen[j][0].max()
        cmap = cm.get_cmap('PiYG', 2)

        fig = go.Figure(data=[go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=ct.flatten(),
            isomin=0,
            isomax=1.1,
            opacity=.1,
            surface_count=30,
            colorscale=['rgb(0, 0, 0)','rgb(52, 235, 229)']),go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=dosis_norm.flatten(),
            colorbar={"title": 'Normalized dose', "titleside": 'right'},
            isomin=0,
            isomax=1.1,
            opacity=0.5,
            opacityscale=[[0, 0],[0.1,0.7],[1,1]], 
            surface_count=17
            )])
        fig.update_scenes(xaxis_title_text='X [cm]',  
                  yaxis_title_text='Y [cm]',  
                  zaxis_title_text='Z [cm]')
    
        if i == 0 and j == 0:
            tit = r'$Worst\;case\;Monte-Carlo\;-\;10^4\;histories$'
        elif i == 0 and j == 1:
            tit = r'$Average\;case\;Monte-Carlo\;-\;10^4\;histories$'
        elif i == 0 and j == 2:
            tit = r'$Best\;case\;Monte-Carlo\;-\;10^4\;histories$'
        elif i == 1 and j == 0:
            tit = r'$Worst\;case\;Monte-Carlo\;-\;10^6\;histories$'
        elif i == 1 and j == 1:
            tit = r'$Average\;case\;Monte-Carlo\;-\;10^6\;histories$'
        elif i == 1 and j == 2:
            tit = r'$Best\;case\;Monte-Carlo\;-\;10^6\;histories$'
        if i == 2 and j == 0:
            tit = r'$Worst\;case\;Monte-Carlo\;-\;10^8\;histories$'
        elif i == 2 and j == 1:
            tit = r'$Average\;case\;Monte-Carlo\;-\;10^8\;histories$'
        elif i == 2 and j == 2:
            tit = r'$Best\;case\;Monte-Carlo\;-\;10^8\;histories$'

        fig.update_layout(
            title=dict(text=f"Dose map {tit}", yref='paper')
        )

        fig.update_layout(margin={"r":100,"t":20,"l":0,"b":0},         
        )

        name = 'eye = (x:2, y:2, z:0.1)'
        camera = dict(
            eye=dict(x=-2, y=-2, z=2)
        )

        fig.update_layout(scene_camera=camera)

        fig.show()
