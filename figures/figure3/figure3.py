import sys

sys.path.insert(1, '../../src')
sys.path.insert(1, '../../configs')

import importlib
import model 
import torch
import os
import utils as u
from trainer import Trainer
from leerTallies import Tally
import glob
import numpy as np
from dataset import BNCTDataset
import plotly.graph_objects as go
import matplotlib.cm as cm

## Load low, mid and high quality MC simulations (and CT)
low_q = Tally.load_tally('./izq_nh__10000.npy')
middle_q = Tally.load_tally('./izq_nh__1000000.npy')
high_q = Tally.load_tally('./izq_nh__100000000.npy')
ct = np.load('./00004.npy')

max_val_low = [np.max(low_q[...,3+i*8]) for i in range(3)]
max_val_high = [np.max(high_q[...,3+i*8]) for i in range(3)]
max_val_low.append(np.max(low_q[...,24]))
max_val_low.append(np.max(low_q[...,26]))
max_val_high.append(np.max(high_q[...,24]))
max_val_high.append(np.max(high_q[...,26]))

## Pre-processing for model
ct_tensor = BNCTDataset.preprocessing_ct(ct)

low_q_tensor = BNCTDataset.preprocessing_sim(low_q, max_val_low, [])
middle_q_tensor,_ = list(BNCTDataset.preprocessing_sim(middle_q, max_val_high, []))
high_q_tensor = BNCTDataset.preprocessing_sim(high_q, max_val_high, [])

## Prepare for input
inp = torch.cat((ct_tensor,low_q_tensor[0],low_q_tensor[1]), dim=0)
out = torch.clone(high_q_tensor[0])
#inp = inp[None,...].to(c.device)
inp = inp[None,...]
out = out[None,...]

ct = inp[0,:4,...]
inp = inp[0,4:9,...]
out = out[0,...]


# denormalization
for i in range(5):
    inp[i] = (inp[i]+1)*max_val_low[i]*0.5
    out[i] = (out[i]+1)*max_val_high[i]*0.5
    middle_q_tensor[i] = (middle_q_tensor[i]+1)*max_val_high[i]*0.5
ct = (ct+1)*5

inp = inp.cpu().detach().numpy()
mid = middle_q_tensor.cpu().detach().numpy()
out = out.cpu().detach().numpy()
ct = ct.cpu().detach().numpy()
map_dose_l_d = u.mapa_dosis_nh(inp,'low',ct)
map_dose_h_d = u.mapa_dosis_nh(out,'high',ct)
map_dose_mid_d = u.mapa_dosis_nh(mid,'high',ct)
ct  = ct/np.max(ct)

# 3D dose map plots
p = np.stack((map_dose_l_d[...,::-1],map_dose_mid_d[...,::-1],map_dose_h_d[...,::-1],ct[0,...,::-1]))
for i in range(3):
    X, Y, Z = np.mgrid[0:24:24j, 0:24:24j, 24:0:24j]
    # dose normalization
    dosis_norm = p[i]/p[i].max()
    
    # exclude extremely low dose values that could hinder the visualization
    dosis_norm[dosis_norm < 0.001] = 0
    # log scale transformation
    dosis_norm = 2+np.log10(1e-5+abs(dosis_norm.min())+dosis_norm)
    dosis_norm = dosis_norm/(dosis_norm.max()+0.0000001)
    # exclude extremely low dose values that could hinder the visualization
    dosis_norm[dosis_norm < 0.001] = 0
    
    # ct normalization
    ct = p[3]/p[3].max()
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
    
    if i == 0:
        tit = r'$Monte-Carlo\;-\;10^4\;histories$'
    elif i == 1:
        tit = r'$Monte-Carlo\;-\;10^6\;histories$'
    else:
        tit = r'$Monte-Carlo\;-\;10^8\;histories$'
        
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

