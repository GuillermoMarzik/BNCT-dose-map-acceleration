import numpy as np
import glob
import os
from torch.utils.data import DataLoader, random_split
from dataset import BNCTDataset
import csv
import torch
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt

def gen_dataloaders(sim_path, ct_path, dataset_name, summary_path, validation_split, batch_size, pred_tal):
    dataset = BNCTDataset(sim_path = sim_path, ct_path = ct_path, summary_path = summary_path, dataset_name = dataset_name, pred_tal = pred_tal, test = False)
    full_dataset_length = len(dataset)
    train_data_length = round(full_dataset_length * validation_split)
    val_data_length = round(full_dataset_length * (1-validation_split))
    train_dataset, val_dataset = random_split(dataset, [train_data_length, val_data_length])
    test_dataset = BNCTDataset(sim_path = sim_path, ct_path = ct_path, summary_path = summary_path, dataset_name = dataset_name, pred_tal = pred_tal, test = True)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    return train_dataloader, val_dataloader, test_dataloader
   
def gen_fill_mat(mat_shape, mat_type, idx):
    mat = np.zeros(mat_shape)
    with open('../../src/ppm_rbe.csv', 'rt') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if row[0] == mat_type:
                break
    for i in range(len(idx)):
        for j in range(len(idx[i][0])):
            mat[i,idx[i][0][j], idx[i][1][j], idx[i][2][j]] = float(row[i+1])
    return mat

def find_idx(ct,chan_type):
    return np.where(ct[chan_type,...] != 0)

def mapa_dosis(tal, typ, ct):
    ct = ct/10
    num_tallies = 5
    batch_size = len(tal)
    idx_aire = find_idx(ct,0)
    idx_blando = find_idx(ct,1)
    idx_hueso = find_idx(ct,2)
    idx_tumor = find_idx(ct,3)
    idx = [idx_aire,idx_blando,idx_hueso,idx_tumor]
    ppm_mat = gen_fill_mat(ct.shape,'ppm',idx)
    cbe = gen_fill_mat(ct.shape,'cbe',idx)
    rbe = gen_fill_mat(ct.shape,'rbe',idx)
    rbe_f = gen_fill_mat(ct.shape,'rbe_f',idx)
    map_dosis = np.zeros((num_tallies,24,24,24))
    for i in range(num_tallies):
        if i == 0:
            for j in range(4):
                map_dosis[i,...] = map_dosis[i,...] + tal[i,...] * cbe[j,...] * ct[j,...] * ppm_mat[j,...] 
        elif i == 1 or i == 2:
            for j in range(4):
                map_dosis[i,...] = map_dosis[i,...] + tal[i,...] * rbe[j,...] * ct[j,...]
        elif i == 3 or i == 4:
            for j in range(4):
                    
                map_dosis[i,...] = map_dosis[i,...] + tal[i,...] * rbe_f[j,...] * ct[j,...]
    map_dosis = np.sum(map_dosis, axis=0)
    return map_dosis
    
def mapa_dosis_nh(tal, typ, ct):
    ct = ct/10
    num_tallies = 5
    tal = np.float64(tal)
    batch_size = len(tal)
    idx_aire = find_idx(ct,0)
    idx_blando = find_idx(ct,1)
    idx_hueso = find_idx(ct,2)
    idx_tumor = find_idx(ct,3)
    idx = [idx_aire,idx_blando,idx_hueso,idx_tumor]
    ppm_mat = gen_fill_mat(ct.shape,'ppm',idx)
    cbe = gen_fill_mat(ct.shape,'cbe',idx)
    rbe = gen_fill_mat(ct.shape,'rbe',idx)
    rbe_f = gen_fill_mat(ct.shape,'rbe_f',idx)
    map_dosis = np.zeros((num_tallies,24,24,24))
    tal_boro = tal[0,...]*10e-6 # para compensar diferencias entre órdenes de los tallies
    frac_masa_h = [0,0.11,0.05,0.11] # aire, blando, hueso, tumor
    frac_masa_n = [0,0.022,0.04,0.022] # aire, blando, hueso, tumor
    for i in range(num_tallies):
        if i == 0:
            for j in range(4):
                map_dosis[i,...] = map_dosis[i,...] + tal_boro * cbe[j,...] * ct[j,...] * ppm_mat[j,...]
        elif i == 1 or i == 2:
            for j in range(4):
                if i == 1:
                    map_dosis[i,...] = map_dosis[i,...] + tal[i,...] * rbe[j,...] * ct[j,...] * frac_masa_h[j]
                else:
                    map_dosis[i,...] = map_dosis[i,...] + tal[i,...] * rbe[j,...] * ct[j,...] * frac_masa_n[j]
        elif i == 3 or i == 4:
            for j in range(4):
                map_dosis[i,...] = map_dosis[i,...] + tal[i,...] * rbe_f[j,...] * ct[j,...]
    map_dosis = np.sum(map_dosis, axis=0)
    return map_dosis
    
def mapa_dosis_tumor_sano(tal, typ, ct):
    ct = ct/10
    num_tallies = 5
    batch_size = len(tal)
    idx_aire = find_idx(ct,0)
    idx_blando = find_idx(ct,1)
    idx_hueso = find_idx(ct,2)
    idx_tumor = find_idx(ct,3)
    idx = [idx_aire,idx_blando,idx_hueso,idx_tumor]
    ppm_mat_tumor = gen_fill_mat(ct.shape,'ppm_tumor',idx)
    ppm_mat_sano = gen_fill_mat(ct.shape,'ppm_sano',idx)
    cbe_tumor = gen_fill_mat(ct.shape,'cbe_tumor',idx)
    rbe_tumor = gen_fill_mat(ct.shape,'rbe_tumor',idx)
    rbe_f_tumor = gen_fill_mat(ct.shape,'rbe_f_tumor',idx)
    cbe_sano = gen_fill_mat(ct.shape,'cbe_sano',idx)
    rbe_sano = gen_fill_mat(ct.shape,'rbe_sano',idx)
    rbe_f_sano = gen_fill_mat(ct.shape,'rbe_f_sano',idx)
    map_dosis_sano = np.zeros((num_tallies,24,24,24))
    map_dosis_tumor = np.zeros((num_tallies,24,24,24))
    for i in range(num_tallies):
        if i == 0:
            for j in range(4):
                map_dosis_tumor[i,...] = map_dosis_tumor[i,...] + tal[i,...] * cbe_tumor[j,...] * ct[j,...] * ppm_mat_tumor[j,...]
                map_dosis_sano[i,...] = map_dosis_sano[i,...] + tal[i,...] * cbe_sano[j,...] * ct[j,...] * ppm_mat_sano[j,...] 
        elif i == 1 or i == 2:
            for j in range(4):
                map_dosis_tumor[i,...] = map_dosis_tumor[i,...] + tal[i,...] * rbe_tumor[j,...] * ct[j,...]
                map_dosis_sano[i,...] = map_dosis_sano[i,...] + tal[i,...] * rbe_sano[j,...] * ct[j,...]
            
        elif i == 3 or i == 4:
            for j in range(4):
                map_dosis_tumor[i,...] = map_dosis_tumor[i,...] + tal[i,...] * rbe_f_tumor[j,...] * ct[j,...]
                map_dosis_sano[i,...] = map_dosis_sano[i,...] + tal[i,...] * rbe_f_sano[j,...] * ct[j,...]
    map_dosis_tumor = np.sum(map_dosis_tumor, axis=0)
    map_dosis_sano = np.sum(map_dosis_sano, axis=0)
    return map_dosis_tumor, map_dosis_sano

def load_pred(m_val_h, m_val_l, model_name):
    # load neural net predictions
    test_pred = np.load(f'../../predictions/{model_name}/test_predictions.npy')
    mod_type = model_name.split('_')[-1]
    
    # processing of the whole test set
    if mod_type == 'nosim' or mod_type == 'global':
        ctg = test_pred['i'][:,0,:4,...]
        p_outg = test_pred['o'][:,0,:5,...]
        e_outg = test_pred['o'][:,0,5:,...]
        p_predg = test_pred['p'][:,0,...]
        pg_raw = np.stack((p_outg,p_predg))
    else:
        p_inpg = test_pred['i'][:,0,4:9,...]
        ctg = test_pred['i'][:,0,:4,...]
        p_outg = test_pred['o'][:,0,:5,...]
        e_outg = test_pred['o'][:,0,5:,...]
        p_predg = test_pred['p'][:,0,...]
        pg_raw = np.stack((p_inpg,p_outg,p_predg))
    
    ctg = (ctg+1)*5
    map_dose_lg = []
    map_dose_hg = []
    map_dose_predg = []
    n_tal = 5
    
    if mod_type == 'nosim' or mod_type == 'global':
        for i in range(n_tal):
            pg_raw[0][:,i,...] = (pg_raw[0][:,i,...]+1)*m_val_h[i]*0.5
            pg_raw[1][:,i,...] = (pg_raw[1][:,i,...]+1)*m_val_h[i]*0.5
        for i in range(len(pg_raw[0])):    
            map_dose_hg.append(mapa_dosis_nh(pg_raw[0][i,...],'high',ctg[i,...]))
            map_dose_predg.append(mapa_dosis_nh(pg_raw[1][i,...],'high',ctg[i,...]))
        pg = np.stack((map_dose_hg,map_dose_predg))
    else:    
        for i in range(n_tal):
            pg_raw[0][:,i,...] = (pg_raw[0][:,i,...]+1)*m_val_l[i]*0.5
            pg_raw[1][:,i,...] = (pg_raw[1][:,i,...]+1)*m_val_h[i]*0.5
            pg_raw[2][:,i,...] = (pg_raw[2][:,i,...]+1)*m_val_h[i]*0.5
        for i in range(len(pg_raw[0])):    
            map_dose_lg.append(mapa_dosis_nh(pg_raw[0][i,...],'low',ctg[i,...]))
            map_dose_hg.append(mapa_dosis_nh(pg_raw[1][i,...],'high',ctg[i,...]))
            map_dose_predg.append(mapa_dosis_nh(pg_raw[2][i,...],'high',ctg[i,...]))
    
        pg = np.stack((map_dose_lg,map_dose_hg,map_dose_predg))
    print('Processing of the test set finished succesfully')
    return pg, ctg, pg_raw

def get_indices_by_ranges(array, num_ranges):
    # Get the minimum and maximum values of the array
    min_val = np.min(array)
    max_val = np.max(array)
    
    # Create the range edges
    edges = np.linspace(min_val, max_val, num_ranges + 1)
    
    # Initialize the list to hold the indices
    indices_list = [[] for _ in range(num_ranges)]
    
    # Populate the list with indices
    for i in range(num_ranges):
        # Find indices for the current range
        if i == num_ranges - 1:
            # Include the maximum value in the last range
            indices = np.where((array >= edges[i]) & (array <= edges[i+1]))[0]
        else:
            indices = np.where((array >= edges[i]) & (array < edges[i+1]))[0]
        indices_list[i].extend(indices)
    
    return indices_list

def plot_cortes2D_tumor_sano(p1,p2,perf_x,perf_z):
    # dose profile plots
    linea_mc_baja_tumor = p1[0][:,perf_x,perf_z]
    linea_mc_baja_sano = p2[0][:,perf_x,perf_z]
    linea_mc_alta_tumor = p1[1][:,perf_x,perf_z]
    linea_mc_alta_sano = p2[1][:,perf_x,perf_z]
    linea_red_tumor = p1[2][:,perf_x,perf_z]
    linea_red_sano = p2[2][:,perf_x,perf_z]
        
    linea_mc_baja_tumor = linea_mc_baja_tumor / linea_mc_alta_tumor.max()
    linea_red_tumor = linea_red_tumor / linea_mc_alta_tumor.max()
    linea_mc_baja_sano = linea_mc_baja_sano / linea_mc_alta_tumor.max()
    linea_mc_alta_sano = linea_mc_alta_sano / linea_mc_alta_tumor.max()
    linea_red_sano = linea_red_sano / linea_mc_alta_tumor.max()
    linea_mc_alta_tumor = linea_mc_alta_tumor / linea_mc_alta_tumor.max()
        
    # spline interpolation for smoother curves
    x = np.linspace(0, 23, 75)
    x_old = np.linspace(0, 23, 24)
    spl = make_interp_spline(x_old, linea_mc_baja_tumor, k=3)
    linea_baja_tumor_smooth = spl(x)
    spl = make_interp_spline(x_old, linea_mc_alta_tumor, k=3)
    linea_alta_tumor_smooth = spl(x)
    spl = make_interp_spline(x_old, linea_red_tumor, k=3)
    linea_red_tumor_smooth = spl(x)
        
    yticks = [0,20,40]
        
    fig, ax = plt.subplots()
    ax.plot(x[5:],linea_baja_tumor_smooth[5:], ls='-', c='b', alpha=0.3, label=r'$Brain\;Tumor\;Dose\;Input\;Sim$')
    ax.plot(linea_mc_baja_sano, ls='--', c='b', alpha=0.3, label=r'$Healthy\;tissue\;Input\;Sim$')
    ax.plot(x[5:],linea_alta_tumor_smooth[5:], ls='-', c='k', label=r'$Brain\;Tumor\;Dose\;Ground\;Truth$')
    ax.plot(linea_mc_alta_sano, ls='--', c='k', label=r'$Healthy\;tissue\;Ground\;Truth$')
    ax.plot(x[5:],linea_red_tumor_smooth[5:], ls='-', c='r', alpha=0.3, label=r'$Brain\;Tumor\;Dose\;Neural\;Network$')
    ax.plot(linea_red_sano, ls='--', c='r', alpha=0.3, label=r'$Healthy\;tissue\;Neural\;Network$')
    ax.legend()
    ax.set_ylim(0, 1.5*linea_alta_tumor_smooth.max())
    ax.set_xlabel(r'$Distance\;[cm]$')
    ax.set_ylabel(r'$Normalized\;Dose$')
    plt.tight_layout()
    plt.show()
