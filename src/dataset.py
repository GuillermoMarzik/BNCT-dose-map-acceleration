import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
import random
import os
import glob
from leerTallies import Tally
#from utils import train_load

def pos_fuente(ini_file, campo):
    with open(ini_file, 'rt') as f:
        cont = f.readlines()
    idx = cont.index(campo+'\n')
    p = cont[idx+2].split('=')[1].split(';')
    pos = [int(i)//10 for i in p]
    d = cont[idx+3].split('=')[1].split(';')
    dir = [int(i) for i in d]
    return pos, dir

def train_load(ct_path, dataset_name, idx):
    cts_list = glob.glob(os.path.join(ct_path, "*"))
    ini_path = '/'.join(ct_path.split('/')[:-1]) + '/ini'
    ini_list = glob.glob(os.path.join(ini_path, "*"))
    sim_list_low = glob.glob(os.path.join('./data/' + dataset_name +'/low', "*"))
    sim_list_high = glob.glob(os.path.join('./data/' + dataset_name +'/high', "*"))
    field = sim_list_low[idx]
    field_name = '_'.join(field.split('/')[-1].split('_')[:-1])
    idx_high = [i for i, x in enumerate(sim_list_high) if field_name == '_'.join(x.split('/')[-1].split('_')[:-1])]
    idx_high = idx_high[0]
    field_ini = field_name.split('_')[0]
    idx_ini = [i for i, x in enumerate(ini_list) if field_ini == x.split('/')[-1].split('.')[0]][0]
    ini_campo = '[' + '_'.join(field.split('/')[-1].split('.')[0].split('_')[2:4]) + ']'
    ct_name = field_name[:5]
    idx_ct = [i for i, x in enumerate(cts_list) if ct_name in x][0]
    sim_low = Tally.load_tally(field)
    sim_high = Tally.load_tally(sim_list_high[idx_high])
    ct = np.load(cts_list[idx_ct])
    pos, dir = pos_fuente(ini_list[idx_ini], ini_campo)
    return ct, sim_low, sim_high, pos, dir

class BNCTDataset(Dataset):
    def __init__(self, sim_path, ct_path, summary_path, dataset_name, pred_tal, test=False):
        if not test:
            self.sim_path = sim_path
        else:
            self.sim_path = sim_path + '/test'
        self.ct_path = ct_path
        self.sim_list = glob.glob(os.path.join(self.sim_path+'/low', "*"))
        self.dataset_name = dataset_name
        self.test = test
        self.pred_tal = pred_tal
        
        low_summary_path = summary_path + '/low.txt'
        high_summary_path = summary_path + '/high.txt'

        with open(low_summary_path, 'r') as f:
            lines = f.readlines()

        #REPETIR PARA TODAS LAS DOSIS 17/11
        #TEST SOLO BORO
        self.low_max_val = []
        self.low_min_val = []
        num_tallies = 7
        for i in range(num_tallies):
            if i < 4:
                self.low_max_val.append(float(lines[1].split(',')[i*4]))
                self.low_min_val.append(float(lines[1].split(',')[i*4+1]))
            elif i < 6:
                self.low_max_val.append(float(lines[1].split(',')[i*4+4]))
                self.low_min_val.append(float(lines[1].split(',')[i*4+5]))
            else:
                self.low_max_val.append(float(lines[1].split(',')[i*4+8]))
                self.low_min_val.append(float(lines[1].split(',')[i*4+9]))

        with open(high_summary_path, 'r') as f:
            lines = f.readlines()

        self.high_max_val = []
        self.high_min_val = []
        for i in range(num_tallies):
            if i < 4:
                self.high_max_val.append(float(lines[1].split(',')[i*4]))
                self.high_min_val.append(float(lines[1].split(',')[i*4+1]))
            elif i < 6:
                self.high_max_val.append(float(lines[1].split(',')[i*4+4]))
                self.high_min_val.append(float(lines[1].split(',')[i*4+5]))
            else:
                self.high_max_val.append(float(lines[1].split(',')[i*4+8]))
                self.high_min_val.append(float(lines[1].split(',')[i*4+9]))

        #self.low_max_val = float(lines[1].split(',')[0])
        #self.low_min_val = float(lines[1].split(',')[1])
        
        #with open(high_summary_path, 'r') as f:
        #    lines = f.readlines()
        #TEST SOLO BORO
        #self.high_max_val = float(lines[1].split(',')[0])
        #self.high_min_val = float(lines[1].split(',')[1])

    def __len__(self):
        return len(self.sim_list)

    def __getitem__(self, idx):
        # Get sim path
        sim_path = self.sim_list[idx]
        # Get simulation ID
        sim_id = '_'.join(sim_path.split('/')[-1].split('_')[:-1])
        # Get CT, low quality sim, error of low quality sim and high quality sim
        ct, sim_low, sim_high, pos, dir = train_load(self.ct_path, self.dataset_name, idx)
        # Preprocessing
        proc_ct = self.preprocessing_ct(ct)
        proc_sim_low_val, proc_sim_low_err = self.preprocessing_sim(sim_low, self.high_max_val, self.low_min_val)
        proc_sim_high_val, proc_sim_high_err = self.preprocessing_sim(sim_high, self.high_max_val, self.low_max_val)
        dist_mat = self.mat_fuente(pos, dir, ct[...,-1].shape)
        az_mat, el_mat = self.ang_fuente(pos, ct[...,-1].shape)
        cam_lib_mat = self.camino_libre(dist_mat)

        if self.pred_tal != -1 and self.pred_tal != -2:
            proc_sim_low_val = proc_sim_low_val[self.pred_tal,...]
            proc_sim_low_err = proc_sim_low_err[self.pred_tal,...]
            proc_sim_high_val = proc_sim_high_val[self.pred_tal,...]
            proc_sim_high_err = proc_sim_high_err[self.pred_tal,...]
    
            proc_sim_low_val = torch.reshape(proc_sim_low_val,(1,24,24,24))
            proc_sim_low_err = torch.reshape(proc_sim_low_err,(1,24,24,24))
            proc_sim_high_val = torch.reshape(proc_sim_high_val,(1,24,24,24))
            proc_sim_high_err = torch.reshape(proc_sim_high_err,(1,24,24,24))
        elif self.pred_tal == -1:
            proc_sim_low_val = torch.reshape(proc_sim_low_val,(5,24,24,24))
            proc_sim_low_err = torch.reshape(proc_sim_low_err,(5,24,24,24))
            proc_sim_high_val = torch.reshape(proc_sim_high_val,(5,24,24,24))
            proc_sim_high_err = torch.reshape(proc_sim_high_err,(5,24,24,24))

        if self.pred_tal != -2:
            inp = torch.cat((proc_ct,proc_sim_low_val,proc_sim_low_err, dist_mat), dim=0) # 3d tensor with n channels
            output = torch.cat((proc_sim_high_val,proc_sim_high_err), dim=0) # 3d tensor with n channels
        else:
            proc_sim_high_val = torch.reshape(proc_sim_high_val,(5,24,24,24))
            proc_sim_high_err = torch.reshape(proc_sim_high_err,(5,24,24,24))
            inp = torch.cat((proc_ct, dist_mat, az_mat, el_mat, cam_lib_mat), dim=0)
            output = torch.cat((proc_sim_high_val, proc_sim_high_val), dim=0)
        #output = torch.cat((proc_sim_high_val[[3],...],proc_sim_high_err[[3],...]), dim = 0) # viejo, solo boro
        # Augmentation
        # Flip
        if random.randint(0, 2):
            inp = inp.flip(-1)
            output = output.flip(-1)
        return inp, output

    @staticmethod
    def preprocessing_ct(ct: np.ndarray) -> torch.Tensor:
        """ 
        Preprocess CT: Casting and Normalization.
        Args:
            ct (Numpy array[dimx, dimy, dimz, channels]): CT to preprocess, where channels can be 0, 1, 2 or 3 (air, soft tissue, bone, tumor).
        Returns:
            ct (Tensor[channels, dimx, dimy, dimz]): Preprocessed CT.
        """
        # Cast to tensor, rearrange and cast to float
        ct = torch.from_numpy(ct)
        ct = torch.permute(ct, (3,0,1,2))
        ct = ct.type(torch.float)
 
        # Normalization
        ct = ct/5 - 1.0
        
        return ct
    
    @staticmethod
    def mat_fuente(pos, dir, shape) -> torch.Tensor:
        """ 
        Calculo la matriz de distancias de cada voxel a la fuente.
        Args:
            pos (Numpy array[posx, posy, posz]): posición de la fuente en x, y, z.
            dir (Numpy array[dirx, diry]): ángulos directores de la fuente.
            shape (Numpy array[shapex, shapey, shapez]): dimensiones de la CT que estamos procesando.
        Returns:
            dist (Tensor[dimx, dimy, dimz]): matriz de distancias a la fuente.
        """
        x, y, z = np.indices(shape)
        coordinates = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        # Calculate the Euclidean distance for each element in A and B
        dist = np.linalg.norm(coordinates - pos, axis=1)
        # Reshape the distances array to match the shape of A (2x2)
        dist = dist.reshape(shape)
        dist = 1 - dist/dist.max()
        dist = np.expand_dims(dist,axis=0)
        dist = torch.from_numpy(dist)
        dist = dist.type(torch.float)
        
        return dist

    @staticmethod
    def ang_fuente(pos, shape) -> tuple:
        """
        Calculo la matriz de ángulos a la fuente.
        Args:
            pos (Numpy array[posx, posy, posz]): posición de la fuente en x, y, z.
            shape (Numpy array[shapex, shapey, shapez]): dimensiones de la CT que estamos procesando.
        Returns:
            azimuth_degrees (Torch tensor): matriz de azimuths a la fuente, en grados.
            elevation_degrees (Torch tensor): matriz de ángulos de elevación a la fuente, en grados.
        """
        x, y, z = np.indices(shape)
        coordinates = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        EPS = np.finfo(float).eps
        v = coordinates - pos
        azimuth = np.arctan2(v[:,1],v[:,0])
        azimuth_degrees = np.degrees(azimuth)
        azimuth_degrees = azimuth_degrees.reshape(shape)
        azimuth_degrees = azimuth_degrees / np.max(azimuth_degrees)
        azimuth_degrees = np.expand_dims(azimuth_degrees, axis=0)
        azimuth_degrees = torch.from_numpy(azimuth_degrees)
        azimuth_degrees = azimuth_degrees.type(torch.float)
        
        elevation = np.arctan2(v[:,2], np.linalg.norm(v[:,:2]))
        elevation_degrees = np.degrees(elevation)
        elevation_degrees = elevation_degrees.reshape(shape)
        elevation_degrees = elevation_degrees / np.max(elevation_degrees+EPS)
        elevation_degrees = np.expand_dims(elevation_degrees, axis=0)
        elevation_degrees = torch.from_numpy(elevation_degrees)
        elevation_degrees = elevation_degrees.type(torch.float)

        return azimuth_degrees, elevation_degrees
    
    @staticmethod
    def camino_libre(dist) -> torch.Tensor:
        """
        Calculo la matriz de caminos libres medios para cada punto, dada una matriz de distancias previamente calculada.
        Args:
            dist (torch.Tensor): matriz de distancias a la fuente.
            shape (Numpy array): dimensiones de la ct que se está procesando.
        Returns:
            camino_mat (torch.Tensor): matriz de caminos libres medios a la fuente.
        """
        camino_mat = torch.exp(-dist/2)
        camino_mat = camino_mat.type(torch.float)
        return camino_mat

    @staticmethod
    def preprocessing_sim(sim: np.ndarray, max_val: list, min_val: list) -> tuple:
        """ 
        Preprocess simulation: Casting and normalization (linear conversion).
        Args:
            sim (Numpy array[dimx, dimy, dimz, channels, type]): Neutronic transport simulation to preprocess, where channels can
            be 0, 1, 2 or 3 (air, soft tissue, bone, tumor) and type can be 0 or 1 (tally value or tally error).
            max_val (float): Maximum value of the dataset under analysis.
            min_val (float): Minimum value of the dataset under analysis.
        Returns:
            sim_val (Tensor[channels, dimx, dimy, dimz]): Processed neutronic transport simulation value.
            sim_err (Tensor[channels, dimx, dimy, dimz]): Processed neutronic transport simulation error.
        """
        # BORO
        sim_val_boro = sim[...,3]
        sim_err_boro = sim[...,7]

        # CEREBRO
        sim_val_h = sim[...,11]
        sim_err_h = sim[...,15]
        
        # CRANEO
        sim_val_n = sim[...,19]
        sim_err_n = sim[...,23]

        # CEREBRO FOTONES
        sim_val_cerebrof = sim[...,24]
        sim_err_cerebrof = sim[...,25]

        # CRANEO FOTONES
        sim_val_craneof = sim[...,26]
        sim_err_craneof = sim[...,27]

        #Normalización lineal
        sim_val_boro = 2*(sim_val_boro / max_val[0])-1
        sim_val_h = 2*(sim_val_h / max_val[1])-1
        sim_val_n = 2*(sim_val_n / max_val[2])-1        
        sim_val_cerebrof = 2*(sim_val_cerebrof / max_val[3])-1
        sim_val_craneof = 2*(sim_val_craneof / max_val[4])-1

        #Concateno las distintas componentes de dosis
        sim_val_boro = np.expand_dims(sim_val_boro, axis=-1)
        sim_val_h = np.expand_dims(sim_val_h, axis=-1)
        sim_val_n = np.expand_dims(sim_val_n, axis=-1)
        sim_val_cerebrof = np.expand_dims(sim_val_cerebrof, axis=-1)
        sim_val_craneof = np.expand_dims(sim_val_craneof, axis=-1)
        sim_val = np.concatenate((sim_val_boro, sim_val_h, sim_val_n, sim_val_cerebrof,
                                  sim_val_craneof, ), axis=-1)

        sim_err_boro = np.expand_dims(sim_err_boro, axis=-1)
        sim_err_h = np.expand_dims(sim_err_h, axis=-1)
        sim_err_n = np.expand_dims(sim_err_n, axis=-1)
        sim_err_cerebrof = np.expand_dims(sim_err_cerebrof, axis=-1)
        sim_err_craneof = np.expand_dims(sim_err_craneof, axis=-1)
        sim_err = np.concatenate((sim_err_boro, sim_err_h, sim_err_n,
                                  sim_err_cerebrof, sim_err_craneof), axis=-1)
        
        sim_val = torch.from_numpy(sim_val)
        sim_val = torch.permute(sim_val, (3,0,1,2))
        sim_val = sim_val.type(torch.float)
        sim_err = torch.from_numpy(sim_err)
        sim_err = torch.permute(sim_err, (3,0,1,2))
        sim_err = sim_err.type(torch.float)

        return (sim_val, sim_err)
