from typing import Optional
from tqdm import tqdm
import torch
import os
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from utils import mapa_dosis

class MSE_pond(nn.Module):
    def __init__(self):
        super(MSE_pond, self).__init__()

    def forward(self, red_pred, targets, ct):
        t = (targets+1)/2
        r = (red_pred+1)/2
        loss = (t+1) * abs(r - t)
        return loss.mean()

class Trainer():
    def __init__(self, model: torch.nn.Module, loss_fn: str, optimizer: torch.optim.Optimizer, num_epochs: int, 
                 train_dataloader: torch.utils.data, val_dataloader: torch.utils.data, device: str,
                 save_root_path: str, logs_path: str, save_period: int, model_name: str, 
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, lr_half: Optional[int] = 5):
        self.model = model
        if loss_fn == 'MSE_dist':
            self.loss = MSE_dist()
        elif loss_fn == 'MSE_pond':
            self.loss = MSE_pond()
        elif loss_fn == 'MSE':
            self.loss = torch.nn.MSELoss()
        elif loss_fn == 'MAE_dose':
            self.loss = MAE_dose()
        else:
            self.loss = torch.nn.L1Loss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_half = lr_half
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_period = save_period
        self.save_root_path = save_root_path
        self.model_name = model_name

        self.train_loss = []
        self.val_loss = []
        self.lr = []
        self.writer = SummaryWriter(log_dir=os.path.join(logs_path, self.model_name))
    
    def make_train_step(self):
        def train_step(in_data, out_data):
            # Model in train mode
            self.model.train()
            # Makes predictions
            dist_mat = in_data[:,-1,...]
            ct = in_data[:,:4,...]
            pred_data = self.model(in_data[:,:-1,...])
            # Computes loss
            loss = self.loss(pred_data, out_data[:,:-5,...], ct)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Returns the loss
            return loss.item()
        # Returns the function that will be called inside the train loop
        return train_step
    
    def train_loop(self):
        train_step = self.make_train_step()
        last_loss = 100
        trigger_times = 0
        patience = 5
        min_val_loss = 100
        if self.lr_scheduler:
            self.scheduler = StepLR(self.optimizer, step_size=self.lr_half, gamma=0.5)
        for epoch in range(1,self.num_epochs+1):
            for data in tqdm(self.train_dataloader):
                in_data, out_data = data[0].to(self.device), data[1].to(self.device)
                loss = train_step(in_data, out_data)
                self.train_loss.append(loss)

            with torch.no_grad():
                for data in tqdm(self.val_dataloader): # validation loop
                    in_val_data, out_val_data = data[0].to(self.device), data[1].to(self.device)
                    self.model.eval() # Model in evaluation mode
                    dist_mat = in_val_data[:,-1,...]
                    ct_val = in_val_data[:,:4,...]
                    pred_val_data = self.model(in_val_data[:,:-1,...])
                    val_loss = self.loss(pred_val_data, out_val_data[:,:-5,...],ct_val).item()
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_state = self.model.state_dict()
                    self.val_loss.append(val_loss)

            print(f'Epoch {epoch} of {self.num_epochs}')
            print("\n METRICS"+ 10 * ' ' + 'Training Set' + 10 * ' ' + 'Validation Set')
            print(f"{len('METRICS' + 11* ' ')* ' '}{loss:.2e}{14 * ' '}{val_loss:.2e}")
            self.writer.add_scalar('Loss/train', loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            if val_loss > last_loss:
                trigger_times += 1
            else:
            	trigger_times = 0
            if trigger_times >= patience:
                print('Early Stopping!')
                torch.save(best_state, self.save_root_path+f'/{self.model_name}/best_model.pt')
                break
            print(f'Trigger times: {trigger_times}')
            print(f'Last loss: {last_loss}')
            print(f'Current loss: {val_loss}')
            last_loss = val_loss
            if epoch % self.save_period == 0:
                torch.save(self.model.state_dict(), self.save_root_path+f'/{self.model_name}/{epoch}.pt')
        if self.lr_scheduler is not None:
            self.scheduler = StepLR(self.optimizer, step_size=self.lr_half, gamma=0.5)
            if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.train_loss[-1] )
            else:
                self.scheduler.step()
            self.lr.append(self.scheduler.get_last_lr()[0])
        else:
            self.lr.append(self.optimizer.param_groups[0]["lr"])
        print(f'LR: {self.lr[-1]}')



