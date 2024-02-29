import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

class NoamOPT:
    def __init__(self,model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
        
    def step (self):
        #Change parameters + rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
        
    def rate(self, step =None):
        #implement the learning rate below
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size **(-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))
            
    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(parameters, d_model):
    return NoamOPT(
        d_model,2,4000, torch.optim.Adam(parameters, lr=0, betas=(0.9,0.98), eps = 1e-9)
    )