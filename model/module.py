import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size*2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size*2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size*2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size*2, y_dim),
                                      nn.ReLU())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
        
        
class Discriminator_50(nn.Module): 
    def __init__(self, num_classes=7):
        super(Discriminator_50, self).__init__()
        # self.drop_out = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, 128),
            nn.Linear(128, num_classes),
        )
        
                
    def forward(self, x):
        # x = self.drop_out(x)
        x = self.fc(x)
        
        return x
        
        
class Discriminator_18(nn.Module): 
    def __init__(self, num_classes=7):
        super(Discriminator_18, self).__init__()
        # self.drop_out = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, num_classes),
        )
        
                
    def forward(self, x):
        # x = self.drop_out(x)
        x = self.fc(x)
        
        return x
