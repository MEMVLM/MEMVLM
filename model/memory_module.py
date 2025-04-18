from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from utils import *
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import numpy as np

entropy_loss_weight = 0.0002
class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, topk=1):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.first = True  # Flag for the first forward pass
        self.topk = topk
        self.q = nn.Linear(fea_dim, fea_dim)
        self.k = nn.Linear(fea_dim, fea_dim)
        self.v = nn.Linear(fea_dim, fea_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def reset_parameters_fea(self, features):
        print('init memory item')
        # Initialize the memory weights using random samples from the input features
        batch_size = features.shape[0]
        assert batch_size >= self.mem_dim, "The number of input features must be greater than or equal to mem_dim."
        
        # Randomly sample mem_dim features from the input
        indices = torch.randperm(batch_size)[:self.mem_dim]
        sampled_features = features[indices].detach().clone()  # Detach to avoid backprop through this
        self.weight.data.copy_(sampled_features)  # Initialize weight with the sampled features

    def forward(self, features, use_topk=True):
        print("topk:",self.topk)
        #if self.first:  # Only initialize during the first forward pass
            #self.reset_parameters_fea(features)
            #self.first = False  # Disable the flag after initialization

        features = features.to(self.weight.device)
        features_norm = self.q(features)
        weight_norm = self.k(self.weight)
        features_norm = F.normalize(features, p=2, dim=-1)  # (batch_size, fea_dim)
        weight_norm = F.normalize(self.weight, p=2, dim=-1)  # (mem_dim, fea_dim)

        # Compute cosine similarity between features and memory entries (batch_size, mem_dim)
        alpha = torch.matmul(features_norm, weight_norm.t())  # (batch_size, mem_dim)
        alpha = F.softmax(alpha, dim=-1)  # (batch_size, mem_dim)

        if use_topk:
            threshold = torch.topk(alpha, self.topk, dim=-1, sorted=True)[0][:, -1].unsqueeze(-1)  # (batch_size, 1)
            alpha = torch.where(alpha >= threshold, alpha, torch.zeros_like(alpha))  # (batch_size, mem_dim)
            alpha = F.normalize(alpha, p=2, dim=-1)  # Normalize after top-k filtering

        reconstructed_features = torch.matmul(alpha, self.v(self.weight))  # (batch_size, fea_dim)

        return {'output': reconstructed_features, 'att': alpha}
        
    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )

# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, topk = 20):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.mode = 'train'
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres, topk = topk)
        self.entropy_loss = EntropyLossEncap()


    def forward(self, input):
        input1 = input
        input = input.permute(0, 2, 1)
        ##############################   
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            ##[1, 256, 2, 16, 16]
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x.to('cuda:1'))
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        ###########################################
        y = y.permute(0, 2, 1)
        if self.mode == 'train':
            entropy_loss = self.entropy_loss(att)
            loss = simloss(input1, y) +  entropy_loss * entropy_loss_weight
            #print('loss:',loss.item(), entropy_loss.item())
        else:
            loss = simloss(input1, y)     
        
        return {'output': y, 'att': att, 'simloss':loss, 'topk_list':None}

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


def simloss(features1, features2):
    assert features1.shape == features2.shape, "Shapes of input tensors must be the same"
    features1 = features1.to(features2.device)
    loss = F.mse_loss(features1, features2).mean()

    return loss

# Helper functions
def index_to_2d(index, grid_size=35):
    return index // grid_size, index % grid_size  # Convert to (x, y) coordinates


def save_patch_losses(features1, features2, filename="patch_losses.txt"):
    assert features1.shape == features2.shape, "Input tensors must have the same shape"
    patch_losses = F.mse_loss(features1.cuda(3), features2.cuda(3), reduction='none').mean(dim=-1).squeeze(0)  # Shape: [2304]
    
    # Save to file
    with open(filename, "w") as f:
        for i, loss in enumerate(patch_losses):
            x, y = index_to_2d(i, grid_size=35)
            f.write(f"Patch ({x}, {y}): Loss {loss.item():.6f}\n")
    
    print(f"Patch losses saved to {filename}")
