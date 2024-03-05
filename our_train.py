import os
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd

import itertools
import numpy as np
import copy
import shutil


device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load('RN101', device)
clip_model = clip_model#.float()

text_features_path = '/home/samyakr2/SHOP/foodseg103_labels.pt'
text_features = torch.load(text_features_path).to(torch.float32)

train_features_path = '/home/samyakr2/SHOP/foodseg103_train_img.pt'
train_features = torch.load(train_features_path)

train_labels_path = '/home/samyakr2/SHOP/foodseg103_train_label.pt'
train_labels = torch.load(train_labels_path)

val_features_path = '/home/samyakr2/SHOP/foodseg103_test_img.pt'
val_features = torch.load(val_features_path)

val_labels_path = '/home/samyakr2/SHOP/foodseg103_test_label.pt'
val_labels = torch.load(val_labels_path)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        
#         print('X.shape', x.shape)
#         print('Y shape',y.shape)
#         x_softmax = self.softmax(x)
#         print()
#         xs_pos = x_softmax[:, 1, :]
#         xs_neg = x_softmax[:, 0, :]
#         y = y.reshape(-1)
#         xs_pos = xs_pos.reshape(-1)
#         xs_neg = xs_neg.reshape(-1)

#         xs_pos = xs_pos[y!=-1]
#         xs_neg = xs_neg[y!=-1]
#         y = y[y!=-1]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg


        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class clip_2fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(clip_2fc, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, output_dim,bias=False)
        )
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim),
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
#         out = self.sigmoid(out)
        return out


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

input_size = train_features[0].size(1)  
hidden_size = 200
num_classes = 103 # len(labels_food)
model = clip_2fc(input_size, hidden_size, num_classes).to(device)

lr = 0.002
max_epochs = 50
warmup_epochs = 1
warmup_constant_lr = 1e-5

optimizer = torch.optim.SGD(model.parameters(), lr = lr)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=1, eta_min=warmup_constant_lr)

criterion = AsymmetricLoss(2, 1) # Y_neg = 2, Y_pos = 1


num_epochs = 20000
len_labels = 103


gamma = torch.ones(len_labels)/len_labels
gamma = gamma.to(device)

best_loss = float('inf')



# gamma = torch.ones(len_labels) / len_labels
# gamma = gamma.to(device)

# alpha = np.zeros(num_epochs)

for epoch in range(num_epochs):
    
    epoch_loss = 0.0
    for features_batch, labels_batch in zip(train_features, train_labels):
        # Flatten features batch
        
    
        features_batch = features_batch.view(features_batch.size(0), -1).to(torch.float32)
        gamma = gamma.detach()
        
        # Convert labels to tensor
        labels_tensor = labels_batch.type(torch.float32).to(device)#torch.tensor(labels_batch, dtype=torch.float32)#
        labels_tensor = labels_tensor.squeeze(dim=1)
        # Forward pass
        outputs = model(features_batch.to(device))
#         print('outputs shape',outputs)
        similarity_text = (text_features @ text_features.T)
        normalized_similarity_text = F.normalize(similarity_text, p=2, dim=1)  # Normalize along the second dimension (rows)
        
        normalized_similarity_text = torch.clamp(normalized_similarity_text, min=0, max=1)  # Clamp values to be between 0 and 1
        normalized_similarity_with_gamma = normalized_similarity_text * gamma

        outputs_reshaped = outputs.unsqueeze(1)
        
        result = torch.sum(outputs_reshaped * normalized_similarity_with_gamma.unsqueeze(0), dim =2)
        pred = result
        
        r = torch.sum(gamma * labels_tensor * torch.sigmoid(pred))
        a = 0.5 * torch.log((1 + r) / (1 - r))

        gamma = gamma * torch.exp(-a * labels_tensor * torch.sigmoid(pred))
        sum_val = torch.sum(gamma)
        gamma = gamma / sum_val
        gamma = torch.min(gamma, dim=0).values
        
        loss = criterion(pred, labels_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
#         print("=="*50)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state_dict = model.state_dict()
        
    if (epoch + 1) % 1000 == 0:
        torch.save(best_model_state_dict, f"/home/samyakr2/SHOP/weights/best_epoch_{epoch+1}.pth")
