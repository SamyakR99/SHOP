import os
# import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import copy
import shutil
from clip import clip

from sklearn.metrics import accuracy_score
from pdb import set_trace as breakpoint


from SHOP.ARK.loads.foodseg103_segments import foodseg103

def build_dataset(data_root, data_split):
    print(' -------------------- Building Dataset ----------------------')
    return foodseg103(data_root, data_split, img_size = 224)

data_root = '/home/samyakr2/food/FoodSeg103/Images'
train_split = "train_seg"
val_split = "test_seg"
test_split = "test_seg"

train_dataset = build_dataset(data_root, train_split)
val_dataset = build_dataset(data_root, val_split)
test_dataset = build_dataset(data_root, test_split)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=80,
                                            shuffle=True,
                                            num_workers=3, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100,
                                            shuffle=False,
                                            num_workers=3, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                            shuffle=False,
                                            num_workers=3, pin_memory=True)

#################

# def backbone_params(self):
#     params = []
#     for name, param in self.named_parameters():
#         if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
#             params.append(param)
#     return params

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_to_cpu():
    url = clip._MODELS['RN101']
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict())

    return model

clip_model = load_clip_to_cpu()
clip_model.float()

params = []
for name, param in clip_model.named_parameters():
    if "visual" in name and 'attnpool' not in name:
        params.append(param)

for param in params:
    param.requires_grad_(False)

params2 = []
for name, param in clip_model.named_parameters():
    if 'attnpool' in name and 'visual' in name:
        params2.append(param)

for param in params2:
    param.requires_grad_(False)

for name, param in clip_model.named_parameters():
    if param.requires_grad:
        print(name)

breakpoint()

def get_features(dataloader):
    all_features_batches = []
    all_labels_batches = []
    all_img_names = []
    for (images, img_names), labels in dataloader:
        
        features, attn_weights  = clip_model.visual(images)
        all_features_batches.append(features.detach())
        all_labels_batches.append(labels)
        all_img_names.append(img_names)
    return all_features_batches, all_labels_batches, all_img_names

# # train_features, train_labels, train_img_names = get_features(train_loader)
# # val_features, val_labels, val_img_names = get_features(test_loader)

# train_features_path = "/home/samyakr2/SHOP/ARK/storage/CONV_foodseg103_train_clip_features_res.pt"
# train_labels_path = '/home/samyakr2/SHOP/ARK/storage/CONV_foodseg103_train_clip_labels_res.pt'
# train_img_names_path = '/home/samyakr2/SHOP/ARK/storage/CONV_foodseg103_train_img_name_res.pt'

# # torch.save(train_features, train_features_path)
# # torch.save(train_labels, train_labels_path)
# # torch.save(train_img_names, train_img_names_path)

# val_features_path = "/home/samyakr2/SHOP/ARK/storage/CONV_foodseg103_val_clip_features_res.pt"
# val_labels_path = '/home/samyakr2/SHOP/ARK/storage/CONV_foodseg103_val_clip_labels_res.pt'
# val_img_names_path = '/home/samyakr2/SHOP/ARK/storage/CONV_foodseg103_val_img_name_res.pt'

# # torch.save(val_features, val_features_path)
# # torch.save(val_labels, val_labels_path)
# # torch.save(val_img_names, val_img_names_path)

# train_features = torch.load(train_features_path)
# train_labels = torch.load(train_labels_path)
# train_img_name = torch.load(train_img_names_path)

# val_features = torch.load(val_features_path)
# val_labels = torch.load(val_labels_path)
# val_img_name = torch.load(val_img_names_path)

print("-----@@@@@@")
class clip_2fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(clip_2fc, self).__init__()
        
        # self.conv1 = nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, output_dim)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.relu(out)
        # breakpoint()
        out = self.conv2(x)
        out = self.relu(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

input_size = 512 
hidden_size = input_size // 2
num_classes = 103 

model = clip_2fc(input_size, hidden_size, num_classes).to(device)
device_count = torch.cuda.device_count()
if device_count > 1:
    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
    model = nn.DataParallel(model)


criterion = nn.CrossEntropyLoss()

params_to_optimize = list(model.parameters())
optimizer = torch.optim.Adam(params_to_optimize, lr=5e-4)  

best_loss = float('inf')
num_epochs = 500


# for (images, img_names), labels in train_loader:
#     img_feat, attn_weights  = clip_model.visual(images)

# import torch.optim.lr_scheduler as lr_scheduler
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  
def train_model(num_epochs, model, criterion, train_loader, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        all_outputs = []
        all_labels = []

        for (images, img_names), labels_batch in train_loader:
            
            features_batch  = clip_model.visual(images)
            # Flatten features batch
            # features_batch = features_batch.to(torch.float32)
            # features_batch = features_batch / features_batch.norm(dim=1, keepdim=True)
            # Convert labels to tensor
            labels_tensor = labels_batch.type(torch.float32).to(device)#torch.tensor(labels_batch, dtype=torch.float32)#
            labels_tensor = labels_tensor.squeeze(dim=1)
            # Forward pass
            outputs = model(features_batch.to(device))
            # print("Why is this not printing")
            # breakpoint()
            loss = criterion(outputs, labels_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                    
                outputs_np = outputs.argmax(dim=1).cpu().detach().numpy()
                labels_np = labels_tensor.argmax(dim=1).cpu().detach().numpy()
                
                all_outputs.append(outputs_np)
                all_labels.append(labels_np)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state_dict = model.state_dict()

        if (epoch + 1) % 10 == 0:
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)
            acc = accuracy_score(all_labels, all_outputs)
            print(f"accuracy_score: {acc}")
            torch.save(best_model_state_dict, f"/home/samyakr2/SHOP/ARK/weights_storage/conv/epoch_{epoch+1}.pth")
        # scheduler.step()

# best_model_state_dict = torch.load("/home/samyakr2/SHOP/ARK/weights_storage/epoch_{}.pth".format(1000))
# model.load_state_dict(best_model_state_dict)
train_model(num_epochs, model, criterion, train_loader, device)


# def test_model(model, features_batches, labels_batches, val_img_names, device):
#     model.eval()  # Set the model to evaluation mode
#     all_labels = []
#     all_outputs = []
#     multilabel_dict = {}
#     with torch.no_grad():  # Disable gradient computation
#         for features_batch, labels_batch, val_img_name in zip(features_batches, labels_batches, val_img_names):
#             # Move batch to device
#             breakpoint()

#             features_batch = features_batch.to(device)
#             labels_tensor = labels_batch.type(torch.float32).to(device)
#             labels_tensor = labels_tensor.squeeze(dim=1)
#             features_batch = features_batch.view(features_batch.size(0), -1).to(torch.float32)
#             outputs = model(features_batch)
#             for idy in range (len(val_img_name)):
#                 multilabel_dict[val_img_name[idy]] = outputs[idy]
#             # Convert outputs and labels to numpy arrays
#             outputs_np = outputs.argmax(dim=1).cpu().detach().numpy()
#             labels_np = labels_tensor.argmax(dim=1).cpu().detach().numpy()

#             all_outputs.append(outputs_np)
#             all_labels.append(labels_np)
    
#     torch.save(multilabel_dict, '/home/samyakr2/SHOP/ARK/output/multilabel_dict.pth')

#     # Concatenate outputs and labels
#     all_outputs = np.concatenate(all_outputs)
#     all_labels = np.concatenate(all_labels)
#     # Compute average precision score
#     acc = accuracy_score(all_labels, all_outputs)
#     print(f"accuracy_score: {acc}")

# # best_model_state_dict = torch.load("/home/samyakr2/SHOP/ARK/weights_storage/epoch_{}.pth".format(490))
# # model.load_state_dict(best_model_state_dict)
# # test_model(model, val_features, val_labels, val_img_name, device)

# # for i in range (10, 3001, 1000):
    
# #     best_model_state_dict = torch.load("/home/samyakr2/SHOP/ARK/weights_storage/epoch_{}.pth".format(i))
# #     model.load_state_dict(best_model_state_dict)
# #     test_model(model, val_features, val_labels, val_img_name,device)
