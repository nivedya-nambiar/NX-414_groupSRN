from utils import load_it_data, visualize_img
import matplotlib.pyplot as plt
import numpy as np
import gdown

path_to_data = '' ## Insert the folder where the data is, if you download in the same folder as this notebook then leave it blank

stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)


import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

print("is working")

import timm
import torch
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


from torch.utils.data import DataLoader, TensorDataset
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
from utils import load_it_data, visualize_img, save_pkl, load_pkl


stimulus_train_tensor = torch.from_numpy(stimulus_train).float()

dataset = TensorDataset(stimulus_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

stimulus_val_tensor = torch.from_numpy(stimulus_val).float()
dataset_val = TensorDataset(stimulus_val_tensor)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

# Load a pre-trained EfficientNet model
efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.eval()  # set to evaluation mode

# Dictionary to store activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# EfficientNet layer structure is different from ResNet
# Here's the mapping of layers from ResNet to EfficientNet:
# ResNet conv1 -> EfficientNet features[0] (conv_stem)
# ResNet layer1 -> EfficientNet features[1] (MBConv block1)
# ResNet layer2 -> EfficientNet features[2] (MBConv block2)
# ResNet layer3 -> EfficientNet features[4] (MBConv block4)
# ResNet layer4 -> EfficientNet features[6] (MBConv block6)
# ResNet avgpool -> EfficientNet avgpool

# Register hooks for the corresponding layers in EfficientNet
efficientnet.features[0].register_forward_hook(get_activation('conv_stem'))  # Equivalent to conv1
efficientnet.features[1].register_forward_hook(get_activation('block1'))     # Equivalent to layer1
efficientnet.features[2].register_forward_hook(get_activation('block2'))     # Equivalent to layer2
efficientnet.features[4].register_forward_hook(get_activation('block4'))     # Equivalent to layer3
efficientnet.features[6].register_forward_hook(get_activation('block6'))     # Equivalent to layer4
efficientnet.avgpool.register_forward_hook(get_activation('avgpool'))       # Same as ResNet avgpool

# For validation data
activations_val = {}

def get_activation_val(name):
    def hook_val(model, input, output):
        activations_val[name] = output.detach()
    return hook_val

# Register the same hooks for validation
efficientnet.features[0].register_forward_hook(get_activation_val('conv_stem'))
efficientnet.features[1].register_forward_hook(get_activation_val('block1'))
efficientnet.features[2].register_forward_hook(get_activation_val('block2'))
efficientnet.features[4].register_forward_hook(get_activation_val('block4'))
efficientnet.features[6].register_forward_hook(get_activation_val('block6'))
efficientnet.avgpool.register_forward_hook(get_activation_val('avgpool'))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

efficientnet = efficientnet.to(device)
stimulus_train_tensor = stimulus_train_tensor.to(device)

# Define the layers to capture
layers = ['conv_stem', 'block1', 'block2', 'block4', 'block6', 'avgpool']

# Extract and save activations for training data
for layer in layers:
    all_activations_layer = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            _ = efficientnet(inputs)  # triggers hooks
            
            batch_activations = activations[layer]
            flattened = batch_activations.flatten(start_dim=1).cpu().numpy()
            all_activations_layer.append(flattened)

    all_activations_layer = np.vstack(all_activations_layer)
    save_pkl(all_activations_layer, f"all_activations_efficientnet_{layer}.pkl")

# Extract and save activations for validation data
efficientnet = efficientnet.to(device)
stimulus_val_tensor = stimulus_val_tensor.to(device)

for layer in layers:
    all_activations_layer = []
    with torch.no_grad():
        for batch in dataloader_val:
            inputs = batch[0].to(device)
            _ = efficientnet(inputs)  # triggers hooks
            
            batch_activations = activations_val[layer]
            flattened = batch_activations.flatten(start_dim=1).cpu().numpy()
            all_activations_layer.append(flattened)

    all_activations_layer = np.vstack(all_activations_layer)
    save_pkl(all_activations_layer, f"all_activations_efficientnet_{layer}_val.pkl")

# Run PCA on the activations
for layer in layers:
    # Load pre-saved activation (NumPy array)
    pca = PCA(n_components=1000)
    activations = load_pkl(f"all_activations_efficientnet_{layer}.pkl")
    activations_val = load_pkl(f"all_activations_efficientnet_{layer}_val.pkl")
    X_pca = pca.fit_transform(activations)
    X_pca_val = pca.transform(activations_val)
    # Save PCA components
    save_pkl(pca, f"pca_efficientnet_{layer}.pkl")
    save_pkl(X_pca, f"X_pca_efficientnet_{layer}.pkl")
    save_pkl(X_pca_val, f"X_pca_efficientnet_{layer}_val.pkl")

