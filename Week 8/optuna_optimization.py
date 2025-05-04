import sys

from utils import load_it_data, visualize_img
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, TensorDataset
import timm
from sklearn.metrics import r2_score
from PIL import Image

from sklearn.decomposition import PCA
#from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# path_to_data = "/home/nsuresh/NX-414_groupSRN/Week 7/"
path_to_data = "/home/scwang/BLCI_project/NX-414_groupSRN/Week 6/"
## Insert the folder where the data is, if you download in the same folder as this notebook then leave it blank

stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

def get_category(label):
    if 'face' in label:
        return 'face'
    elif 'table' in label:
        return 'table'
    elif 'ship' in label:
        return 'boat'
    elif 'car' in label:
        return 'car'
    elif 'chair' in label:
        return 'chair'
    elif 'airplane' in label:
        return 'plane'
    elif label in ['cow', 'lioness', 'dog', 'gorilla', 'turtle', 'elephant', 'hedgehog', 'bear']:  # animals
        return 'animal'
    elif label in ['watermelon', 'walnut', 'raspberry', 'pear', 'peach', 'apricot', 'apple', 'strawberry']:  # fruits
        return 'fruit'
    else:
        return 'unknown'
    
y_classes = [get_category(obj) for obj in objects_train]

y_classes_pair = [(obj,get_category(obj)) for obj in objects_train]
print(y_classes_pair.count("unknown"))
print(y_classes_pair)

others = [obj for obj in objects_train if get_category(obj) == 'unknown']
print(set(others))

y_classes_val = [get_category(obj) for obj in objects_val]

n_stimulus, n_channels, img_size, _ = stimulus_train.shape
print(_)
_, n_neurons = spikes_train.shape
# print(_)
print('The train dataset contains {} stimuli and {} IT neurons'.format(n_stimulus,n_neurons))
print('Each stimulus have {} channgels (RGB)'.format(n_channels))
print('The size of the image is {}x{}'.format(img_size,img_size))

num_classes = len(np.unique(y_classes))
num_classes

from torch.utils.data import Dataset
from PIL import Image
import torch

class MultiTaskDataset(Dataset):
    def __init__(self, stimuli, spikes, classes):
        self.stimuli = stimuli
        self.spikes = spikes
        self.classes = classes

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        # Load image
        img = self.stimuli[idx]

        spike_target = torch.tensor(self.spikes[idx], dtype=torch.float32)
        class_label = torch.tensor(self.classes[idx], dtype=torch.long)

        return img, spike_target, class_label

label_encoder = LabelEncoder()
encoded_classes = label_encoder.fit_transform(y_classes)
encoded_classes_val = label_encoder.transform(y_classes_val)

train_dataset = MultiTaskDataset(
    stimuli=stimulus_train,
    spikes=spikes_train,
    classes=encoded_classes,
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = MultiTaskDataset(
    stimuli=stimulus_val,
    spikes=spikes_val,
    classes=encoded_classes_val,
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

class multitaskNet(nn.Module):
    def __init__(self, resnet):
        super(multitaskNet, self).__init__()
        self.res = resnet
        # for param in self.res.parameters():
        #     param.requires_grad = False
        self.features = nn.Sequential(
            self.res.conv1,
            self.res.bn1,
            self.res.relu,
            self.res.maxpool,
            self.res.layer1,
            self.res.layer2,
            self.res.layer3  # <-- extract activations from here
        )

        # Get number of output channels from layer3
        dummy_input = torch.randn(1, 3, 224, 224)
        out = self.features(dummy_input)
        c, h, w = out.shape[1:]
        flattened = c * h * w

        # MLP to process the features into spike train
        self.spike_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 168),
            #nn.Tanh(),
            #nn.Linear(512, 168)
        )

        self.classifier_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        spikes = self.spike_predictor(x)
        cls_input = self.res.layer4(x)
        class_logits =  self.classifier_head(cls_input)
        return spikes, class_logits
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import optuna
import torch.nn.functional as F

def objective(trial):
    # Sample hyperparameters
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1.0)
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)

    # Model setup
    resnet = models.resnet50(pretrained=True)
    model = multitaskNet(resnet).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    spike_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    best_val_corr = -1.0
    num_epochs = 30  # or 50

    for epoch in range(num_epochs):
        model.train()
        for images, spike_targets, class_labels in train_loader:
            images, spike_targets, class_labels = images.to(device), spike_targets.to(device), class_labels.to(device)
            optimizer.zero_grad()
            pred_spikes, class_logits = model(images)
            loss = spike_criterion(pred_spikes, spike_targets) + alpha * class_criterion(class_logits, class_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate correlation on validation set
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for images, spikes, _ in val_loader:
            images, spikes = images.to(device), spikes.to(device)
            pred_spikes, _ = model(images)
            val_preds.append(pred_spikes.cpu().numpy())
            val_targets.append(spikes.cpu().numpy())
    
    y_pred = np.vstack(val_preds)
    y_true = np.vstack(val_targets)

    corr_vals = [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
    avg_corr = np.mean([c for c in corr_vals if not np.isnan(c)])

    return avg_corr

study = optuna.create_study(
    direction='maximize',
    study_name='multitask_net_optimization',
    storage='sqlite:///multitask_net_optimization.db',
    load_if_exists=True,
    )

study.optimize(objective, n_trials=30)

print("Best alpha:", study.best_params['alpha'])
print("Best learning rate:", study.best_params['lr'])
print("Best validation correlation:", study.best_value)