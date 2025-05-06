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

print("is working")

import timm
import torch
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader



class ITNeuralPredictor:
    def __init__(self, 
                 num_neurons=168, 
                 model_name='convnext_large_mlp.clip_laion2b_augreg_ft_in1k',
                 feature_extraction_only=True,
                 device=None):
        """
        A model to predict IT neural responses to visual stimuli
        
        Args:
            num_neurons: Number of output neurons to predict
            model_name: Name of the pretrained model to use as feature extractor
            feature_extraction_only: If True, only use the model for feature extraction without fine-tuning
            device: Device to run the model on (will use CUDA if available when None)
        """
        self.num_neurons = num_neurons
        self.model_name = model_name
        self.feature_extraction_only = feature_extraction_only
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self._build_model()
            
    def _build_model(self):
        """Build the neural prediction model"""
        # Load the backbone model
        print(f"Loading pretrained model: {self.model_name}")
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0  # Remove classifier to get features
        )
        
        # Freeze the backbone if using for feature extraction only
        if self.feature_extraction_only:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Put backbone in eval mode for feature extraction
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # Get feature dimension by doing a forward pass with dummy data
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            print(f"Feature dimension: {self.feature_dim}")
        
        # Create a regression head to predict neural activity
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_neurons)
        ).to(self.device)
        
    def train_model(self, 
              stimulus_train, 
              spikes_train, 
              stimulus_val=None,
              spikes_val=None,
              batch_size=32, 
              num_epochs=50, 
              learning_rate=0.001,
              weight_decay=1e-5,
              patience=10):
        """
        Train the model to predict neural responses
        
        Args:
            stimulus_train: Numpy array of training images (N, C, H, W)
            spikes_train: Numpy array of neural responses (N, num_neurons)
            stimulus_val: Numpy array of validation images
            spikes_val: Numpy array of validation responses
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            
        Returns:
            Dictionary of training history
        """
        # Convert data to PyTorch tensors
        train_images = torch.tensor(stimulus_train, dtype=torch.float32)
        train_responses = torch.tensor(spikes_train, dtype=torch.float32)
        
        if stimulus_val is not None and spikes_val is not None:
            val_images = torch.tensor(stimulus_val, dtype=torch.float32)
            val_responses = torch.tensor(spikes_val, dtype=torch.float32)
            has_validation = True
        else:
            has_validation = False
        
        # Create data loaders
        train_dataset = TensorDataset(train_images, train_responses)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if has_validation:
            val_dataset = TensorDataset(val_images, val_responses)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.regression_head.parameters(), 
                               lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_corr': [],
            'val_ev': []
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.backbone.eval()  # Keep backbone in eval mode if feature extraction only
            self.regression_head.train()
            
            train_loss = 0.0
            for batch_imgs, batch_responses in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_imgs = batch_imgs.to(self.device)
                batch_responses = batch_responses.to(self.device)
                
                # Extract features (with no_grad if feature extraction only)
                if self.feature_extraction_only:
                    with torch.no_grad():
                        features = self.backbone(batch_imgs)
                else:
                    features = self.backbone(batch_imgs)
                
                # Forward pass through regression head
                predictions = self.regression_head(features)
                loss = criterion(predictions, batch_responses)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_imgs.size(0)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            if has_validation:
                self.backbone.eval()
                self.regression_head.eval()
                
                val_loss = 0.0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_imgs, batch_responses in val_loader:
                        batch_imgs = batch_imgs.to(self.device)
                        batch_responses = batch_responses.to(self.device)
                        
                        features = self.backbone(batch_imgs)
                        predictions = self.regression_head(features)
                        
                        val_loss += criterion(predictions, batch_responses).item() * batch_imgs.size(0)
                        
                        all_preds.append(predictions.cpu().numpy())
                        all_targets.append(batch_responses.cpu().numpy())
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                # Calculate correlation and explained variance
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                
                # Calculate correlation for each neuron
                corrs = []
                for n in range(self.num_neurons):
                    corr = np.corrcoef(all_preds[:, n], all_targets[:, n])[0, 1]
                    corrs.append(corr if not np.isnan(corr) else 0)
                
                # Calculate explained variance for each neuron
                evs = []
                for n in range(self.num_neurons):
                    ev = explained_variance_score(all_targets[:, n], all_preds[:, n])
                    evs.append(ev if not np.isnan(ev) else 0)
                
                mean_corr = np.mean(corrs)
                mean_ev = np.mean(evs)
                
                history['val_corr'].append(mean_corr)
                history['val_ev'].append(mean_ev)
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Mean Corr: {mean_corr:.4f}, Mean EV: {mean_ev:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save the best model
                    self.save_model('best_model.pth')
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping after {epoch+1} epochs")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Load the best model if validation was used
        if has_validation and os.path.exists('best_model.pth'):
            self.load_model('best_model.pth')
        
        return history
    
    def predict(self, stimulus):
        """
        Predict neural responses for a batch of images
        
        Args:
            stimulus: Numpy array of images (N, C, H, W)
            
        Returns:
            Numpy array of predicted neural responses
        """
        # Convert to tensor if numpy array
        if isinstance(stimulus, np.ndarray):
            stimulus = torch.tensor(stimulus, dtype=torch.float32)
            
        self.backbone.eval()
        self.regression_head.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(stimulus), 32):  # Process in batches of 32
                batch = stimulus[i:i+32].to(self.device)
                features = self.backbone(batch)
                batch_preds = self.regression_head(features)
                predictions.append(batch_preds.cpu().numpy())
                
        return np.vstack(predictions)
    
    def evaluate(self, stimulus, spikes):
        """
        Evaluate the model on a test set
        
        Args:
            stimulus: Numpy array of images (N, C, H, W)
            spikes: Numpy array of neural responses (N, num_neurons)
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(stimulus)
        
        # Calculate correlation for each neuron
        corrs = []
        for n in range(self.num_neurons):
            corr = np.corrcoef(predictions[:, n], spikes[:, n])[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0)
        
        # Calculate explained variance for each neuron
        evs = []
        for n in range(self.num_neurons):
            ev = explained_variance_score(spikes[:, n], predictions[:, n])
            evs.append(ev if not np.isnan(ev) else 0)
        
        results = {
            'correlations': np.array(corrs),
            'explained_variances': np.array(evs),
            'mean_correlation': np.mean(corrs),
            'median_correlation': np.median(corrs),
            'mean_explained_variance': np.mean(evs),
            'median_explained_variance': np.median(evs)
        }
        
        return results
    
    def save_model(self, filepath):
        """Save the model to a file"""
        torch.save({
            'backbone_state_dict': self.backbone.state_dict() if not self.feature_extraction_only else None,
            'regression_head_state_dict': self.regression_head.state_dict(),
            'feature_extraction_only': self.feature_extraction_only,
            'model_name': self.model_name,
            'num_neurons': self.num_neurons
        }, filepath)
        
    def load_model(self, filepath):
        """Load the model from a file"""
        checkpoint = torch.load(filepath)
        
        if not self.feature_extraction_only and checkpoint['backbone_state_dict'] is not None:
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            
        self.regression_head.load_state_dict(checkpoint['regression_head_state_dict'])
        
    def plot_training_history(self, history):
        """Plot the training history"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot correlation
        if 'val_corr' in history and len(history['val_corr']) > 0:
            plt.subplot(1, 3, 2)
            plt.plot(history['val_corr'])
            plt.xlabel('Epoch')
            plt.ylabel('Mean Correlation')
            plt.title('Validation Correlation')
        
        # Plot explained variance
        if 'val_ev' in history and len(history['val_ev']) > 0:
            plt.subplot(1, 3, 3)
            plt.plot(history['val_ev'])
            plt.xlabel('Epoch')
            plt.ylabel('Mean Explained Variance')
            plt.title('Validation Explained Variance')
        
        plt.tight_layout()
        plt.show()
        
    def plot_neuron_metrics(self, results, neuron_indices=None):
        """
        Plot metrics for each neuron
        
        Args:
            results: Dictionary of evaluation results
            neuron_indices: Optional list of neuron indices to highlight
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot correlations
        axes[0].hist(results['correlations'], bins=20)
        axes[0].axvline(results['mean_correlation'], color='r', linestyle='dashed', linewidth=2, 
                      label=f'Mean: {results["mean_correlation"]:.4f}')
        axes[0].axvline(results['median_correlation'], color='g', linestyle='dashed', linewidth=2,
                      label=f'Median: {results["median_correlation"]:.4f}')
        axes[0].set_xlabel('Correlation')
        axes[0].set_ylabel('Number of Neurons')
        axes[0].set_title('Correlation Distribution')
        axes[0].legend()
        
        # Plot explained variances
        axes[1].hist(results['explained_variances'], bins=20)
        axes[1].axvline(results['mean_explained_variance'], color='r', linestyle='dashed', linewidth=2,
                      label=f'Mean: {results["mean_explained_variance"]:.4f}')
        axes[1].axvline(results['median_explained_variance'], color='g', linestyle='dashed', linewidth=2,
                      label=f'Median: {results["median_explained_variance"]:.4f}')
        axes[1].set_xlabel('Explained Variance')
        axes[1].set_ylabel('Number of Neurons')
        axes[1].set_title('Explained Variance Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # If specific neuron indices are provided, show their metrics
        if neuron_indices is not None:
            print("Performance for selected neurons:")
            for idx in neuron_indices:
                print(f"Neuron {idx}: Correlation = {results['correlations'][idx]:.4f}, "
                      f"Explained Variance = {results['explained_variances'][idx]:.4f}")










path_to_data = '' ## Insert the folder where the data is, if you download in the same folder as this notebook then leave it blank

from utils import load_it_data, visualize_img
#stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)


stimulus_val = torch.tensor(stimulus_val, dtype=torch.float32)
spikes_val = torch.tensor(spikes_val, dtype=torch.float32)

val_dataset = TensorDataset(stimulus_val, spikes_val)
val_loader = DataLoader(val_dataset, batch_size=64)



#model = torch.load('it_neural_predictor.pth', map_location=torch.device('cpu'))
#state_dict = torch.load('it_neural_predictor.pth', map_location=torch.device('cpu'))

model = ITNeuralPredictor()

loaded_dict = torch.load('it_neural_predictor.pth')

# Print the keys to see what's inside
print(loaded_dict.keys())

# Apply the loaded state dictionary to your model
model.load_state_dict(state_dict)


model.eval()
val_all = None 
y_all = None

with torch.no_grad():
    for X_val, y_val in val_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        val_outputs = model(X_val)
        val_np = val_outputs.detach().cpu().numpy()
        y_np = y_val.detach().cpu().numpy()
        if val_all is None:
            val_all = val_np
            y_all = y_np
        else:
            val_all = np.concatenate((val_all,val_np),axis=0)
            y_all = np.concatenate((y_all,y_np),axis=0)

correlations = []
for i in range(y_all.shape[1]):
    corr, _ = pearsonr(y_all[:, i], val_all[:, i])
    correlations.append(corr)
avg_corr_val = np.mean(correlations)

ev_val = explained_variance_score(y_all, val_all)

print(f"Avg correlation of validation outputs and ground truth for the best model = {avg_corr_val}")
print(f"Explained variance of validation outputs and ground truth for the best model = {ev_val}")
