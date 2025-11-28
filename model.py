import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl

class EnhancedCNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Increased channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Added layer
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),  # Increased hidden units
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model_params(model):
    """Get model parameters as numpy arrays"""
    with torch.no_grad():
        return [val.cpu().numpy() for val in model.state_dict().values()]

def set_model_params(model, params):
    """Set model parameters from numpy arrays or Flower Parameters object"""
    if isinstance(params, fl.common.Parameters):
        params = fl.common.parameters_to_ndarrays(params)
    
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.from_numpy(np.array(v)) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
