import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first_lin=nn.Linear(784,512)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(p=0.2)
        self.project=nn.Linear(512,10)
        self.sig=nn.Sigmoid()
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x=self.first_lin(images)
        x=self.relu(x)
        x=self.drop(x)
        x=self.project(x)
        x=self.sig(x)
        return torch.round(x,decimals=4)

        
