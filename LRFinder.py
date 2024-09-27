import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder  # Import LRFinder


# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.func = -5 * self.x + 1
        self.y = self.func + 0.4 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Getting length of the data
    def __len__(self):
        return self.len


# Create dataset object
data_set = Build_Data()

# Define model, criterion, and dataloader
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()

# Creating Dataloader object
trainloader = DataLoader(dataset=data_set, batch_size=1)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Using LRFinder to find optimal learning rate
lr_finder = LRFinder(
    model, optimizer, criterion, device="cpu"
)  # Use "cuda" if using GPU

# Run the learning rate finder over a range of learning rates
lr_finder.range_test(trainloader, end_lr=1, num_iter=100)

# Plot the learning rate vs. loss graph
lr_finder.plot()  # Inspect the plot to find the optimal learning rate
lr_finder.reset()  # Reset the model and optimizer to initial state

# Re-define model and optimizer based on the best learning rate
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Choose lr based on plot

# Initialize lists to store losses
loss_SGD = []
n_iter = 20

# Training loop using SGD
for i in range(n_iter):
    for x, y in trainloader:
        # Making a prediction in forward pass
        y_hat = model(x)
        # Calculating the loss
        loss = criterion(y_hat, y)
        # Store loss
        loss_SGD.append(loss.item())
        # Zeroing gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()
