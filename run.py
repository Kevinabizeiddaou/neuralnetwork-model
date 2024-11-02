#run.py
from model import NeuralNetwork, train, evaluate
from data_utils import BuildDataset
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize data loaders
train_loader, test_loader = BuildDataset(batch_size=16)

# Model parameters
input_size = 4  # Iris dataset features
hidden_size = 10
output_size = 3  # Iris classes

# Initialize model, criterion, and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, criterion, optimizer, train_loader, epochs=10)

# Evaluate the model
evaluate(model, test_loader)
