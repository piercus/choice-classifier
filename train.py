import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
from choicecls import MNIST, Classifier

# Load MNIST dataset
mnist_transform = transforms.Compose([
    transforms.ToTensor()
])
num_choices = 3;
choice_features_size= 10

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
custom_mnist_dataset = MNIST(mnist_dataset, num_choices, choice_features_size)

# Split dataset into training and validation sets
train_size = int(0.8 * len(custom_mnist_dataset))
val_size = len(custom_mnist_dataset) - train_size
train_dataset, val_dataset = random_split(custom_mnist_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier(num_choices, choice_features_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, choices, ground_truth, label_gt in train_loader:
        images, choices, ground_truth, label_gt = images.to(device), choices.to(device), ground_truth.to(device), label_gt.to(device)

        optimizer.zero_grad()
        probabilities = model(images, choices)
        # print('dtypes', probabilities.dtype, ground_truth.dtype, label_pred.dtype, label_gt.dtype)
        # print('shapes', probabilities.shape, ground_truth.shape, label_pred.shape, label_gt.shape)
        loss = criterion(probabilities, ground_truth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average training loss for this epoch
    average_loss = running_loss / len(train_loader)
    
    # Validate the model
    model.eval()
    val_running_loss = 0.0
    num_correct1 = 0
    num_correct2 = 0
    num_samples = 0
    with torch.no_grad():
        for images, choices, ground_truth, gt_label in val_loader:
            images, choices, ground_truth, gt_label = images.to(device), choices.to(device), ground_truth.to(device), gt_label.to(device)
            
            probabilities = model(images, choices)
            loss = criterion(probabilities, ground_truth)
            val_running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_indices1 = torch.max(probabilities, 1)
            _, true_indices1 = torch.max(ground_truth, 1)
            num_correct1 += (predicted_indices1 == true_indices1).sum().item()
            
            num_samples += images.size(0)

    val_average_loss = val_running_loss / len(val_loader)
    accuracy1 = num_correct1 / num_samples * 100
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}, Validation Loss: {val_average_loss:.4f}, Accuracy Choice: {accuracy1:.2f}%')
