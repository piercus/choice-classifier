import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np

# Custom MNIST dataset with choices
class MNIST(Dataset):
    def __init__(self, mnist_dataset, num_choices, choice_features_size):
        self.mnist_dataset = mnist_dataset
        self.num_choices = num_choices
        self.choice_features_size = choice_features_size

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]

        # Create one-hot encoded label
        one_hot_label = np.zeros(self.choice_features_size)
        one_hot_label[label] = 1

        # Generate num_choices - 1 random labels different from the correct label
        random_labels = np.random.choice([i for i in range(self.choice_features_size) if i != label], self.num_choices - 1, replace=False)

        # Create choices
        choices = [one_hot_label]
        for r_label in random_labels:
            one_hot_r_label = np.zeros(10)
            one_hot_r_label[r_label] = 1
            choices.append(one_hot_r_label)

        # Shuffle choices and store the index of the correct choice
        
        correct_index = np.random.choice(self.num_choices, 1, replace=False)
        
        indexes = np.random.permutation(self.num_choices)
        correct_index = np.where(indexes == 0)[0][0]
                
        shuffled_choices = np.array([choices[i] for i in indexes])
        choices = torch.tensor(shuffled_choices, dtype=torch.float32)

        # One-hot encode the index of the correct choice
        ground_truth = torch.zeros(3)
        ground_truth[correct_index] = 1
        return image, choices, ground_truth, torch.tensor(one_hot_label, dtype=torch.float32)
