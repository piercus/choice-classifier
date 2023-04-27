import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

from .query_head import QueryHead

class Net(nn.Module):
    def __init__(self, num_img_ftrs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, num_img_ftrs)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

class Classifier(nn.Module):
    def __init__(self, choice_features_size, choice_encoding_size=4,num_img_ftrs=128, num_heads=1):
        super(Classifier, self).__init__()
        self.choice_features_size = choice_features_size
        self.choice_encoding_size = choice_encoding_size
        self.num_img_ftrs = num_img_ftrs

        # Image feature extraction using ResNet
        self.net = Net(num_img_ftrs)
        # Choice processing using MLP
        self.choice_mlp = nn.Linear(self.choice_features_size, self.choice_encoding_size)
        self.query = QueryHead(dim_context=num_img_ftrs, dim_query=self.choice_encoding_size)

    def forward(self, image, choices):
        # Extract image features
        image_feat = self.net(image)
        # print('choices', choices.shape)
        choice_encoded = self.choice_mlp(choices)
        return self.query(choice_encoded, image_feat)

        # image_key = image_feat.unsqueeze(1)
        # image_features = image_features.view(image_features.size(0), -1)
        # image_value = image_key
        
        # Process choices
        #choice_encoded = self.choice_mlp(choices)
        # Apply attention mechanism


        # Classify the combined features
        # probabilities = self.final(attention)
        # print('image_feat', image_feat.shape)
        label_pred = self.direct_label(image_feat)
        # print('label_pred', label_pred.shape)
        
        # print('choices', choices.shape)
        # print('label_pred', label_pred.shape)   
        # print('image_feat', image_feat.shape)   
               
        # attention = self.attention(choices, image_key)
        # print('attention', attention.shape)
        # probs = self.final(attention)
        a = label_pred.unsqueeze(2)
        # print('choices', choices.shape)
        # print('a', a.shape)
        probs = torch.matmul(choices, a).squeeze(2)
        # print('probs', probs.shape)
