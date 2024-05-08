#!/usr/bin/env python
# coding: utf-8

# # Lets import some things

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms.v2 as transforms


# # Decide if cuda

# In[26]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# # Load dataset

# In[27]:


batchsize = 16
num_classes = 102
learning_rate = 0.001
num_epochs = 50


# In[28]:


trainingData = datasets.Flowers102(
    root="data",
    split = "train",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
)
testData = datasets.Flowers102(
    root="data",
    split= "test",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
)
valData = datasets.Flowers102(
    root="data",
    split = "val",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
)


# # Get some dataloaders

# In[29]:


train_dataloader = DataLoader(trainingData, batch_size=batchsize, shuffle=True, num_workers=4)
test_dataloader = DataLoader(testData, batch_size=batchsize, shuffle=True,num_workers=4)
val_dataloader = DataLoader(valData, batch_size=batchsize, shuffle=True,num_workers=4)


# # Neural Network class

# In[30]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*64*64 ,1024),
            nn.ReLU(),
            nn.Linear(1024,102),
        )
        
        
    def forward(self, x):
        x= self.features(x)
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
#conv way selected, this is a uhhh LeNet5 for MNIST so it doesnt work like yeah of course pretty placeholder


# # todo possibly do the image display thing/ save&load models/ https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html / tune hyperparams

# # Model = something

# In[31]:


model = NeuralNet().to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# # Actually do the training, needs to print less often

# In[32]:


for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = cost(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch: [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# # Display the training, testing, validation accuracy

# In[34]:


with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in train_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    acc = 100 * correct/total
print(f'Accuracy on train data: {acc}')
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    acc = 100 * correct/total
print(f'Accuracy on test data: {acc}')
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in val_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    acc = 100 * correct/total
print(f'Accuracy on val data: {acc}')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Classifier.ipynb')
