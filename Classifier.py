#!/usr/bin/env python
# coding: utf-8

# # Lets import some things

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


# # Decide if cuda

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = True


# # Load dataset

# In[3]:


batchsize = 64
num_classes = 102
learning_rate = 0.0015
num_epochs = 50


# In[4]:


trainingData = datasets.Flowers102(
    root="data",
    split = "train",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
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

# In[5]:


train_dataloader = DataLoader(trainingData, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(testData, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)


# # Neural Network class

# In[6]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#half size
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#half size
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*64*64 ,1024),
            nn.ReLU(),
            nn.Dropout(0.2), #add a bit of randomness for some fun  + generality
            nn.Linear(1024,102),
        )
        
        
    def forward(self, x):
        x= self.features(x)
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
# i should stop drinking because where did i imagine this up !?


# # Model = something

# In[7]:


model = NeuralNet().to(device,non_blocking=True)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# # Actually do the training, needs to print less often

# In[ ]:


def train():
    for epoch in range(num_epochs):
        running_loss = 0.0
        batches = 0
        for i, (images,labels) in enumerate(train_dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            optimizer.zero_grad()
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batches +=1
        print(f'Epoch: {epoch+1}, Avg Loss: {running_loss/batches:4f}, Num Batches: {batches}')
train()
# next steps: the running loss is a bit jank since batches is always 16 it will print the same batch, should be shuffled though but doesnt look like it ? 
# maybe im confusing shuffles with individual weights


# # Display the training, testing, validation accuracy

# In[ ]:


def evaluate(dataloader):
    model.eval()
    correct = 0
    total =0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        acc= float(100*correct/total)
    return acc
print(f'val acc: {evaluate(val_dataloader):.3f}%')
print(f'test acc: {evaluate(test_dataloader):.3f}%')
print(f'train acc: {evaluate(train_dataloader):.3f}%')


# # Save model

# In[ ]:


def save(pathname):
    torch.save(NeuralNet().state_dict(), f'{pathname}.pth')
    print(f'Saved model to {pathname}.pth')


# # Load model

# In[ ]:


def load(pathname):
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(f'{pathname}.pth'))
    print(f'Loaded model from {pathname}.pth')


# # Command line to convert this notebook to a python file, the reason is for readability of the code from github lol

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Classifier.ipynb')


# # todo possibly do the image display thing/ https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html / tune hyperparams
