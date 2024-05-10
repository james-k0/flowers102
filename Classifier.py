#!/usr/bin/env python
# coding: utf-8

# # Lets import some things

# In[16]:


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

# In[17]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = True


# # Load dataset

# In[18]:


batchsize = 16
num_classes = 102
learning_rate = 0.001
num_epochs = 100


# In[19]:


trainingData = datasets.Flowers102(
    root="data",
    split = "train",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        # transforms.RandomResizedCrop((256, 256),antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30), 
        transforms.ColorJitter(contrast=0.1,hue=0.1,brightness=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


# In[20]:


print(f'training data has: {len(trainingData)} images')
print(f'validation data has: {len(valData)} images')
print(f'test data has: {len(testData)} images')


# # Get some dataloaders

# In[21]:


train_dataloader = DataLoader(trainingData, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=6)
test_dataloader = DataLoader(testData, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)


# # Neural Network class

# In[22]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#half size stride defaults to kernel size
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*32*32,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,102),
            nn.LogSoftmax(dim=1)
        )
        
        
    def forward(self, x):
        x= self.features(x)
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


# # Model = something

# In[23]:


model = NeuralNet().to(device, non_blocking=True)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# # Actually do the training, needs to print less often

# In[24]:


def train():
    loss_per_epoch = [] # clear all data incase retraining
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
        loss_per_epoch.append(running_loss / batches)
        print(f'Epoch: {epoch+1}, Avg Loss: {running_loss/batches:4f}, Num Batches: {batches}')
    return loss_per_epoch


# # Plot the avg loss against epochs

# In[25]:


def plot_train(epoch_losses):
    plt.plot(epoch_losses, label='Training Loss')
    # plt.plot(validation_epoch_losses,label='Validation Loss')   
    plt.legend()
    plt.show()


# # Display the training, testing, validation accuracy

# In[26]:


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
# print(f'val acc: {evaluate(val_dataloader):.3f}%')
# print(f'test acc: {evaluate(test_dataloader):.3f}%')
# print(f'train acc: {evaluate(train_dataloader):.3f}%')


# In[27]:


def all_eval():
    print(f'val acc: {evaluate(val_dataloader):.3f}%')
    print(f'test acc: {evaluate(test_dataloader):.3f}%')
    print(f'train acc: {evaluate(train_dataloader):.3f}%')


# # Save model

# In[28]:


def save(pathname):
    torch.save(NeuralNet().state_dict(), f'{pathname}.pth')
    print(f'Saved model to {pathname}.pth')


# # Load model

# In[29]:


def load(pathname):
    # model = NeuralNet().to(device)
    model.load_state_dict(torch.load(f'{pathname}.pth'))
    print(f'Loaded model from {pathname}.pth')


# In[30]:


# def main():
#     pass


# In[ ]:


if __name__ == '__main__':
    training_epoch_losses = train()
    plot_train(training_epoch_losses)
    all_eval()


# # Command line to convert this notebook to a python file, the reason is for readability of the code from github lol

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Classifier.ipynb')


# # todo possibly do the image display thing/ https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html / tune hyperparams

# In[ ]:




