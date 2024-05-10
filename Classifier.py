#!/usr/bin/env python
# coding: utf-8

# # Lets import some things

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import time


# # Decide if cuda

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = True


# # Load dataset

# In[ ]:


batchsize = 30
num_classes = 102
learning_rate = 0.0004
num_epochs = 100


# In[ ]:


trainingData = datasets.Flowers102(
    root="data",
    split = "train",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomResizedCrop((256, 256),antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), 
        transforms.ColorJitter(contrast=0.05,brightness=0.05, saturation=0.05),
        transforms.RandomAffine(degrees=10,translate=(0.1,0.1),scale=(0.9,1.1),shear=0.15),
        transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1,1.)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
testData = datasets.Flowers102(
    root="data",
    split= "test",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
valData = datasets.Flowers102(
    root="data",
    split = "val",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)


# In[ ]:


print(f'training data has: {len(trainingData)} images')
print(f'validation data has: {len(valData)} images')
print(f'test data has: {len(testData)} images')


# # Get some dataloaders

# In[ ]:


train_dataloader = DataLoader(trainingData, batch_size=batchsize, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True,prefetch_factor=16)
test_dataloader = DataLoader(testData, batch_size=batchsize, shuffle=False, num_workers=4)
val_dataloader = DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=4)


# # Neural Network class

# In[ ]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(3,32,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#128x128
            #conv2
            nn.Conv2d(32,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#64x64
            #conv3
            nn.Conv2d(64,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#32x32
            #conv4
            nn.Conv2d(128,256,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#16x16
            #conv5
            nn.Conv2d(256,512,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#8x8
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024,num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        
    def forward(self, x):
        x= self.features(x)
        x= self.global_avg_pool(x)
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


# # Model = something

# In[ ]:


model = NeuralNet().to(device, non_blocking=True)
cost = nn.CrossEntropyLoss().to(device, non_blocking=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.1,verbose=True)


# # Actually do the training, needs to print less often

# In[ ]:


def train():
    loss_per_epoch = [] # clear all data incase retraining
    val_epochs = []
    quit_early_counter = 0
    last_epoch_loss = None
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
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
            
            running_loss += loss.item() #so this line is a sync point for the cpu and gpu commenting it out greatly reduces the training time (up to 25%  in my testing)
            batches +=1
        
        avg_loss = running_loss / batches
        loss_per_epoch.append(avg_loss)
        
        val_acc,val_loss = evaluate(val_dataloader)
        val_epochs.append(val_acc)
        scheduler.step(val_loss)
        
        if last_epoch_loss is not None and abs(last_epoch_loss - avg_loss)<0.01:
            quit_early_counter += 1
        else:
            quit_early_counter = 0
        last_epoch_loss = avg_loss
        
        epoch_length = time.time() - epoch_start
        print(f'Epoch: {epoch+1}, Avg Loss: {avg_loss:4f}, Num Batches: {batches}, Epoch Time: {epoch_length:.2f}, Validation Acc: {val_acc:.3f}%')
        
        if quit_early_counter >= 3:
            print('val acc isnt improving over last 5 so stop training')
            break
            
    return np.array(loss_per_epoch), np.array(val_epochs)


# # Plot the avg loss against epochs

# In[ ]:


def plot_array(array,name):
    plt.plot(array, label=f'{name}')
    # plt.plot(validation_epoch_losses,label='Validation Loss')   
    plt.legend()
    plt.show()


# # Display the training, testing, validation accuracy

# In[ ]:


def evaluate(dataloader):
    model.eval()
    correct = 0
    total =0
    total_loss =0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = cost(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            total_loss += loss.item() *labels.size(0)
        acc= 100*correct/total
        avg_loss = total_loss/total
    return acc, avg_loss


# In[ ]:


def all_eval():
    accval, _= evaluate(val_dataloader)
    print(f'val acc: {accval:.3f}%')
    acctest, _ = evaluate(test_dataloader)
    print(f'test acc: {acctest:.3f}%')
    trainacc, _ = evaluate(train_dataloader)
    print(f'train acc: {trainacc:.3f}%')


# # Save model

# In[ ]:


def save(pathname):
    torch.save(NeuralNet().state_dict(), f'{pathname}.pth')
    print(f'Saved model to {pathname}.pth')


# # Load model

# In[ ]:


def load(pathname,mod,dev):
    mod = NeuralNet().to(dev)
    mod.load_state_dict(torch.load(f'{pathname}.pth'))
    print(f'Loaded model from {pathname}.pth')


# # 

# In[ ]:


# def main():
#     pass


# In[ ]:


if __name__ == '__main__':
    training_epoch_losses, val_acc_per = train()
    plot_array(training_epoch_losses,'training epoch losses')
    plot_array(val_acc_per,'validation accuracy per epoch')
    all_eval()


# # Command line to convert this notebook to a python file, the reason is for readability of the code from github lol

# In[ ]:


# all_eval()


# In[22]:


def convert():
    get_ipython().system('jupyter nbconvert --to script Classifier.ipynb')


# In[21]:


convert()


# In[ ]:


# load('30e465-39&89',model,device)


# # todo possibly do the image display thing/ https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html / tune hyperparams

# In[ ]:




