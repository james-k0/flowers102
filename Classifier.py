#!/usr/bin/env python
# coding: utf-8

# # Lets import some things

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from IPython import get_ipython
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import time
from torch.utils.tensorboard import SummaryWriter


# # I found a text file with the classnames

# In[ ]:


names = [ 'pink primrose','hard-leaved pocket orchid','canterbury bells','sweet pea','english marigold','tiger lily','moon orchid','bird of paradise','monkshood','globe thistle','snapdragon',"colt's foot",'king protea','spear thistle','yellow iris','globe-flower','purple coneflower','peruvian lily','balloon flower','giant white arum lily','fire lily','pincushion flower','fritillary','red ginger','grape hyacinth','corn poppy','prince of wales feathers','stemless gentian','artichoke','sweet william','carnation','garden phlox','love in the mist','mexican aster','alpine sea holly','ruby-lipped cattleya','cape flower','great masterwort','siam tulip','lenten rose','barbeton daisy','daffodil','sword lily','poinsettia','bolero deep blue','wallflower','marigold','buttercup','oxeye daisy','common dandelion','petunia','wild pansy','primula','sunflower','pelargonium','bishop of llandaff','gaura','geranium','orange dahlia','pink-yellow dahlia?','cautleya spicata','japanese anemone','black-eyed susan','silverbush','californian poppy','osteospermum','spring crocus','bearded iris','windflower','tree poppy','gazania','azalea','water lily','rose','thorn apple','morning glory','passion flower','lotus','toad lily','anthurium','frangipani','clematis','hibiscus','columbine','desert-rose','tree mallow','magnolia','cyclamen ','watercress','canna lily','hippeastrum ','bee balm','ball moss','foxglove','bougainvillea','camellia','mallow','mexican petunia','bromelia','blanket flower','trumpet creeper','blackberry lily']


# # Decide if cuda

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = True


# # Load dataset

# In[ ]:


trainingData = datasets.Flowers102(
    root="data",
    split = "train",
    download = True,
    transform = transforms.Compose([
        transforms.RandomResizedCrop((256, 256),scale=(0.9,1.0),),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), 
        transforms.ColorJitter(contrast=0.1,brightness=0.12, saturation=0.1,hue=0.05),#becaise the dataset will evidently be sensitve to colour might want to avoid this one
        transforms.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.9,1.1),shear=0.1),
        transforms.Resize((224,224)),
        transforms.ToTensor().to(device),
        transforms.Normalize([0.432, 0.381, 0.296], [0.258, 0.209, 0.221])
    ])
)
testData = datasets.Flowers102(
    root="data",
    split= "test",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor().to(device),
        transforms.Normalize([0.432, 0.381, 0.296], [0.258,  0.209, 0.221] )
    ])
)
valData = datasets.Flowers102(
    root="data",
    split = "val",
    download = True,
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor().to(device),
        transforms.Normalize([0.432, 0.381, 0.296], [0.258,  0.209, 0.221] )
    ])
)


# In[ ]:


print(f'training data has: {len(trainingData)} images')
print(f'validation data has: {len(valData)} images')
print(f'test data has: {len(testData)} images')


# # Neural Network class

# In[ ]:


class NeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            #2
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #4
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            #5
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #6
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        
        self.classifier = nn.Sequential(
            #7
            nn.Dropout(0.5),
            nn.Linear(512,num_classes), 
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))#6.5
        
    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x= self.classifier(x)
        return x


# # Training loop

# In[ ]:


def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):
    cost = nn.CrossEntropyLoss(label_smoothing=0.1).to(device) #choose the cost function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001) #choose the optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=8, factor=0.5, mode='min') #choose the scheduler
    
    best_val_acc = 0.0 #for if model stops on its not best epoch
    best_epoch = 0
    
    loss_per_epoch = [] #return values
    val_epochs = []
    tra_epochs = []
    
    quit_early_counter = 0 #quit early if model not improving or accuracy hits 100% for train set
    last_epoch_loss = None
    
    for epoch in range(num_epochs): #iterate over num_epochs
        epoch_start = time.time() #track how long epoch took, was just interested
        model.train() #some pytorch magic
        
        running_loss = 0.0
        batches = 0
        
        for i, (images, labels) in enumerate(train_dataloader): #per batch evaluate 
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = cost(outputs, labels)
            loss.backward() #the bit where it all happens
            optimizer.step()
            
            running_loss += loss.item()
            batches += 1
        
        avg_loss = running_loss / batches
        loss_per_epoch.append(avg_loss)
        
        val_acc, val_avg_loss = evaluate(model=model, dataloader=val_dataloader, device=device, cost=cost)
        val_epochs.append(val_acc)
        tra_acc, train_avg_loss = evaluate(model=model, dataloader=train_dataloader, device=device, cost=cost)
        tra_epochs.append(tra_acc)
        
        scheduler.step(val_avg_loss) #scheduler is there to reduce lr when begins to plateau
        
        
        
        if last_epoch_loss is not None and abs(last_epoch_loss - avg_loss) < 0.01:
            quit_early_counter += 1
        else:
            quit_early_counter = 0
        last_epoch_loss = avg_loss
        
        epoch_length = time.time() - epoch_start
        
        print(f'\nEpoch: {epoch+1}, Num Batches: {batches}, Avg Loss: {avg_loss:.4f}, Epoch Took: {epoch_length:.1f}s, Validation: {val_acc:.3f}% acc, {val_avg_loss:.3f} loss, Training: {tra_acc:.3f}% acc, {train_avg_loss:.3f} loss') #print out all info pe repoch
        
        if val_acc > best_val_acc: #save best model
            best_val_acc = val_acc
            best_epoch = epoch
            save(model, "best_model_checkpoint")
        
        if quit_early_counter >= 7: #quit early if needed
            print('Validation accuracy hasnt improved over the last 7 epochs. Stopping training.')
            break
    
    #training is now over
    print(f'Best validation accuracy: {best_val_acc:.3f}% at epoch {best_epoch+1}')
    print(f'Average loss: {avg_loss:.4f}, Training accuracy: {tra_acc:.3f}%, Validation accuracy: {val_acc:.3f}%')
    
    load(model, "best_model_checkpoint", device)  # Load the best model checkpoint
    all_eval(model=model, device=device, cost=nn.CrossEntropyLoss())  # Evaluate based on the best validation accuracy
    
    return np.array(loss_per_epoch), np.array(val_epochs), np.array(tra_epochs)


# # Function to plot some list

# In[ ]:


def plot_array(array,name):
    plt.plot(array, label=f'{name}')  
    plt.legend()
    plt.show()


# # Evaluate the model for some dataloader against some cost function

# In[ ]:


def evaluate(model, dataloader, device, cost):
    model.eval() # set model ot evalutation mode
    correct = 0
    total =0
    total_loss =0.0
    with torch.no_grad(): #so that it doesnt calculate gradients 
        for images, labels in dataloader: #gather images and labels for a batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = cost(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            total_loss += loss.item() *labels.size(0)
        acc= 100 *correct/total
        avg_loss = total_loss/total
    return acc, avg_loss


# # evaluate on test, validation and training data

# In[ ]:


def all_eval(model, device, cost):
    accval, _= evaluate(model=model,dataloader=val_dataloader,device=device,cost=cost)
    print(f'val acc: {accval:.3f}%')
    acctest, _ = evaluate(dataloader=test_dataloader,model=model,cost=cost,device=device)
    print(f'test acc: {acctest:.3f}%')
    trainacc, _ = evaluate(dataloader=train_dataloader,model=model,cost=cost,device=device)
    print(f'train acc: {trainacc:.3f}%')


# # Save model

# In[ ]:


def save(model, pathname):
    torch.save(model.state_dict(), f'{pathname}.pth')
    print(f'Saved model to {pathname}.pth')


# # Load model

# In[ ]:


def load(model, pathname ,device):
    model.load_state_dict(torch.load(f'{pathname}.pth'))
    model.to(device)
    print(f'Loaded model from {pathname}.pth')


# # Visualise samples

# In[ ]:


def visualize_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        sample_index = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[sample_index]

        image = image * torch.tensor([0.258, 0.209, 0.221]).view(3, 1, 1) + torch.tensor([0.432, 0.381, 0.296]).view(3, 1, 1)

        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].set_title(f'{names[label]}')
        axes[i].axis('off')

    plt.show()
visualize_samples(testData)


# # main loop

# In[ ]:


if __name__ == '__main__':
    #define hyper parameters
    BATCH_SIZE = 16
    NUM_CLASSES = 102
    LEARNING_RATE = 0.0015
    NUM_EPOCHS = 300
    #define dataloaders
    train_dataloader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, prefetch_factor=4,persistent_workers=True)
    test_dataloader = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, prefetch_factor=4,persistent_workers=True)
    val_dataloader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=False, num_workers=6,prefetch_factor=4,persistent_workers=True)
    # define model on cuda device
    model = NeuralNet(num_classes=NUM_CLASSES).to(device)
    
    def trainandplot():
        #train the model
        training_epoch_losses, val_acc_per, tra_acc_per = train(model=model,train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=device)
        #plot the data returned by the training loop
        plot_array(training_epoch_losses,'training epoch losses')
        plot_array(val_acc_per,'validation accuracy per epoch')
        plot_array(tra_acc_per,'training accuracy per epoch')
    # trainandplot()
    
    def loadandeval():
        load(model=model, pathname='65', device=device)
        all_eval(model=model, device=device, cost=nn.CrossEntropyLoss())
    # loadandeval()


# # Plot predicted top k for a random flower

# In[ ]:


def plot_pred(model, dataloader, names, device):
    model.eval() 
    
    img_total = len(dataloader.dataset) #so that it iterates over whole thing not one batch
    
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    rand_index = torch.randint(0, img_total, (1,)).item()
    image,label = dataloader.dataset[rand_index]
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top5_probs, top5_index = torch.topk(probs, 5) #obtain the top 5 predictions
    
    top5_probs = top5_probs[0].cpu().numpy()
    top5_index = top5_index[0].cpu().numpy()
    
    fig, ax = plt.subplots()
    image_display = image.cpu().squeeze().permute(1, 2, 0) 
    image_display = image_display * torch.tensor([0.258, 0.209, 0.221]).view(3, 1, 1) + torch.tensor([0.432, 0.381, 0.296]).view(3, 1, 1)
    image_display = image_display.numpy()
    ax.imshow(image_display)
    ax.set_title(f'{names[label]}')
    ax.axis('off')
    
    fig, ax = plt.subplots()
    sns.barplot(x=top5_probs, y=[names[i] for i in top5_index], ax=ax, palette='pastel')
    ax.set_title('Top 5 Predictions')
    ax.set_xlabel('Probability')
    plt.show()

# plot_pred(model, test_dataloader, names, device)


# # runs a command that converts this notebook to a py script

# In[21]:


def convert():
    get_ipython().system('jupyter nbconvert --to script Classifier.ipynb')


# In[ ]:


#convert()

