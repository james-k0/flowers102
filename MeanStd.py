import torch
from torchvision import datasets,transforms

data_transform = transforms.Compose([
    transforms.Resize((224,224)), #assuming that the transform to 224 is consistent with ur preprocessing
    transforms.ToTensor()
])

dataset = datasets.Flowers102(root='data',transform=data_transform,split='train')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
channels = 3
mean = torch.zeros(channels)
std = torch.zeros(channels)

for image, _ in dataset:
    mean+= torch.mean(image, dim=(1,2))
    std+= torch.std(image,dim=(1,2))
mean/= len(dataset)
std/= len(dataset)

print('Mean', mean.cpu().numpy())
print('Std', std.cpu().numpy())