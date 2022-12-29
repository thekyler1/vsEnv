import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn

class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        slicer = slice(-9)
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0][slicer], self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(image)
        return image, label

path_trainSet = f"/Users/necatiisik/lfw_dataset/pairsDevTrain.txt"
path_testSet = f"/Users/necatiisik/lfw_dataset/pairsDevTest.txt"

datasetPath = f"/Users/necatiisik/lfw_dataset/lfw/"
trainLabelPath = f"/Users/necatiisik/lfw_dataset/train_fixed.csv"
testLabelPath = f"/Users/necatiisik/lfw_dataset/test_fixed.csv"


tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

train_data = CustomImageDataset(trainLabelPath, datasetPath, transform=tf, target_transform=None)
test_data = CustomImageDataset(testLabelPath, datasetPath, transform=tf, target_transform=None)

batch_size = 64

train_loader = DataLoader(dataset = train_data,
                          batch_size = batch_size, 
                          shuffle = True)

test_loader = DataLoader(dataset = test_data,
                         batch_size = batch_size, 
                         shuffle = True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*250*250, 10240),
            nn.ReLU(),
            nn.Linear(10240, 10240),
            nn.ReLU(),
            nn.Linear(10240, 5749)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")


