# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:24:48 2022

@author: Maria
"""


# Imports
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models

# useful for confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# this class creates a simplified dataset that only contains the information
# regarding malignancy, ignoring the remaining attributes
class DatasetSimplifiedVersion (Dataset):
    def __init__ (self, csv_file, root_dir, transform = None):
        self.info = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__ (self):
        return len(self.info)
    
    def __getitem__ (self, index):
        # gets the important information about each image: its name and label
        img_name = os.path.join(self.root_dir, self.info.iloc[index, 0])
        image = io.imread(img_name)
        label = self.info.iloc[index, 1]
        
        # we are working with grayscale images but resnet18 works with RGB 
        # images => there is the need to create a fake RGB image using ours as 
        # a starting point; to do so, we replicate the gray channel twice => 
        # we end up with 3 identical channels (this shouldn't affect model's 
        # efficiency)
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Image.fromarray(image)
        sample = {'image': image, 'label': label}
        
        # applies the required transform to the image, mantaining the label
        if self.transform:
            transformedImg = self.transform(sample['image'])
            sample = {'image': transformedImg, 'label': label}

        return sample

# check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model selection => pretrained resnet18 with a modification in the last layer 
pretrained = True
model = models.resnet18(pretrained=pretrained)

# (only 2 output classes needed)
model.fc = nn.Linear(512, 2)
model.to(device)

# defining the used method to calculate the loss and the used optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adagrad(model.parameters(), lr=0.0003)

# the images aren't grayscale anymore when "transform" is called => we have to 
# normalize 3 channels instead of 1

if pretrained:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

else:
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]


transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0,360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)

transform_checkup = transform_test

# differentiate between Spyder and Slurm
if device.type == 'cpu':
    initPath = 'C:/Users/Maria/Documents/GitHub/breast-cancer-classification'
    num_workers = 0
else:
    initPath = '/nas-ctm01-homes/mecaldeira'
    num_workers = 4

# initialize batch_size
batch_size = 10

# building trainset, trainloader, testset and testloader
csv_file = 'labels.csv'


# used cl's folder
folder = 'all'

# Create a dictionary to append the: root_dir_train, trainset, trainloader
train_vars_dict = dict()

for i in range (5):
    train_vars_dict["root_dir_train" + str(i + 1)] = initPath + '/data/cl/' + folder + '/train/sub' + str(i + 1)

    if i == 0:
        total_trainset = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = train_vars_dict["root_dir_train" + str (i + 1)], transform = transform_train)
    else:
        total_trainset += DatasetSimplifiedVersion(csv_file = csv_file, root_dir = train_vars_dict["root_dir_train" + str (i + 1)], transform = transform_train)
        
    train_vars_dict["trainloader" + str (i + 1)] = torch.utils.data.DataLoader(total_trainset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

root_dir_test = initPath + '/data/cl/' + folder + '/val'
testset = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = root_dir_test, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

root_dir_checkup = initPath + '/data/cl/' + folder + '/test'
checkupset = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = root_dir_checkup, transform = transform_checkup)
checkuploader = torch.utils.data.DataLoader(checkupset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

# Create a dictionary to append the: root_dir_checkup, checkupset, checkuploader
checkup_vars_dict = dict()

for i in range (5):
    checkup_vars_dict["root_dir_checkup" + str(i + 1)] = initPath + '/data/cl/' + folder + '/testSub/sub' + str(i + 1)
    checkup_vars_dict["checkupset" + str (i + 1)] = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = checkup_vars_dict["root_dir_checkup" + str (i + 1)], transform = transform_checkup)
    checkup_vars_dict["checkuploader" + str (i + 1)] = torch.utils.data.DataLoader(checkup_vars_dict["checkupset" + str (i + 1)], batch_size = batch_size, shuffle = True, num_workers = num_workers)
   
    if i == 0:
        checkupset = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = checkup_vars_dict["root_dir_checkup" + str (i + 1)], transform = transform_checkup)
    else:
        checkupset += DatasetSimplifiedVersion(csv_file = csv_file, root_dir = checkup_vars_dict["root_dir_checkup" + str (i + 1)], transform = transform_checkup)
    
    
# defining the possible labels
classes = ('benign', 'malignant')


# this variable saves the maximum accuracy of the model during the training 
# process
accMaxTrain = 0
accMaxTest = 0
epochMax = -1

# initialize number of subsets used
numSubsets = 1

# count of the number of consecutive epochs where there was no improvement in accuracy
count = 0
previousNum = 0
actualNum = 0

trainingAccuracy = []
testAccuracy = []
trainingLoss = []
epochList = []
numSubArr = []
isLast = False
epoch = -1

confMatxs = {'tl1': [],
            'pl1': [],
            'tl2': [],
            'pl2': [],
            'tl3': [],
            'pl3': [],
            'tl4': [],
            'pl4': [],
            'tl5': [],
            'pl5': []}

trainloader = train_vars_dict["trainloader5"];

# training the model using 'trainset'
while not isLast:
    epoch += 1
    actualNum += 1
    epochList.append(epoch + 1)
    running_loss = 0.0
    totalTrain = 0
    correctTrain = 0
    
    model.train()
    
    for i, data in enumerate(train_vars_dict["trainloader" + str (numSubsets)], 0):
        # inputs -> "fake RGB" image
        # labels -> a string indicating if the lesion is benign (0) or malignant (1)
        inputs = data ['image']
        inputs = inputs.to(device)
        labels = data ['label']
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model.train()(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # updating the loss value
        running_loss += loss.cpu()
            
        # calculating the train accuracy
        _, predicted = torch.max(outputs.data, 1)
        totalTrain += labels.size(0)
        correctTrain += (predicted == labels).sum().item()
            
            
    trainAcc = 100 * correctTrain / totalTrain
    trainingAccuracy.append(trainAcc)
    trainingLoss.append((running_loss / totalTrain * batch_size).cpu().detach().numpy())
    numSubArr.append(numSubsets)
    
    if trainAcc > accMaxTrain:
        accMaxTrain = trainAcc
        count = 0
    else:
        count += 1
     
    # adds a new subset if the accuaracy in the trainset doesn't improve for 5
    # consecutive epochs or if it exceeds 95% (the model already learned to
    # evaluate the current images)
    if (count >= 15 or accMaxTrain > 95) and actualNum >= 0.9 * previousNum:
        previousNum = actualNum
        actualNum = 0
        count = 0
        accMaxTrain = 0
            
        true_labels = np.array([])
        pred_labels = np.array([])

        model.eval() 
        
        with torch.no_grad():
            for data in checkuploader:
                images = data ['image']
                images = images.to(device)
                im_labels = data ['label']
                im_labels = im_labels.to(device)
        
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
        
                true_labels = np.concatenate ((true_labels, im_labels.cpu().numpy()))
                pred_labels = np.concatenate ((pred_labels, predicted.cpu().numpy()))

        
        confMatxs['tl' + str(numSubsets)] += true_labels.tolist()
        confMatxs['pl' + str(numSubsets)] += pred_labels.tolist()
        
        if numSubsets < 5:
            numSubsets += 1
        else:
            isLast = True
            
    # testing the trained model using 'testset'
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data in testloader:
            images = data ['image']
            images = images.to(device)
            labels = data ['label']
            labels = labels.to(device)
        
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    testAccuracy.append(acc)
    print(f'\nAccuracy of epoch {epoch + 1}: {acc} % \nLoss: {running_loss / totalTrain}')

    if acc >= accMaxTest:
        accMaxTest = acc
        torch.save(model, 'invCL')
        epochMax = epoch
        

bestModel = torch.load('invCL')

# testing the trained model using 'test' folder
bestModel.eval()

with torch.no_grad():
    for data in checkuploader:
        images = data ['image']
        images = images.to(device)
        im_labels = data ['label']
        im_labels = im_labels.to(device)
        
        # calculate outputs by running images through the network
        outputs = bestModel(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        
        total += im_labels.size(0)
        correct += (predicted == im_labels).sum().item()


# determine the accuracy in the 'test' folder
accCheckup =  100 * correct / total

tl_list = []
pl_list = []

with torch.no_grad():
    for sub in range(5):
        t_labels = np.array([])
        p_labels = np.array([])
        
        for i, data in enumerate(checkup_vars_dict["checkuploader" + str (sub + 1)], 0):
            
            images = data ['image']
            images = images.to(device)
            im_labels = data ['label']
            im_labels = im_labels.to(device)
        
            # calculate outputs by running images through the network
            outputs = bestModel(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
        
            t_labels = np.concatenate ((t_labels, im_labels.cpu().numpy()))
            p_labels = np.concatenate ((p_labels, predicted.cpu().numpy()))
            
        tl_list.append(t_labels)
        pl_list.append(p_labels)


fig, axs = plt.subplots (2, 2, figsize = (10, 8))

axs[0][0].plot(np.array(epochList), np.array(numSubArr), color = 'red')
axs[0][0].set_title('Number of Subsets')

axs[0][1].plot(np.array(epochList), np.array(trainingLoss), color = 'blue')
axs[0][1].set_title('Training Loss')
    
axs[1][0].plot(np.array(epochList), np.array(trainingAccuracy), color = 'green')
axs[1][0].set_title('Training Acc')
    
axs[1][1].plot(np.array(epochList), np.array(testAccuracy), color = 'purple')
axs[1][1].set_title('Validation Acc')

# Save figure into disk
figure_name = 'plots_invCL' 
figure_path = os.path.join(initPath, f"{figure_name}")
plt.savefig(fname = figure_path, bbox_inches = 'tight')
    
# determine the confusion matrixes for the data in the 'test' folder
extra = ''
      
for i in range (5):
    if len(confMatxs['tl' + str(i + 1)]) == 0:
        break
    
    if i == 1:
        extra = 's'
        
    confMatrix = confusion_matrix(confMatxs['tl' + str(i + 1)], confMatxs['pl' + str(i + 1)], normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels = classes)
    disp.plot()
    disp.ax_.set_title('Confusion Matrix -> ' + str(i + 1) + ' subset' + extra)
            
    # Save figure into disk
    figure_name = 'cm_inv_' + str(i + 1) + 'sub'
    figure_path = os.path.join(initPath, f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    
for i in range (5):
    # determine the confusion matrix for the data in the 'test' folder _> by subset
    confMatrix = confusion_matrix(tl_list[i], pl_list[i], normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels = classes)
    disp.plot()
    disp.ax_.set_title('Confusion Matrix -> Subset ' + str (i + 1))

    # Save figure into disk
    figure_name = 'cm_inv_sub' + str(i + 1)
    figure_path = os.path.join(initPath, f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')

print(f'Finished Training -> Inverted CL\nMaximum accuracy reached at validation: {accMaxTest} % at epoch {epochMax + 1}\nAccuracy of the best model in testset: {accCheckup} %\nTotal number of epochs: {epoch + 1}')