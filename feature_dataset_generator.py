# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 08:56:26 2022

@author: Maria
"""

# Imports
import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage import io

# PyTorch Imports
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models


# this class creates a simplified dataset that only contains the information
# regarding malignancy, ignoring the remaining attributes
class DatasetTrain (Dataset):
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
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Image.fromarray(image)
        
        angle = [0, 90, 180, 270]
        samples = {'image': [],
                   'label': []}
        
        # we are working with grayscale images but resnet18 works with RGB 
        # images => there is the need to create a fake RGB image using ours as 
        # a starting point; to do so, we replicate the gray channel twice => 
        # we end up with 3 identical channels (this shouldn't affect model's 
        # efficiency)
        for rotation in range (len(angle)):
            newImage1 = TF.rotate(image, angle[rotation])
            newImage2 = TF.hflip(newImage1)

            if self.transform:
                newImage1 = self.transform(newImage1)
                newImage2 = self.transform(newImage2)
                
            samples['image'].append(newImage1)
            samples['image'].append(newImage2)
            samples['label'].append(label)
            samples['label'].append(label)
            
        return samples

class DatasetTestCheckup (Dataset):
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
model.train()

# removing the last layer -> we get a feature extractor
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(device)

# # defining the used method to calculate the loss and the used optimizer
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adagrad(model.parameters(), lr=0.0003)

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

# WARNING: '' was set to false to allow a deterministic output
for i in range (5, 0, -1):
    train_vars_dict["root_dir_train" + str(i)] = initPath + '/data/cl/' + folder + '/train/sub' + str(i)
    trainset = DatasetTrain(csv_file = csv_file, root_dir = train_vars_dict["root_dir_train" + str (i)], transform = transform_train)
    train_vars_dict["trainloader" + str (i)] = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

root_dir_test = initPath + '/data/cl/' + folder + '/val'
testset = DatasetTestCheckup(csv_file = csv_file, root_dir = root_dir_test, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

root_dir_checkup = initPath + '/data/cl/' + folder + '/test'
checkupset = DatasetTestCheckup(csv_file = csv_file, root_dir = root_dir_checkup, transform = transform_checkup)
checkuploader = torch.utils.data.DataLoader(checkupset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

# Create a dictionary to append the: root_dir_checkup, checkupset, checkuploader
checkup_vars_dict = dict()

for i in range (5):
    checkup_vars_dict["root_dir_checkup" + str(i + 1)] = initPath + '/data/cl/' + folder + '/testSub/sub' + str(i + 1)
    checkup_vars_dict["checkupset" + str (i + 1)] = DatasetTestCheckup(csv_file = csv_file, root_dir = checkup_vars_dict["root_dir_checkup" + str (i + 1)], transform = transform_checkup)
    checkup_vars_dict["checkuploader" + str (i + 1)] = torch.utils.data.DataLoader(checkup_vars_dict["checkupset" + str (i + 1)], batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
# defining the possible labels
classes = ('benign', 'malignant')

generalPath = initPath + '/data/cl/' + folder + '/simpDataset' 
path = generalPath + '/train/sub'

with torch.no_grad():
    for sub in range (5, 0, -1):
        for i, data in enumerate(train_vars_dict["trainloader" + str (sub)], 0):
            # inputs -> "fake RGB" image
            # labels -> a string indicating if the lesion is benign (0) or malignant (1)
            input_sample = data ['image']
            label_sample = data ['label']
        
            for dataAug in range (len(input_sample)):
                if dataAug == 0:
                    inputs = input_sample[dataAug]
                    labels = label_sample[dataAug]
                else:
                    inputs = torch.cat ((inputs, input_sample[dataAug]))
                    labels = torch.cat ((labels, label_sample[dataAug]))
        
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            if i==0:
                outputs = model.train()(inputs)
                out_lbl = labels
            else:
                outputs = torch.cat ((outputs, model.train()(inputs)))
                out_lbl = torch.cat ((out_lbl, labels))
            
        #save stuff -> create folders
        finalPath = path + str(sub)
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
    
        attr = {'number': [],
                'label': []}
        for index in range (0, len(out_lbl)):
            final_arr = outputs[index]
            final_arr.cpu().detach().numpy();
            np.save(finalPath + '//' + str(index + 1) + '.npy', final_arr)
            attr["number"].append (str(index + 1) + '.npy')
            attr["label"].append (out_lbl[index].tolist())
        
        attr_df = pd.DataFrame(attr)
        attr_df.to_csv(finalPath + '//' + 'labels.csv', header=True, index=False)
    
model.eval()

path = generalPath + '/val'

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images = data ['image']
        images = images.to(device)
        labels = data ['label']
        labels = labels.to(device)
        
        if i==0:
            outputs = model(images)
            out_lbl = labels
        else:
            outputs = torch.cat ((outputs, model(images)))
            out_lbl = torch.cat ((out_lbl, labels))
            
    if not os.path.exists(path):
        os.makedirs(path)
    
    attr = {'number': [],
            'label': []}
    for index in range (0, len(out_lbl)):
        final_arr = outputs[index]
        final_arr.cpu().detach().numpy();
        np.save(path + '//' + str(index + 1) + '.npy', final_arr)
        attr["number"].append (str(index + 1) + '.npy')
        attr["label"].append (out_lbl[index].tolist())
        
    attr_df = pd.DataFrame(attr)
    attr_df.to_csv(path + '//' + 'labels.csv', header=True, index=False)
        
    
path = generalPath + '/test'

with torch.no_grad():
    for i, data in enumerate(checkuploader, 0):

        images = data ['image']
        images = images.to(device)
        labels = data ['label']
        labels = labels.to(device)
        
        if i==0:
            outputs = model(images)
            out_lbl = labels
        else:
            outputs = torch.cat ((outputs, model(images)))
            out_lbl = torch.cat ((out_lbl, labels))
            
    if not os.path.exists(path):
        os.makedirs(path)
    
    attr = {'number': [],
            'label': []}
    for index in range (0, len(out_lbl)):
        final_arr = outputs[index]
        final_arr.cpu().detach().numpy();
        np.save(path + '//' + str(index + 1) + '.npy', final_arr)
        attr["number"].append (str(index + 1) + '.npy')
        attr["label"].append (out_lbl[index].tolist())
        
    attr_df = pd.DataFrame(attr)
    attr_df.to_csv(path + '//' + 'labels.csv', header=True, index=False)

path = generalPath + '/testSub/sub'

with torch.no_grad():
    for sub in range(5):
        for i, data in enumerate(checkup_vars_dict["checkuploader" + str (sub + 1)], 0):
            images = data ['image']
            images = images.to(device)
            im_labels = data ['label']
            im_labels = im_labels.to(device)
            
            if i==0:
                outputs = model(images)
                out_lbl = im_labels
            else:
                outputs = torch.cat ((outputs, model(images)))
                out_lbl = torch.cat ((out_lbl, im_labels))
                
        #save stuff -> create folders
        finalPath = path + str(sub + 1)
        if not os.path.exists(finalPath):
            os.makedirs(finalPath)
    
        attr = {'number': [],
                'label': []}
        
        for index in range (0, len(out_lbl)):
            final_arr = outputs[index]
            final_arr.cpu().detach().numpy();
            np.save(finalPath + '//' + str(index + 1) + '.npy', final_arr)
            attr["number"].append (str(index + 1) + '.npy')
            attr["label"].append (out_lbl[index].tolist())
            
            attr_df = pd.DataFrame(attr)
            attr_df.to_csv(finalPath + '//' + 'labels.csv', header=True, index=False)