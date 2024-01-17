import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SiameseNetwork(nn.Module):

    def __init__(self, norm=False):
        super(SiameseNetwork, self).__init__()
        self.norm = norm

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,2)
        )

    def forward_once(self, x):
        # Funkcja zostanie wywołana osobno dla każdej próbki
        # Jej wyjście zostanie wykorzystane w celu określenia odległości
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        if self.norm:
            output = F.normalize(output, p=2)
        return output

    def forward(self, input1, input2):
        # Funkcja przyjmuje obie próbki i zwraca parę wektorów
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
    
        return output1, output2
    
class TripletNetwork(nn.Module):

    def __init__(self, norm=False):
        super(TripletNetwork, self).__init__()
        self.norm = norm

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,2)
        )

    def forward_once(self, x):
        # Funkcja zostanie wywołana osobno dla każdej próbki
        # Jej wyjście zostanie wykorzystane w celu określenia odległości
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        if self.norm:
            output = F.normalize(output, p=2)
        return output

    def forward(self, input_anchor, input_positive, input_negative):
        # Funkcja przyjmuje obie próbki i zwraca parę wektorów
        output_anchor = self.forward_once(input_anchor)
        output_positive = self.forward_once(input_positive)
        output_negative = self.forward_once(input_negative)

        return output_anchor, output_positive, output_negative
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0, loss_type=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_type = loss_type

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=2.0, loss_type='relu'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_type = loss_type

    def forward(self, output1, output2, output3):
        distance_positive = F.pairwise_distance(output1, output2, keepdim = True)
        distance_negative = F.pairwise_distance(output1, output3, keepdim = True)
        if self.loss_type == 'relu':
            loss_contrastive = torch.mean(F.relu(distance_positive - distance_negative + self.margin))
        elif self.loss_type == 'max':
            loss_contrastive = torch.mean(torch.clamp(distance_positive - distance_negative + self.margin, min=0))
        elif self.loss_type == 'leaky':
            loss_contrastive = torch.mean(F.leaky_relu(distance_positive - distance_negative + self.margin))
        elif self.loss_type == 'dist':
            loss_contrastive = torch.mean(distance_positive - distance_negative + self.margin)
        return loss_contrastive
        
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None, same_person_prob=0.5):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.same_person_prob = same_person_prob

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # Potrzebujemy mniej więcej połowę par podobnych obrazków
        should_get_same_class = random.random()
        if should_get_same_class <= self.same_person_prob:
            while True:
                # Wyszukiwanie zdjęcia tej samej osoby
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                # Wyszukiwanie zdjęcia innej  osoby
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    
class TripletNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        while True:
            # Wyszukiwanie zdjęcia tej samej osoby
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break

        while True:
            # Wyszukiwanie zdjęcia innej  osoby
            img2_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] != img2_tuple[1]:
                break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def __len__(self):
        return len(self.imageFolderDataset.imgs)