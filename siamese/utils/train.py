import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import apart
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import time
from math import ceil
import json
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE
from safetensors.torch import  save_model, load_model
import plotly.subplots as sp
import torchvision.transforms as transforms
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from copy import deepcopy
import torchvision.datasets as datasets
from models import *
from utils.plots import *
from tqdm import tqdm

def train(model, dl_train, dl_val, device, criterion, optimizer, l1_weight=None, weight_decay=0, epochs=100, triplet=False):
    criterion_loss_train = []
    l1_loss_train = []
    l2_loss_train = []
    training_loss_train = []
    len_train = len(dl_train)

    criterion_loss_val = []
    l1_loss_val = []
    l2_loss_val = []
    validation_loss_val = []
    len_val = len(dl_val)
    
    start_time = time.time()
# Iteracja po epokach
    for epoch in range(epochs):    # iteracja po pakietach
        criterion_loss_train_epoch = []
        l1_loss_train_epoch = []
        l2_loss_train_epoch = []
        training_loss_train_epoch = []
        
        train_batch(model, 
                    dl_train, 
                    device, 
                    criterion, 
                    optimizer, 
                    l1_weight, 
                    weight_decay, 
                    criterion_loss_train_epoch, 
                    l1_loss_train_epoch, 
                    l2_loss_train_epoch, 
                    training_loss_train_epoch,
                    triplet=triplet)
        
        criterion_loss_val_epoch = []
        l1_loss_val_epoch = []
        l2_loss_val_epoch = []
        validation_loss_val_epoch = []
        with torch.no_grad():
            train_batch(model, 
                dl_train, 
                device,  
                criterion, 
                optimizer, 
                l1_weight, 
                weight_decay, 
                criterion_loss_val_epoch, 
                l1_loss_val_epoch, 
                l2_loss_val_epoch, 
                validation_loss_val_epoch,
                validation=True,
                triplet=triplet)

        criterion_loss_train.append(np.mean(criterion_loss_train_epoch)/len_val)
        l1_loss_train.append(np.mean(l1_loss_train_epoch)/len_train)  
        l2_loss_train.append(np.mean(l1_loss_train_epoch)/len_train)
        training_loss_train.append(np.mean(training_loss_train_epoch)/len_train)
        
        criterion_loss_val.append(np.mean(criterion_loss_val_epoch)/len_val)
        l1_loss_val.append(np.mean(l1_loss_val_epoch)/len_val)  
        l2_loss_val.append(np.mean(l1_loss_val_epoch)/len_val)
        validation_loss_val.append(np.mean(validation_loss_val_epoch)/len_val)
        
        
        print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train: avg_loss: {criterion_loss_train[-1]:.4f}, "
                f"Validation: avg_loss: {criterion_loss_val[-1]:.4f}")

    execution_time = time.time() - start_time
    losses_scores = {
        'criterion_loss_train': criterion_loss_train,
        'l1_loss_train': l1_loss_train,
        'l2_loss_train': l2_loss_train,
        'train_loss_train': training_loss_train,
        'criterion_loss_val': criterion_loss_val,
        'l1_loss_val': l1_loss_val,
        'l2_loss_val': l2_loss_val,
        'validation_loss_val': validation_loss_val,
    }
    return model, losses_scores, execution_time

def train_batch(model, 
                dl_train,
                device, 
                criterion, 
                optimizer, 
                l1_weight, 
                weight_decay, 
                criterion_loss_train_epoch, 
                l1_loss_train_epoch, 
                l2_loss_train_epoch, 
                validation_loss_val_epoch,
                validation=False,
                triplet=False):
    for i, (img0, img1, label) in enumerate(dl_train, 0):
        if validation:
            model.eval()
        # Przeniesienie obrazów i etykiet do pamięci karty graficznej
        

        # Ustawienie wartości gradientów na zero
        optimizer.zero_grad()

        # Przepuszczenie dwóch obrazów przez sieć i uzyskanie dwóch wektorów wyjściowych
        if triplet:
            img0, img1, img2 = img0.to(device), img1.to(device), label.to(device)
            # :p nie chce refactor
            output1, output2, label = model(img0, img1, img2)
        else:
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output1, output2 = model(img0, img1)
            
        if l1_weight:
            l1_loss_single_train = sum(p.abs().sum() for p in model.parameters()).to('cuda')
            l1_loss_single_train = l1_loss_single_train * l1_weight
            l1_loss_train_epoch.append(l1_loss_single_train.item())
        else:
            l1_loss_single_train = 0.0
            l1_loss_train_epoch.append(l1_loss_single_train)
                
        l2_loss_single_train = calculate_l2_norm(model, weight_decay)
        l2_loss_train_epoch.append(l2_loss_single_train.item())

        # Obliczenie wartości funkcji straty na podstawie wektorów wyjściowych
        loss_contrastive = criterion(output1, output2, label)
        criterion_loss_train_epoch.append(loss_contrastive.item())
            
        loss_contrastive = loss_contrastive + l1_loss_single_train
        validation_loss_val_epoch.append(loss_contrastive.item()+l2_loss_single_train.item())

        # Obliczenie zmian wag w oparciu o wartość funkcji straty
        if validation:
            model.train()
        else:
            loss_contrastive.backward()
            
            # Aktualizacja wag sieci
            optimizer.step()
            

def calculate_l2_norm(model, weight_decay=0.0):
    l2_norm = 0.0
    
    # Dodanie składnika związanego z weight_decay
    for param in model.parameters():
        l2_norm += weight_decay * torch.norm(param.data, p=2)**2
    
    return l2_norm


def create_logs(epochs,
                lr,
                criterion,
                optimizer,
                batch_size,
                weight_decay,
                model_name,
                exp_title,
                exp_index,
                execution_time,
                loss_type,
                margin,
                same_person_prob,
                l1_weight,          
                history):
    logs = history.copy()
    logs['epochs'] = epochs
    logs['lr'] = lr
    logs['batch_size'] = batch_size
    logs['weight_decay'] = weight_decay
    logs['model_name'] = model_name
    logs['optimizer'] = str(optimizer)
    logs['criterion'] = str(criterion)
    logs['loss_type'] = str(loss_type)
    logs['margin'] = str(margin)
    logs['same_person_prob'] = str(same_person_prob)
    logs['exp_title'] = exp_title
    logs['exp_index'] = exp_index
    logs['l1_weight'] = l1_weight
    logs['execution_time'] = execution_time

    return logs

def download_dataset(model_name, same_person_prob=None):
    # Load the ImageFolder dataset
    folder_dataset_train = datasets.ImageFolder(root="./data/faces/training/")
    folder_dataset_test = datasets.ImageFolder(root="./data/faces/testing/")

    # Transformation for resizing and converting to tensor
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                        transforms.ToTensor()])

    if model_name == 'siamese':
        # Initialize the SiameseNetworkDataset
        dl_train = SiameseNetworkDataset(imageFolderDataset=folder_dataset_train,
                                                transform=transformation, same_person_prob=same_person_prob)
        dl_val = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transformation, same_person_prob=same_person_prob)
    elif model_name == 'triplet':
        dl_train = TripletNetworkDataset(imageFolderDataset=folder_dataset_train,
                                                transform=transformation)
        dl_val = TripletNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transformation)
    
    return dl_train, dl_val
        
def get_data(data_type):
    # Load the ImageFolder dataset
    if data_type == 'test':
        folder_dataset_test = datasets.ImageFolder(root="./data/faces/testing/")
    else:
        folder_dataset_test = datasets.ImageFolder(root="./data/faces/training/")
    # Transformation for resizing and converting to tensor
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                        transforms.ToTensor()])

    imageFolderDataset = folder_dataset_test.imgs
    
    data_X = []
    data_Y = []
    for img_tuple in folder_dataset_test.imgs:
        # Otwierasz obrazek
        img = Image.open(img_tuple[0])

        # Konwertujesz do skali szarości
        img = img.convert("L")

        # Zastosujesz transformacje
        img = transformation(img)
        data_X.append(img)
        data_Y.append(img_tuple[1])

    return data_X, data_Y




def save_dict(input_dict, file_name):
    try:
        with open(file_name, 'a') as plik:
            plik.write(json.dumps(input_dict) + '\n')
        print(f'Słownik został zapisany do pliku {file_name}.')
    except Exception as e:
        print(f'Błąd podczas zapisywania słownika do pliku: {e}')


def do_exp(model, train_loader, val_loader, device, criterion, optimizer, l1_weight, 
           weight_decay, epochs, triplet, lr, batch_size, loss_type, margin, same_person_prob, model_name, exp_title, exp_index, root):
    model, history, execution_time = train(model, train_loader, val_loader, device, criterion, 
                                                 optimizer, l1_weight, weight_decay, epochs, triplet)
    
    logs = create_logs(epochs=epochs,
                        lr=lr,
                        criterion=criterion,
                        optimizer=optimizer,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        model_name=model_name,
                        exp_title=exp_title,
                        exp_index=exp_index,
                        execution_time=execution_time,
                        loss_type=loss_type,
                        margin=margin,
                        same_person_prob=same_person_prob,
                        l1_weight=l1_weight,          
                        history=history)
    
    save_exp(model, model_name, exp_title, exp_index, root, logs)


def save_exp(model, 
            model_name, 
            exp_title, 
            exp_index,  
            root, 
            logs=False,
            show=False):
    
    save_model(model, f'{root}/models/model_{exp_title}_{exp_index}_{model_name}.safetensors')
    print('Saving model', exp_index)

    if logs:
        save_dict(logs, f'{root}/logs.json')
        print('Zapis do logs.json')


def load_models(exp_title, model_types, device):
    models = []
    model_names = []
    for i, model_type in enumerate(model_types):
        
        model_name = f'model_{exp_title}_{i}_{model_type}.safetensors'
        file_path = f'models/{model_name}'
        
        if model_type == 'siamese':
            model = SiameseNetwork().to(device)
        elif model_type == 'triplet':
            model = TripletNetwork().to(device)
        else:
            raise Exception(ValueError)
        
        load_model(model, file_path)
        model_names.append(model_name)
        models.append(model)
            
    return  models, model_types, model_names