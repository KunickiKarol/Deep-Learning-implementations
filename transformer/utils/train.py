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
import os

import gdown
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoModel, AutoTokenizer

def froze_param(model, non_freeze=False):
    for name, param in model.named_parameters():
        if non_freeze:
            if name.startswith(non_freeze):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

def train(model, train_dataloader, device, loss_func, optimizer):
    losses_epoch = []
    model.train()
    total_loss = 0

    # przygotowanie listy do przechowywania predykcji modelu
    total_preds=[]

    for batch in train_dataloader:

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        model.zero_grad()

        preds = model(sent_id, mask)
        
        loss = loss_func(preds, labels)
        loss_value = loss.item()
        losses_epoch.append(loss_value)
        total_loss = total_loss + loss_value

        loss.backward()

        # Normalizacja wartości gradientów
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds=preds.detach().cpu().numpy()

    total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)

    # Predykcje modelu mają wymiary (liczba pakietów, rozmiar pakietu, liczba klas).
    # Przekształcenie ich do wymiarów (liczba próbek, liczba klas)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, losses_epoch

def evaluate(model, val_dataloader, device, loss_func):
    model.eval()

    total_loss = 0

    losses_epoch = []
    total_preds = []

    for batch in val_dataloader:
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        with torch.no_grad():

            preds = model(sent_id, mask)

            loss = loss_func(preds,labels)
            loss_value = loss.item()
            losses_epoch.append(loss_value)
            total_loss = total_loss + loss_value

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)

    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, losses_epoch

def fit(model, train_dataloader, val_dataloader, device, epochs, loss_func, optimizer):
    best_valid_loss = float('inf')
    model_path = 'actual_weights.pt'
    # Inicjalizacja list na wartości funkcji straty na zbiorze uczącym i walidacyjnym
    train_losses_avg=[]
    valid_losses_avg=[]
    
    
    train_losses_epoches=[]
    valid_losses_epoches=[]
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print('  Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss, _, train_losses_epoch = train(model, train_dataloader, device, loss_func, optimizer)
        valid_loss, _, val_losses_epoch = evaluate(model, val_dataloader, device, loss_func)
        
        # zapisanie najlepszego modelu
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_path')

        train_losses_avg.append(train_loss)
        valid_losses_avg.append(valid_loss)
        
        train_losses_epoches.extend(train_losses_epoch)
        valid_losses_epoches.extend(val_losses_epoch)
        print(f'    Training Loss: {train_loss:.3f}\tValidation Loss: {valid_loss:.3f}\tTime: {time.time()-epoch_start_time}')
        
    execution_time = time.time() - start_time
    model.load_state_dict(torch.load('model_path'))
    return model, execution_time, train_losses_avg, valid_losses_avg, train_losses_epoches, valid_losses_epoches


def create_logs(epochs,
                lr,
                loss_func,
                optimizer,
                batch_size, 
                max_length, 
                padding, 
                truncation,
                pretrained_model_name,
                model_type,
                exp_title,
                exp_index,
                execution_time,
                train_losses_avg,
                valid_losses_avg,
                train_losses_epoches,
                valid_losses_epoches,
                report):
    logs = {}
    logs['epochs'] = epochs
    logs['lr'] = lr
    logs['batch_size'] = batch_size
    logs['max_length'] = max_length
    logs['padding'] = padding
    logs['truncation'] = truncation
    logs['valid_losses_epoches'] = valid_losses_epoches
    logs['pretrained_model_name'] = pretrained_model_name
    logs['model_type'] = model_type
    logs['optimizer'] = str(optimizer)
    logs['train_losses_epoches'] = str(train_losses_epoches)
    logs['loss_func'] = str(loss_func)
    logs['valid_losses_avg'] = str(valid_losses_avg)
    logs['exp_title'] = exp_title
    logs['exp_index'] = exp_index
    logs['train_losses_avg'] = train_losses_avg
    logs['execution_time'] = execution_time
    logs['report'] = report

    return logs

def get_model(pretrained_model_name, model_type, device):
    if pretrained_model_name.startswith('roberta'):
        non_freeze = 'pooler'
    else:
        non_freeze=False
        
    if 'large' in pretrained_model_name:
        input_size = 1024
    else:
        input_size = 768
        
    
    pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
    froze_param(pretrained_model, non_freeze)
    model = model_type(pretrained_model, input_size=input_size)
    model = model.to(device)
    return model

def tokenize_data(pretrained_model_name, max_length, padding, truncation, batch_size, device, test_dl=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer_setup = get_tokenizer_setup(tokenizer, max_length=max_length, padding=padding, truncation=truncation)
    train_dataloader, val_dataloader, test_data, weights = get_data(tokenizer_setup, device, batch_size=batch_size, test_dl=test_dl)
    return train_dataloader, val_dataloader, test_data, weights

def tokenize(text, tokenizer_setup):
    tokens = tokenizer_setup['tokenizer'].batch_encode_plus(text.tolist(),
                                            max_length = tokenizer_setup['max_length'],
                                            padding=tokenizer_setup['padding'],
                                            truncation=tokenizer_setup['truncation']
                                            )
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    
    return seq, mask
    
def get_tokenizer_setup(tokenizer, max_length=25, padding='max_length', truncation=True):
    tokenizer_setup = {'tokenizer': tokenizer,
                        'max_length': max_length,
                        'padding': padding,
                        'truncation':truncation
                        }
    
    return tokenizer_setup 

def get_weights(train_labels, device):
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    weights= torch.tensor(class_weights,dtype=torch.float)
    return weights.to(device)

def get_data(tokenizer_setup, device, batch_size=32, random_state=2024, data_file='data.csv', test_dl=False):
    df = pd.read_csv(data_file)

    train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],
                                                                        random_state=random_state,
                                                                        test_size=0.3,
                                                                        stratify=df['label'])


    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=random_state,
                                                                    test_size=0.5,
                                                                    stratify=temp_labels)

    train_seq, train_mask = tokenize(train_text, tokenizer_setup)
    val_seq, val_mask = tokenize(val_text, tokenizer_setup)
    test_seq, test_mask = tokenize(test_text, tokenizer_setup)

    train_y = torch.tensor(train_labels.tolist())
    val_y = torch.tensor(val_labels.tolist())
    test_y = torch.tensor(test_labels.tolist())


    train_data = TensorDataset(train_seq, train_mask, train_y)
    # Przygotowanie obiektu klasy pozwalającej na próbkowanie zbioru uczącego
    train_sampler = RandomSampler(train_data)
    # Przygotowanie obhiektu klasy DataLoader dla zbioru uczącego
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_seq, val_mask, val_y)
    # Przygotowanie obiektu klasy pozwalającej na próbkowanie zbioru uczącego
    val_sampler = SequentialSampler(val_data)
    # Przygotowanie obhiektu klasy DataLoader dla zbioru walidacyjnego
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
    
    weights = get_weights(train_labels, device)
    
    if test_dl:
        test_data = TensorDataset(test_seq, test_mask, test_y)
        # Przygotowanie obiektu klasy pozwalającej na próbkowanie zbioru uczącego
        test_sampler = SequentialSampler(test_data)
        # Przygotowanie obhiektu klasy DataLoader dla zbioru walidacyjnego
        test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)
        
        return train_dataloader, val_dataloader, test_dataloader, weights
    else:
        return train_dataloader, val_dataloader, {'seq': test_seq, 'mask':test_mask, 'y':test_y}, weights


def save_dict(input_dict, file_name):
    try:
        with open(file_name, 'a') as plik:
            plik.write(json.dumps(input_dict) + '\n')
        print(f'Słownik został zapisany do pliku {file_name}.')
    except Exception as e:
        print(f'Błąd podczas zapisywania słownika do pliku: {e}')


def do_exp(model, train_dataloader, val_dataloader, test_data, device, loss_func, optimizer, epochs, lr, 
           batch_size, max_length, padding, truncation, pretrained_model_name, exp_title, exp_index, root):
    
    model, execution_time, train_losses_avg, valid_losses_avg, train_losses_epoches, valid_losses_epoches = fit(model, 
                                                                                                                train_dataloader,
                                                                                                                val_dataloader,
                                                                                                        device,
                                                                                                                epochs, 
                                         loss_func, optimizer)
    # Wygenerowanie predykcji dla zbioru testowego
    with torch.no_grad():
        preds = model(test_data['seq'].to(device), test_data['mask'].to(device))
        preds = preds.detach().cpu().numpy()
        
    preds = np.argmax(preds, axis = 1)
    report = classification_report(test_data['y'], preds, output_dict=True)
    
    logs = create_logs(epochs=epochs,
                        lr=lr,
                        loss_func=loss_func,
                        optimizer=optimizer,
                        batch_size=batch_size,
                        max_length=max_length, 
                        padding=padding, 
                        truncation=truncation, 
                        pretrained_model_name=pretrained_model_name,
                        model_type=str(model),
                        exp_title=exp_title,
                        exp_index=exp_index,
                        execution_time=execution_time,
                        train_losses_avg=train_losses_avg,
                        valid_losses_avg=valid_losses_avg,
                        train_losses_epoches=train_losses_epoches,
                        valid_losses_epoches=valid_losses_epoches,
                        report=report)
    
    save_exp(model, pretrained_model_name, exp_title, exp_index, root, logs)


def save_exp(model, 
            pretrained_model_name, 
            exp_title, 
            exp_index,
            root,
            logs=False):
    
    save_model(model, f'{root}/models/model_{exp_title}_{exp_index}_{pretrained_model_name}_{str(model)}.safetensors')
    print('Saving model', exp_index)

    if logs:
        save_dict(logs, f'{root}/logs.json')
        print('Zapis do logs.json')

def get_number(model_file, exp_title):
    model_file = model_file[len(f'model_:{exp_title}_'):]
    number = model_file.split('_')[0] 
    return number

def get_pretrained_name(model_file, exp_title):
    # bad code:<
    model_file = model_file[len(f'model_:{exp_title}_'):]
    pretrained_name = model_file.split('_')[1] 
    return pretrained_name

def get_model_name(model_file, exp_title):
    # bad code:<
    model_file = model_file[len(f'model_:{exp_title}_'):]
    model_name = '_'.join(model_file.split('_')[2:]).replace(".safetensors", "")
    return model_name
    
def load_models(exp_title, model_names, device):
    models = []
    pretrained_names = []
    model_names = []
    matching_files = {
                        model_file: get_number(model_file, exp_title) 
                        for model_file in os.listdir('models') 
                        if model_file.startswith(f'model_{exp_title}')
                     }
    matching_files = sorted(matching_files, key=matching_files.get)
    for i, model_file in enumerate(matching_files):
        pretrained_name = get_pretrained_name(model_file, exp_title)
        model_name = get_model_name(model_file, exp_title)
        model = get_model(pretrained_name, globals()[model_name], device)
        
        load_model(model, f'models/{model_file}')
        model_names.append(model_name)
        pretrained_names.append(pretrained_name)
        models.append(model)
            
    return  models, pretrained_names, model_names

def load_models_gen(exp_title, model_names, device):
    models = []
    pretrained_names = []
    model_names = []
    matching_files = {
                        model_file: get_number(model_file, exp_title) 
                        for model_file in os.listdir('models') 
                        if model_file.startswith(f'model_{exp_title}')
                     }
    matching_files = sorted(matching_files, key=matching_files.get)
    yield len(matching_files)
    
    for i, model_file in enumerate(matching_files):
        pretrained_name = get_pretrained_name(model_file, exp_title)
        model_name = get_model_name(model_file, exp_title)
        model = get_model(pretrained_name, globals()[model_name], device)
        
        load_model(model, f'models/{model_file}')
        model_names.append(model_name)
        pretrained_names.append(pretrained_name)
        models.append(model)
            
        yield  model, pretrained_name, model_name