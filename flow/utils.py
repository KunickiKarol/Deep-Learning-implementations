import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from keras.datasets.mnist import load_data
import time
from math import ceil
import json
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE
from safetensors.torch import  save_model, load_model
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from models import *


def download_dataset():
    (trainX, trainY), (testX, testy) = load_data()
    trainX = data_transform(trainX)
    testX = data_transform(testX)
    return  trainX, trainY, testX, testy

def data_transform(data_X):
    data_X = (np.float32(data_X) + torch.rand(data_X.shape).numpy()) / 255.
    data_X = data_X.clip(0, 1)
    data_X = torch.tensor(data_X.reshape(-1, 28 * 28))
    return data_X

def save_dict(input_dict, file_name):
    try:
        with open(file_name, 'a') as plik:
            plik.write(json.dumps(input_dict) + '\n')
        print(f'Słownik został zapisany do pliku {file_name}.')
    except Exception as e:
        print(f'Błąd podczas zapisywania słownika do pliku: {e}')
        
def get_dataloader(dataset, batch_size):
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    return data

def plot_img(normalizing_flow, logistic_distribution, title='tmp.png'):
    nb_data = 10
    fig, axs = plt.subplots(nb_data, nb_data, figsize=(10, 10))
    for i in range(nb_data):
        for j in range(nb_data):
            x = normalizing_flow.invert(logistic_distribution.sample().unsqueeze(0)).data.cpu().numpy()
            axs[i, j].imshow(x.reshape(28, 28).clip(0, 1), cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.savefig(title)
    plt.show()

def do_exp(model, data_train, data_val, device, lr, weight_decay, epochs, optimizer, l1_weight, distribution, batch_size, model_name, exp_title, exp_index, root):
    model_flow, history, execution_time = train(normalizing_flow=model, 
                                                dataloader_train=data_train, 
                                                dataloader_val=data_val,
                                                device=device, 
                                                epochs=epochs, 
                                                optimizer=optimizer, 
                                                l1_weight=l1_weight, 
                                                distribution=distribution)
    
    logs = create_logs(epochs=epochs,
                        lr=lr,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        model_name=model_name,
                        optimizer=optimizer,
                        exp_title=exp_title,
                        exp_index=exp_index,
                        execution_time=execution_time,
                        distribution=distribution,
                        l1_weight=l1_weight,          
                        history=history)
    
    save_samples(model_flow, distribution, model_name, exp_title, exp_index, root, logs)

def name_to_type(model_name):
    return  model_name.split('_')[-1].split('.')[0].upper()

def plot_sample(normalizing_flow, logistic_distribution, exp_title, exp_index, model_type):
    nb_data = 10
    fig, axs = plt.subplots(nb_data, nb_data, figsize=(10, 10))
    for i in range(nb_data):
        for j in range(nb_data):
            if model_type == 'GLOW':
                x = normalizing_flow.invert().data.cpu().numpy()
            elif model_type == "NICE":
                x = normalizing_flow.invert(logistic_distribution.sample().unsqueeze(0)).data.cpu().numpy()
            axs[i, j].imshow(x.reshape(28, 28).clip(0, 1), cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.savefig(f'model_{exp_title}_{exp_index}_{model_type}.png')
    plt.close()
    
def plot_pictures(normalizing_flows, logistic_distributions, exp_title, exp_indexes, model_names):
    model_types = [name_to_type(x) for x in model_names]
    num_models = len(normalizing_flows)
    num_rows = (num_models + 1) // 2  # Określ liczbę wierszy na podstawie liczby modeli
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    for i in range(len(normalizing_flows)):
        plot_sample(normalizing_flows[i], logistic_distributions[i], exp_title, exp_indexes[i], model_types[i])
        img_path = f'model_{exp_title}_{exp_indexes[i]}_{model_types[i]}.png'

        # Wyznacz indeksy wiersza i kolumny
        row_index = (exp_indexes[i]) // 2
        col_index = (exp_indexes[i]) % 2

        # Dodaj obraz do wykresu
        img = plt.imread(img_path)
        axes[row_index, col_index].imshow(img)
        axes[row_index, col_index].axis('off')
        axes[row_index, col_index].set_title(f"{model_names[i]} - {exp_indexes[i]}")

    plt.tight_layout()
    plt.show()
    
def save_samples(model_flow, 
                 distribution,
                 model_name, 
                 exp_title, 
                 exp_index,  
                 root, 
                 logs=False,
                 show=False):
    
    save_model(model_flow, f'{root}/model_{exp_title}_{exp_index}_{model_name}.safetensors')
    print('Saving model', exp_index)

    if logs:
        save_dict(logs, f'{root}/logs.json')
        print('Zapis do logs.json')

    
    if show:
        plot_sample(model_flow, distribution, exp_title, exp_index, model_name)
        
def create_logs(epochs,
                lr,
                batch_size,
                weight_decay,
                model_name,
                optimizer,
                exp_title,
                exp_index,
                execution_time,
                distribution,
                l1_weight,          
                history):
    logs = history.copy()
    logs['epochs'] = epochs
    logs['lr'] = lr
    logs['batch_size'] = batch_size
    logs['weight_decay'] = weight_decay
    logs['model_name'] = model_name
    logs['optimizer'] = str(optimizer)
    logs['exp_title'] = exp_title
    logs['exp_index'] = exp_index
    logs['distribution'] = str(distribution)
    logs['l1_weight'] = l1_weight
    logs['execution_time'] = execution_time

    return logs

def train(normalizing_flow, dataloader_train, dataloader_val, device, epochs, optimizer, l1_weight, distribution):
    opt = optimizer
    
    jacobian_log_train = []
    distribution_log_train = []
    training_loss_train = []
    l1_loss_train = []

    jacobian_log_val = []
    distribution_log_val = []
    validation_loss_val = []
    l1_loss_val = []
    
    start_time = time.time()
    len_train = len(dataloader_train)
    len_val = len(dataloader_val)
    for epoch in range(epochs):
        # Train
        log_jacob_list_train = []
        log_dist_list_train = []
        loss_train_list_train = []
        l1_loss_list_train = []

        for batch_train in tqdm(dataloader_train):
            z, log_jacobian = normalizing_flow(batch_train.to(device))
            
            if l1_weight:
                l1_loss_single_train = sum(p.abs().sum() for p in normalizing_flow.parameters()).to('cuda')
                l1_loss_single_train = l1_loss_single_train * l1_weight
                l1_loss_list_train.append(l1_loss_single_train.item())
            else:
                l1_loss_single_train = 0.0
                l1_loss_list_train.append(l1_loss_single_train)
                
            dist_log_train = distribution.log_pdf(z)
            log_likelihood_train = dist_log_train + log_jacobian
            loss_train = -log_likelihood_train.sum()

            opt.zero_grad()
            loss_train.backward()
            opt.step()
            
            log_jacob_list_train.append(-log_jacobian.sum().item())
            log_dist_list_train.append(-dist_log_train.sum().item())
            loss_train_list_train.append(loss_train.item())

        # Validation
        log_jacob_list_val = []
        log_dist_list_val = []
        loss_val_list_val = []
        l1_loss_list_val = []
        with torch.no_grad():
            for batch_val in tqdm(dataloader_val):
                z_val, log_jacobian_val = normalizing_flow(batch_val.to(device))
                
                if l1_weight:
                    l1_loss_single_val = sum(p.abs().sum() for p in normalizing_flow.parameters()).to('cuda')
                    l1_loss_single_val = l1_loss_single_val * l1_weight
                    l1_loss_list_val.append(l1_loss_single_val.item())
                else:
                    l1_loss_single_val = 0.0
                    l1_loss_list_val.append(l1_loss_single_val)
                    
                dist_log_val = distribution.log_pdf(z_val)
                log_likelihood_val = dist_log_val + log_jacobian_val
                loss_val = -log_likelihood_val.sum()

                log_jacob_list_val.append(-log_jacobian_val.sum().item())
                log_dist_list_val.append(-dist_log_val.sum().item())
                loss_val_list_val.append(loss_val.item())

        # Average metrics for train
        jacobian_log_train.append(np.mean(log_jacob_list_train)/len_train)
        distribution_log_train.append(np.mean(log_dist_list_train)/len_train)  
        training_loss_train.append(np.mean(loss_train_list_train)/len_train)
        l1_loss_train.append(np.mean(l1_loss_list_train)/len_train)
        
        # Average metrics for validation
        jacobian_log_val.append(np.mean(log_jacob_list_val)/len_val)
        distribution_log_val.append(np.mean(log_dist_list_val)/len_val)  
        validation_loss_val.append(np.mean(loss_val_list_val)/len_val)
        l1_loss_val.append(np.mean(l1_loss_list_val)/len_val)
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train: avg_loss: {training_loss_train[-1]:.4f}, "
              f"Validation: avg_loss: {validation_loss_val[-1]:.4f}")

    execution_time = time.time() - start_time
    losses_scores = {
        'train_loss': training_loss_train,
        'val_loss': validation_loss_val,
        'jacob_train': jacobian_log_train,
        'dist_train': distribution_log_train,
        'l1_train': l1_loss_train,
        'jacob_val': jacobian_log_val,
        'dist_val': distribution_log_val,
        'l1_val': l1_loss_val
    }
    return normalizing_flow, losses_scores, execution_time

def get_colorscale():
    cmap = plt.get_cmap('tab10')
    color_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rgb_colors = [cmap(i) for i in color_indices]
    colorscale = set()
    for color in rgb_colors:
        r, g, b, _ = [int(255 * x) for x in color]
        colorscale.add(f'rgb({r},{g},{b})')
    return list(colorscale)

def get_reprezentation(model, device, num_batches, trainX, trainY):
    all_z = []
    all_labels = np.array([])
    for i, (x, y) in enumerate(zip(trainX, trainY), start=1):
        with torch.no_grad():
            z, _ = model(x.view(1, -1).to(device))
        z = z.to('cpu').detach().numpy()
        all_z.append(z.squeeze())
        all_labels = np.append(all_labels, y)
        if i >= num_batches:
            break
    return all_z, all_labels

def get_reprezentation_fake(model, device, num_batches, trainX, trainY):
    all_z = []
    all_labels = np.array([])
    for i, (x, y) in enumerate(zip(trainX, trainY), start=1):
        with torch.no_grad():
            z, _ = model(x.view(1, -1).to(device))
        z = z.to('cpu').detach().numpy()
        all_z.append(x.view(1, -1).detach().numpy().squeeze())
        all_labels = np.append(all_labels, y)
        if i >= num_batches:
            break
        
    return all_z, all_labels

def calculate_logistic_regression_accuracy(X_train, y_train=False, X_test=False, y_test=False):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    if X_test is False:
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def create_single_subplot_tsne(model, model_name, data, device, transform_model, test=True, standarize=[True, True], num_batches=np.inf):
    trainX, trainY, testX, testY = data
    z_train, labels_train = get_reprezentation(model, device, num_batches, trainX, trainY)
    if test:
        z_test, labels_test = get_reprezentation(model, device, num_batches, testX, testY)
        
    if standarize[0]:
        scaler = StandardScaler()
        z_train = scaler.fit_transform(z_train)
        if test:
            z_test = scaler.transform(z_test)
    
    model_transform = deepcopy(transform_model)
    
    z_transformed_train = model_transform.fit_transform(np.array(z_train))
    
    if test:
        z_transformed_test = model_transform.transform(np.array(z_test))
    
    if standarize[1]:
        scaler = StandardScaler()
        z_transformed_train = scaler.fit_transform(z_transformed_train)
        if test:
            z_transformed_test = scaler.transform(z_transformed_test)
            
    colorscale = get_colorscale()
    subplot = go.Scatter(
        x=z_transformed_train[:, 0],
        y=z_transformed_train[:, 1],
        mode='markers',
        name=model_name,
        marker=dict(color=labels_train, colorscale=colorscale)
    )

    if test:
        accuracy_score = calculate_logistic_regression_accuracy(z_transformed_train, labels_train, z_transformed_test, labels_test)
    else:
        accuracy_score = calculate_logistic_regression_accuracy(z_transformed_train, labels_train)
    
    return subplot, accuracy_score



def plot_latent_multi(models, exp_title, models_names, data, device, transform_model, test=True, standarize=[True, True], num_batches=np.inf):
    data[0].shuffle = False
    data[2].shuffle = False
    num_models = len(models)
    fig = sp.make_subplots(rows=ceil(num_models/2), cols=2, subplot_titles=models_names)
    scores = []
    for i, model in enumerate(models):
        subplot, rand_score = create_single_subplot_tsne(model, models_names[i], data, device, transform_model, test, standarize, num_batches)
        print(models_names[i], i//2+1, i%2+1)

        fig.add_trace(subplot, row=i//2+1, col=i%2+1)
        scores.append(rand_score)
        

    fig.update_layout(
        title=f"Latent Space Visualization {str(transform_model)}",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        width=1200,
        height=500*ceil(num_models/2)# Ustaw szerokość figury
    )

    fig.show()

    ranking = [(model_name, score) for model_name, score in zip(models_names, scores)]
    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    print('Adjusted logistic regression accuracy')
    for line in ranking:
        print(f'{line[0]}: {line[1]}')
    data[0].shuffle = True
    data[2].shuffle = True

def load_models(exp_title, num_indexes, model_names_types, device, data_dim=28*28, hidden_dim=1000, nice_size=4):
    models = []
    model_names = []
    for i, model_type in zip(range(num_indexes),model_names_types):
        model_name = f'model_{exp_title}_{i}_{model_type}.safetensors'
        model_names.append(model_name)
        
        if type(data_dim) is not list:
            acutal_data_dim = data_dim
        else:
            acutal_data_dim = data_dim[i]
            
        if type(hidden_dim) is not list:
            acutal_hidden_dim = hidden_dim
        else:
            acutal_hidden_dim = hidden_dim[i]
            
        if type(nice_size) is not list:
            acutal_nice_size = nice_size
        else:
            acutal_nice_size = nice_size[i]
            
        print(model_type, acutal_data_dim)
        if model_type == 'NICE':
            model = NICE(acutal_data_dim, acutal_hidden_dim, acutal_nice_size).to(device)
        else:
            raise Exception(ValueError)
        load_model(model, model_name) 
        models.append(model)
            
    return  models, model_names  
    
"""

def plot_reconstructed(autoencoder, device, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def construct_vector(latent_dims):
    return (-3, 3), (-3, 3)
    
        

def to_df(nazwa_pliku):
    dane = []
    try:
        with open(nazwa_pliku, 'r') as plik:
            for linia in plik:
                slownik = json.loads(linia)
                dane.append(slownik)
        df = pd.DataFrame(dane)
        return df
    except Exception as e:
        print(f'Błąd podczas wczytywania pliku: {e}')
        return None
"""


    
