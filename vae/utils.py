import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import time
from math import ceil
import json
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE
from safetensors.torch import  save_model, load_model
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from models import VariationalAutoencoder, Autoencoder


def load_models(exp_title, num_indexes, model_names_types, latent_dims, device):
    models = []
    model_names = []
    for i, model_type in zip(range(num_indexes),model_names_types):
        model_name = f'generator_{exp_title}_{i}_{model_type}.safetensors'
        model_names.append(model_name)
        if type(latent_dims) is not list:
            acutal_latent_size = latent_dims
        else:
            acutal_latent_size = latent_dims[i]
        print(model_type, acutal_latent_size)
        if model_type == 'ae':
            model = Autoencoder(acutal_latent_size).to(device)
        else:
            model = VariationalAutoencoder(acutal_latent_size).to(device)
        load_model(model, model_name) 
        models.append(model)
            
    return  models, model_names


def download_dataset():
    mnist  = torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True)
    return  mnist


def get_dataloader(dataset, batch_size):
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    return data

def get_colorscale():
    cmap = plt.get_cmap('tab10')
    color_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rgb_colors = [cmap(i) for i in color_indices]
    colorscale = set()
    for color in rgb_colors:
        r, g, b, _ = [int(255 * x) for x in color]
        colorscale.add(f'rgb({r},{g},{b})')
    return list(colorscale)

def get_rand_index(X_data, all_labels):
  kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')
  clusters = kmeans.fit_predict(X_data)
  return adjusted_rand_score(all_labels, clusters)


def create_single_subplot_tsne(autoencoder, model_name, data, device, num_batches=100, latent_size=2):
    all_z = np.array([])
    all_labels = np.array([])
    for i, (x, y) in enumerate(data, start=1):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        all_z = np.append(all_z, z)
        all_labels = np.append(all_labels, y)
        if i >= num_batches:
            break
    all_z = all_z.reshape(-1, latent_size)
    all_labels = all_labels.reshape(-1)
    print(all_z.shape)
    tsne = TSNE(n_components=2, random_state=0)
    if latent_size != 1:
        z_tsne = tsne.fit_transform(all_z)

        colorscale = get_colorscale()
        subplot = go.Scatter(
            x=z_tsne[:, 0],
            y=z_tsne[:, 1],
            mode='markers',
            name=model_name,
            marker=dict(color=all_labels, colorscale=colorscale)
        )
        rand_score = get_rand_index(z_tsne, all_labels)
    else:
        colorscale = get_colorscale()
        subplot = go.Scatter(
            x=all_z[:, 0],
            y=all_z[:, 0],
            mode='markers',
            name=model_name,
            marker=dict(color=all_labels, colorscale=colorscale)
        )
        rand_score = get_rand_index(all_z, all_labels)
    
    return subplot, rand_score


def plot_latent_tsne(autoencoder, data, device, num_batches=100):
    all_z = np.array([])
    all_labels = np.array([])
    for i, (x, y) in enumerate(data, start=1):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        all_z = np.append(all_z, z)
        all_labels = np.append(all_labels, y)
        if i >= num_batches:
            break
    all_z = all_z.reshape(-1, 2)
    all_labels = all_labels.reshape(-1)

    tsne = TSNE(n_components=2, random_state=0)
    z_tsne = tsne.fit_transform(all_z)



    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=all_labels, cmap='tab10')
    plt.colorbar()
    plt.show()


def plot_latent(autoencoder, data, device, num_batches=100):
    for i, (x, y) in enumerate(data, start=1):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i >= num_batches:
            plt.colorbar()
            break
        
def create_single_subplot(model, model_name, data, device, num_batches=100):
    x_values = []
    y_values = []
    X_data = []
    color_values = []
        
    for j, (x, y) in enumerate(data, start=1):
        z = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        x_values.extend(z[:, 0])
        y_values.extend(z[:, 1])
        X_data.extend(z[:, :])
        color_values.extend(y.tolist())
        if j >= num_batches:
            break

    colorscale = get_colorscale()

    subplot = go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        name=model_name,
        marker=dict(color=color_values, colorscale=list(colorscale))
    )
    rand_score = get_rand_index(X_data, color_values)
    
    return subplot, rand_score

def plot_latent_multi(models, exp_title, models_names, data, device, plot_type, num_batches=100, latent_sizes=None):
    data.shuffle = False
    num_models = len(models)
    fig = sp.make_subplots(rows=ceil(num_models/2), cols=2, subplot_titles=exp_title)
    scores = []
    for i, model in enumerate(models):
        if plot_type == 'latent':
            subplot, rand_score = create_single_subplot(model, models_names[i], data, device, num_batches)
        elif plot_type == 'tsne':
            if latent_sizes:
                subplot, rand_score = create_single_subplot_tsne(model, models_names[i], data, device, num_batches, latent_sizes[i])
            else:
                subplot, rand_score = create_single_subplot_tsne(model, models_names[i], data, device, num_batches)
        print(models_names[i], i//2+1, i%2+1)
        fig.add_trace(subplot, row=i//2+1, col=i%2+1)
        scores.append(rand_score)

    fig.update_layout(
        title=f"Latent Space Visualization {plot_type}",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        width=1200,
        height=500*ceil(num_models/2)# Ustaw szerokość figury
    )

    fig.show()

    ranking = [(model_name, score) for model_name, score in zip(models_names, scores)]
    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    print('Adjusted rand score')
    for line in ranking:
      print(f'{line[0]}: {line[1]}')
    data.shuffle = True

def loss_0(x, x_hat):
  return 0.0

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
    
    
def loss_mse(x, x_hat):
    return ((x - x_hat)**2).sum() 


def train(autoencoder, 
          data, 
          device,
          lr=0.0002,
          epochs=20, 
          optimizer=torch.optim.Adam, 
          loss_func=loss_mse,
          l1_weight=0.00,
          use_kl=True,
          b_value=1):
    start_time = time.time()
    opt = optimizer(autoencoder.parameters(), lr=lr)
    kl_avg = []
    l1_avg = []
    loss_avg = []
    loss_sum_avg = []
    for epoch in range(epochs):
        l1_list = np.array([])
        kl_list = np.array([])
        loss_list = np.array([])
        loss_sum_list = np.array([])
        for x, _ in tqdm(data):
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)

            if l1_weight:
              l1_loss = sum(p.abs().sum() for p in autoencoder.parameters()).to('cuda')
              l1_loss = l1_loss * l1_weight
              l1_list = np.append(l1_list, l1_loss.item())
            else:
              l1_loss = 0.0
              l1_list = np.append(l1_list, l1_loss)
            
            
            new_loss = loss_mse(x, x_hat)
            if loss_func is loss_0:
                new_loss.zero_()
            
            if use_kl:
              kl = b_value * autoencoder.encoder.kl
            else:
              kl = 0.0
              
            loss = new_loss + kl + l1_loss
            loss.backward()
            opt.step()
            
            try:
                kl_list = np.append(kl_list, kl.item())
            except AttributeError:
                kl_list = np.append(kl_list, kl)
            loss_list = np.append(loss_list, new_loss.item())
            
            loss_sum_list = np.append(loss_sum_list, loss.item())
            
        print(f"Epoch [{epoch+1}/{epochs}], avg_loss: {np.mean(loss_sum_list):.4f}")
        l1_avg.append(np.mean(l1_list))
        kl_avg.append(np.mean(kl_list))
        loss_avg.append(np.mean(loss_list))
        loss_sum_avg.append(np.mean(loss_sum_list))
    execution_time = time.time() - start_time
    losses_scores = {
        'l1': l1_avg,
        'kl': kl_avg,
        'basic_loss': loss_avg,
        'sum_loss': loss_sum_avg
        }
    return autoencoder, losses_scores, execution_time

def create_logs(epochs,
                lr,
                batch_size,
                latent_dims,
                model_name,
                optimizer,
                exp_title,
                exp_index,
                execution_time,
                loss_func,
                l1_weight,
                b_value,
                use_kl,          
                history):
    logs = history.copy()
    logs['epochs'] = epochs
    logs['lr'] = lr
    logs['batch_size'] = batch_size
    logs['latent_dims'] = latent_dims
    logs['model_name'] = model_name
    logs['optimizer'] = str(optimizer)
    logs['exp_title'] = exp_title
    logs['exp_index'] = exp_index
    logs['loss_func'] = str(loss_func)
    logs['l1_weight'] = l1_weight
    logs['b_value'] = b_value
    logs['use_kl'] = use_kl
    logs['execution_time'] = execution_time

    return logs

def construct_vector(latent_dims):
    return (-3, 3), (-3, 3)
    
def save_samples(data,
                 model,
                 model_name,
                 exp_title,
                 exp_index,
                 latent_dims,
                 device,
                 root,
                 logs=False,
                 show=False):
    # file_name = f'{exp_title}_{exp_index}.png'
    #zapis plot_latent(vae, data, device)
    
    # r0, r1 = construct_vector
    #zapis plot_reconstructed(vae, device, r0=r0, r1=r1)
    
    #x, y = next(iter(data))
    #x_1 = x[y == 1][1].to(device)
    #x_2 = x[y == 0][1].to(device)
    #zapis interpolate(vae, x_1, x_2, n=20)
    
    save_model(model, f'{root}/generator_{exp_title}_{exp_index}_{model_name}.safetensors')
    print('Saving model', exp_index)

    if logs:
        dict_to_json(logs, f'{root}/logs.json')
        print('Zapis do logs.json')

    
    if show:
        plot_latent(model, data, device)
        
        r0, r1 = construct_vector(latent_dims=latent_dims)
        plot_reconstructed(model, device, r0=r0, r1=r1)
        
        x, y = next(iter(data))
        x_1 = x[y == 1][1].to(device)
        x_2 = x[y == 0][1].to(device)
        interpolate(model, x_1, x_2, n=20)
        
        
def dict_to_json(slownik, nazwa_pliku):
    print(slownik)
    try:
        with open(nazwa_pliku, 'a') as plik:
            plik.write(json.dumps(slownik) + '\n')
        print(f'Słownik został zapisany do pliku {nazwa_pliku}.')
    except Exception as e:
        print(f'Błąd podczas zapisywania słownika do pliku: {e}')

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


def do_exp(latent_dims, batch_size, epochs, lr, optimizer, loss_func, l1_weight, model_name, exp_title, data, exp_index, b_value, model, device, use_kl, root):
    autoencoder, history, execution_time = train(autoencoder=model,
                                                data=data,
                                                device=device,
                                                lr=lr,
                                                epochs=epochs,
                                                optimizer=optimizer,
                                                loss_func=loss_func,
                                                l1_weight=l1_weight,
                                                use_kl=use_kl,
                                                b_value=b_value)
    logs = create_logs(epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                latent_dims=latent_dims,
                model_name=model_name,
                optimizer=optimizer,
                exp_title=exp_title,
                exp_index=exp_index,
                execution_time=execution_time,
                loss_func=loss_func,
                l1_weight=l1_weight,
                b_value=b_value,
                use_kl=use_kl,          
                history=history)
    save_samples(data, model, model_name, exp_title, exp_index, latent_dims, device, root, logs)
    
