import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.distributions
from math import ceil
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from sklearn.metrics import adjusted_rand_score
from models import *
from utils.train import *

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
    
def get_colorscale():
    cmap = plt.get_cmap('tab10')
    color_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rgb_colors = [cmap(i) for i in color_indices]
    colorscale = set()
    for color in rgb_colors:
        r, g, b, _ = [int(255 * x) for x in color]
        colorscale.add(f'rgb({r},{g},{b})')
    return list(colorscale)

def get_reprezentation(model, device, num_batches, train_dataloader, rep_idx):
    all_z = np.array([])
    all_preds = np.array([])
    all_labels = np.array([])
    model.eval()
    for i, batch in enumerate(train_dataloader, start=1):
        with torch.no_grad():
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch

            output = model.get_rep(sent_id, mask)
            preds, z=  output[0], output[rep_idx+1]
            preds=preds.detach().cpu().numpy()
            z=z.detach().cpu().numpy()
            labels=labels.detach().cpu().numpy()
        if all_z.size != 0:
            all_z = np.vstack((all_z, z))
        else:
            all_z = z
        if all_preds.size != 0:
            all_preds = np.vstack((all_preds, preds))
        else:
            all_preds = preds
        #all_z = np.append(all_z, z)
        #all_preds = np.append(all_preds, preds)
        all_labels = np.append(all_labels, labels)
        if i >= num_batches:
            break
    model.train()
    all_preds = np.argmax(all_preds, axis=1)
    return all_z, all_labels, all_preds


def calculate_logistic_regression_accuracy(X_train, y_train, X_test=False, y_test=False):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    if X_test is False:
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def get_rand_index(X_data, all_labels):
  kmeans = KMeans(n_clusters=40, random_state=0, n_init='auto')
  clusters = kmeans.fit_predict(X_data)
  return adjusted_rand_score(all_labels, clusters)

def create_single_subplot_tsne(model, model_name, plots_names, data, device, transform_model, test, rep_idx, num_batches=np.inf):
    train_dataloader, _, test_dataloader, _ = data
    z_train, labels_train, preds_train = get_reprezentation(model, device, num_batches, train_dataloader, rep_idx)
    z_test, labels_test, preds_test = get_reprezentation(model, device, num_batches, test_dataloader, rep_idx)
    
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)
    
    model_transform = deepcopy(transform_model)
    
    if model_transform is not False:
        
        if test:
            z_transformed_train = model_transform.fit_transform(np.array(z_train))
            z_transformed_test = model_transform.transform(np.array(z_test))
        else:
            z_transformed = model_transform.fit_transform(np.concatenate((z_train, z_test), axis=0))
            z_transformed_train, z_transformed_test = z_transformed[:len(z_train)], z_transformed[len(z_train):]
        
    else:
        z_transformed_train = np.array(z_train)
        z_transformed_test= np.array(z_test)
    bad_preds = np.where(preds_test != labels_test, 'cross', 'circle')
            
    colorscale = get_colorscale()
    """subplot = go.Scatter(
        x=z_transformed_train[:, 0],
        y=z_transformed_train[:, 1],
        mode='markers',
        name=model_name,
        marker=dict(color=labels_train, colorscale=colorscale),
        text=labels_train
    )
    subplot = go.Scatter(
        x=z_transformed_test[:, 0],
        y=z_transformed_test[:, 1],
        mode='markers+text',
        name=model_name,
        marker=dict(symbol='cross', size=10, color=preds_test, colorscale=colorscale),
        text=preds_test
    )"""
    subplot = go.Scatter(
        x=z_transformed_test[:, 0],
        y=z_transformed_test[:, 1],
        mode='markers',
        name=plots_names,
        marker=dict(color=labels_test, colorscale=colorscale, symbol=bad_preds),
        text=labels_test
    )

    if test:
        score = calculate_logistic_regression_accuracy(z_transformed_train, labels_train, z_transformed_test, labels_test)
    else:
        score = calculate_logistic_regression_accuracy(z_transformed_train, labels_train)
        
        
    """score = get_rand_index(np.concatenate((z_transformed_train, z_transformed_test), axis=0), 
                                np.concatenate((labels_train, [x+37 for x in labels_test]), axis=0))"""
    return subplot, _, score



"""def plot_latent_multi(models, pretrained_names, model_names, device, transform_model, test, max_lengths, paddings,
                      truncations, batch_sizes, rep_idx, num_batches=np.inf):
    num_models = len(models)
    fig = sp.make_subplots(rows=ceil(num_models/2), cols=2, subplot_titles=pretrained_names)
    scores = []
    
    for i, model in enumerate(models):
        data = tokenize_data(pretrained_names[i], max_lengths[i], paddings[i],
                             truncations[i], batch_sizes[i], device, test_dl=True)
        print("i")
        subplot, subplot_cross, score = create_single_subplot_tsne(model, model_names[i],
                                                                   pretrained_names[i], data, device, 
                                                         transform_model, test, rep_idx,  num_batches)

        fig.add_trace(subplot, row=i//2+1, col=i%2+1)
        #fig.add_trace(subplot_cross, row=i//2+1, col=i%2+1)
        scores.append(score)
        

    fig.update_layout(
        title=f"Latent Space Visualization {str(transform_model)}",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        width=1200,
        height=500*ceil(num_models/2)# Ustaw szerokość figury
    )

    fig.show()

    ranking = [(model_name, score) for model_name, score in zip(model_names, scores)]
    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    print('Adjusted logistic regression accuracy')
    for line in ranking:
        print(f'{line[0]}: {line[1]}')"""


def plot_latent_multi_gen(models_gen, device, transform_model, test, max_lengths, paddings,
                      truncations, batch_sizes, rep_idx, true_scores, plots_names, num_batches=np.inf):
    num_models = next(models_gen)
    fig = sp.make_subplots(rows=ceil(num_models/2), cols=2, subplot_titles=plots_names)
    scores = []
    
    for i, (model, pretrained_name, model_name) in enumerate(models_gen):

        data = tokenize_data(pretrained_name, max_lengths[i], paddings[i],
                             truncations[i], batch_sizes[i], device, test_dl=True)
        print(i)
        subplot, subplot_cross, score = create_single_subplot_tsne(model, model_name,
                                                                   plots_names[i], data, device, 
                                                         transform_model, test, rep_idx,  num_batches)

        fig.add_trace(subplot, row=i//2+1, col=i%2+1)
        #fig.add_trace(subplot_cross, row=i//2+1, col=i%2+1)
        scores.append((f'{pretrained_name}_{model_name}', score))
        model.to('cpu')
        del model
        del data
        torch.cuda.empty_cache()
        

    fig.update_layout(
        title=f"Latent Space Visualization {str(transform_model)}",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        width=1200,
        height=500*ceil(num_models/2)# Ustaw szerokość figury
    )

    fig.show()

    ranking = sorted(true_scores, key=lambda x: x[1], reverse=True)
    print('Adjusted logistic regression accuracy')
    for line in ranking:
        print(f'{line[0]}: {line[1]}')
        
           
def name_to_type(model_name):
    return  model_name.split('_')[-1].split('.')[0].upper()


