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

def get_reprezentation(model, device, num_batches, trainX, trainY, model_type='siamese'):
    all_z = []
    all_labels = np.array([])
    model.eval()
    for i, (x, y) in enumerate(zip(trainX, trainY), start=1):
        with torch.no_grad():
            x_input = x.to(device).unsqueeze(1)
            if model_type == 'triplet':
                z, _, _ = model(x_input, x_input, x_input)
            else:
                z, _,  = model(x_input, x_input)
        z = z.to('cpu').detach().numpy()
        all_z.append(z[0])
        all_labels = np.append(all_labels, y)
        if i >= num_batches:
            break
    model.train()
    return all_z, all_labels


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

def create_single_subplot_tsne(model, model_name, model_type, data, device, transform_model, test, standarize=[True, True], num_batches=np.inf):
    train_X, train_Y, test_X, test_Y = data
    z_train, labels_train = get_reprezentation(model, device, num_batches, train_X, train_Y, model_type)
    z_test, labels_test = get_reprezentation(model, device, num_batches, test_X, test_Y, model_type)
        
    if standarize[0]:
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
        marker=dict(color=labels_train, colorscale=colorscale),
        text=labels_train
    )
    subplot_cross = go.Scatter(
        x=z_transformed_test[:, 0],
        y=z_transformed_test[:, 1],
        mode='markers+text',
        name=model_name,
        marker=dict(symbol='cross', size=10, color=labels_test, colorscale=colorscale),
        text=labels_test
    )

    """if test:
        accuracy_score = calculate_logistic_regression_accuracy(z_transformed_train, labels_train, z_transformed_test, labels_test)
    else:
        accuracy_score = calculate_logistic_regression_accuracy(z_transformed_train, labels_train)"""
        
        
    rand_score = get_rand_index(np.concatenate((z_transformed_train, z_transformed_test), axis=0), 
                                np.concatenate((labels_train, [x+37 for x in labels_test]), axis=0))
    return subplot, subplot_cross, rand_score



def plot_latent_multi(models, models_names,  models_types, data, device, transform_model, test, standarize=[True, True], num_batches=np.inf):
    num_models = len(models)
    fig = sp.make_subplots(rows=ceil(num_models/2), cols=2, subplot_titles=models_names)
    scores = []
    for i, model in enumerate(models):
        subplot, subplot_cross, score = create_single_subplot_tsne(model, models_names[i], models_types[i], data, device, 
                                                         transform_model, test, standarize, num_batches)

        fig.add_trace(subplot, row=i//2+1, col=i%2+1)
        fig.add_trace(subplot_cross, row=i//2+1, col=i%2+1)
        scores.append(score)
        

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

    
def name_to_type(model_name):
    return  model_name.split('_')[-1].split('.')[0].upper()


