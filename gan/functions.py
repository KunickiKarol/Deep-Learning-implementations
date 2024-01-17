


import os

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import json
import pandas as pd
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm.notebook import tqdm

def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


# In[7]:


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax], stats), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    # show_images(next(iter(dl))[0], nmax)
    for images, _ in dl:
        show_images(images, nmax)
        break


# In[2]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# In[ ]:


class Generator_lsgan(nn.Module):
    def __init__(self, latent_size):
        super(Generator_lsgan, self).__init__()

        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_size, 128 * self.init_size ** 2))  # 64 * 64 * 3 for 64x64x3 images

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_lsgan(nn.Module):
    def __init__(self):
        super(Discriminator_lsgan, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# <h2>Podgląd próbek danych znajdujących się w pojedynczym pakiecie</h2>

# In[8]:


#show_batch(train_dl)


# <h2>Implementacja wrappera dla obiektów klasy <i>Dataloader</i></h2>

# In[9]:


def to_device(data, device):
    """
    Przeniesienie wybranej instancji struktury danych
    do pamięci wybranego urządzenia
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Inicjalizacja wrappera"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """
        Zwrócenie jednego pakietu danych po przeniesieniu go
        do pamięci wybranego urządzenia
        """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """
        Zwrócenie liczby pakietów danych znajdujących się wewnątrz obiektu klasy Dataloader
        """
        return len(self.dl)



def zapisz_slownik_do_pliku_json(slownik, nazwa_pliku):
    try:
        with open(nazwa_pliku, 'a') as plik:
            plik.write(json.dumps(slownik) + '\n')
        print(f'Słownik został zapisany do pliku {nazwa_pliku}.')
    except Exception as e:
        print(f'Błąd podczas zapisywania słownika do pliku: {e}')

def wczytaj_do_dataframe(nazwa_pliku):
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




def fit(model,
        train_dl,
        batch_size,
        lr,
        criterion,
        optimizer,
        epochs,
        latent_size,
        device,
        mse=False
        ):
    start_time = time.time()
    model["discriminator"].train()
    model["generator"].train()
    torch.cuda.empty_cache()

    # Inicjalizacja struktur danych do przechowywania wartości funkcji straty
    # oraz wyników dla rzeczywistych i wygenerowanych obrazów
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Inicjalizacja optymalizatora

    
    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        real_score_per_epoch = []
        fake_score_per_epoch = []
        for real_images, _ in tqdm(train_dl):
            # Uczenie modelu dyskryminatora
            # Inicjalizacja wartości gradientu obliczanego przez optymalizator

            optimizer["discriminator"].zero_grad()

            # Wygenerowanie predykcji dla rzeczywistych obrazów
            # oraz obliczenie wartości funkcji straty
            real_preds = model["discriminator"](real_images)
            real_targets = torch.ones(real_images.size(0), 1, device=device)
            real_loss = criterion["discriminator"](real_preds, real_targets)
            cur_real_score = torch.mean(real_preds).item()

            # Wygenerowanie sztucznych obrazów
            if model["model_name"] =='lsgan':
                latent = torch.randn(64, latent_size, device=device)
            else:
                latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = model["generator"](latent)

            # Wygenerowanie predykcji dla sztucznych obrazów
            # oraz obliczenie wartości funkcji straty
            fake_preds = model["discriminator"](fake_images)
            fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
            fake_loss = criterion["discriminator"](fake_preds, fake_targets)
            cur_fake_score = torch.mean(fake_preds).item()

            real_score_per_epoch.append(cur_real_score)
            fake_score_per_epoch.append(cur_fake_score)
            
            # Aktualizacja wag dyskryminatora
            if mse:
                loss_d = 0.5 * (real_loss + fake_loss)
            else:
                loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer["discriminator"].step()
            loss_d_per_epoch.append(loss_d.item())

            # Uczenie modelu generatora
            # Inicjalizacja wartości gradientu obliczanego przez optymalizator
            optimizer["generator"].zero_grad()

            # Wygenerowanie sztucznych obrazów
            if model["model_name"] =='lsgan':
                latent = torch.randn(64, latent_size, device=device)
            else:
                latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = model["generator"](latent)

            # Próba oszukania modelu dyskryminatora
            preds = model["discriminator"](fake_images)
            
            
            if model["model_name"] =='lsgan':
                targets = torch.ones((64, 1), device=device)
            else:
                targets = torch.ones(batch_size, 1, device=device)
            
            loss_g = criterion["generator"](preds, targets)

            # Aktualizacja wag generatora
            loss_g.backward()
            optimizer["generator"].step()
            loss_g_per_epoch.append(loss_g.item())

        # Zapisanie wartości funkcji straty i średnich wartości predykcji w danej epoce
        losses_g.append(np.mean(loss_g_per_epoch))
        losses_d.append(np.mean(loss_d_per_epoch))
        real_scores.append(np.mean(real_score_per_epoch))
        fake_scores.append(np.mean(fake_score_per_epoch))


        # Zapisanie wartości funkcji straty i wyników dla ostatniej porcji danych
        print(f"Epoch [{epoch+1}/{epochs}], loss_g: {losses_g[-1]:.4f}, loss_d: {losses_d[-1]:.4f}, real_score: {real_scores[-1]:.4f}, fake_score: {fake_scores[-1]:.4f}")
    
    execution_time = time.time() - start_time
    losses_scores = {
            'losses_g': losses_g,
            'losses_d': losses_d,
            'real_scores': real_scores,
            'fake_scores': fake_scores
            }

    return losses_scores, model, execution_time




def get_basic_generator(latent_size, actvation=False):
    if actvation:
        actvation = actvation
    else:
        actvation = nn.ReLU(True)
    generator = nn.Sequential(
        # in: latent_size x 1 x 1

        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        actvation,
        # out: 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        actvation,
        # out: 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        actvation,
        # out: 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        actvation,
        # out: 64 x 32 x 32

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: 3 x 64 x 64
    )
    return generator





def get_lsgan_generator(latent_size):

    return Generator_lsgan(latent_size)


# In[17]:


def get_generator(device, latent_size, generator_name, weights_normal=False, actvation=False):
    if generator_name == 'basic':
        generator = get_basic_generator(latent_size, actvation)
    elif generator_name =='lsgan':
        generator = get_lsgan_generator(latent_size)

    if weights_normal:
        generator.apply(weights_init_normal)
    generator = to_device(generator, device)

    return generator


# In[18]:


def get_basic_discriminator(actvation=False):
    if actvation:
        actvation = actvation
    else:
        actvation = nn.LeakyReLU(0.2, inplace=True)
    discriminator = nn.Sequential(
    # in: 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    actvation,
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    actvation,
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    actvation,
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    actvation,
    # out: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid()
    )

    return discriminator


# In[ ]:


def get_lsgan_discriminator():


    return Discriminator_lsgan()


# In[3]:


def get_discriminator(device, discriminator_name, weights_normal=False, actvation=False):
    print(discriminator_name)
    if discriminator_name =='basic':
        discriminator = get_basic_discriminator(actvation)
    elif discriminator_name =='lsgan':
        discriminator = get_lsgan_discriminator()

    
    if weights_normal:
        discriminator.apply(weights_init_normal)
    discriminator = to_device(discriminator, device)

    return discriminator


# In[20]:


def create_logs(history,
                epochs,
                lr,
                batch_size,
                latent_size,
                model_name,
                criterions,
                optimizers,
                exp_title,
                exp_index,
                time):
    logs = history.copy()
    logs['epochs'] = epochs
    logs['lr'] = lr
    logs['batch_size'] = batch_size
    logs['latent_size'] = latent_size
    logs['model_name'] = model_name
    logs['criterions'] = str(criterions)
    logs['optimizers'] = str(optimizers)
    logs['exp_title'] = exp_title
    logs['exp_index'] = exp_index
    logs['time'] = time

    return logs


# In[21]:



def save_samples(batch_size,
                 latent_size,
                 model,
                 exp_title,
                 exp_index,
                 device,
                 stats,
                 root,
                 logs=False,
                 show=False):
    if model["model_name"] =='lsgan':
        fixed_latent = torch.randn(64, latent_size, device=device)
    else:
        fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    #fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    fake_images = model['generator'](fixed_latent)


    fake_fname = f'{exp_title}_{exp_index}.png'
    save_image(denorm(fake_images, stats), os.path.join(root, fake_fname), nrow=8)

    torch.save(model['generator'].state_dict(),
               f'{root}/generator_{exp_title}_{exp_index}_{model["model_name"]}')
    torch.save(model['discriminator'].state_dict(),
               f'{root}/discriminator_{exp_title}_{exp_index}_{model["model_name"]}')

    if logs:
        zapisz_slownik_do_pliku_json(logs, f'{root}/logs.json')
        print('Zapis do logs.json')

    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))




def load_model(device,
               model_name, 
               discriminator_path,  
               generator_path,
               latent_size=128,
               discriminator_name=False, 
               generator_name=False):
    if discriminator_name:
        pass
    else:
        discriminator = get_discriminator(device, model_name)
    if generator_name:
        pass
    else:
        generator = get_generator(device, latent_size, model_name)
        
    discriminator.load_state_dict(torch.load(f'./{discriminator_path}'))
    generator.load_state_dict(torch.load(f'{generator_path}'))
    
    model = {
        "discriminator": discriminator.to(device),
        "generator": generator.to(device),
        'model_name': model_name
    }
    return model


# In[47]:

