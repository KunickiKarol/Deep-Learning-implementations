import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Small_Ext_Arch(nn.Module):
    def __init__(self, pretrained_model, input_size=768):
        super().__init__()
        self.name = 'Small_Ext_Arch'
        self.pretrained_model = pretrained_model
        self.fc1 = nn.Linear(input_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.pretrained_model(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.softmax(x)
        return x
    
    def get_rep(self, sent_id, mask):
        _, cls_hs = self.pretrained_model(sent_id, attention_mask=mask, return_dict=False)
        x_fc1 = self.fc1(cls_hs)
        x = self.softmax(x_fc1)
        return x, cls_hs, x_fc1, x_fc1, x_fc1
    
    def set_name(self, new_name):
        self.name = new_name
    
    def __str__(self):
        return self.name
    
    
class Ext_Arch(nn.Module):
    def __init__(self, pretrained_model, input_size=768):
        super().__init__()
        self.name = 'Ext_Arch'
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.pretrained_model(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def get_rep(self, sent_id, mask):
        _, cls_hs = self.pretrained_model(sent_id, attention_mask=mask, return_dict=False)
        x_fc1 = self.fc1(cls_hs)
        x = self.relu(x_fc1)
        x = self.dropout(x)
        x_fc2 = self.fc2(x)
        x = self.softmax(x_fc2)
        return x, cls_hs, x_fc1, x_fc2, x_fc2
    
    def set_name(self, new_name):
        self.name = new_name
    
    def __str__(self):
        return self.name
    

class Big_Ext_Arch(nn.Module):
    def __init__(self, pretrained_model, input_size=768):
        super().__init__()
        self.name = 'Big_Ext_Arch'
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        # Dodane warstwy
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)  # Dodatkowa warstwa
        self.fc3 = nn.Linear(256, 2)   # Dodatkowa warstwa
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.pretrained_model(sent_id, attention_mask=mask, return_dict=False)
        
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)  # Dodatkowa warstwa
        x = self.relu(x)  # Dodatkowa warstwa
        x = self.dropout(x)  # Dodatkowa warstwa
        
        x = self.fc3(x)  # Dodatkowa warstwa
        x = self.softmax(x)
        
        return x
    
    def get_rep(self, sent_id, mask):
        _, cls_hs = self.pretrained_model(sent_id, attention_mask=mask, return_dict=False)
        
        x_fc1 = self.fc1(cls_hs)
        x = self.relu(x_fc1)
        x = self.dropout(x)
        
        x_fc2 = self.fc2(x)  # Dodatkowa warstwa
        x = self.relu(x_fc2)  # Dodatkowa warstwa
        x = self.dropout(x)  # Dodatkowa warstwa
        
        x_fc3 = self.fc3(x)  # Dodatkowa warstwa
        x = self.softmax(x_fc3)
        
        return x, cls_hs, x_fc1, x_fc2, x_fc3

    def set_name(self, new_name):
        self.name = new_name
    
    def __str__(self):
        return self.name
