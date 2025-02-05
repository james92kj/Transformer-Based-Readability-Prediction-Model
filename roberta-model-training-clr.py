#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from transformers import (AutoTokenizer, AutoModel, 
                        AutoConfig, AutoModelForTokenClassification,
                        AutoModelForSequenceClassification)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from tqdm.auto import tqdm
from enum import Enum
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import torch
import pickle
import os
import math
import re
import json
import gc
import time


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# utility for setting the seed 
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


# In[ ]:


# parameters 
SEED = 42
MAX_LENGTH = 300

OUTPUT_FOLDER = "."
MODEL_ROOT = "./models"

os.makedirs(MODEL_ROOT, exist_ok=True)

n_epochs = 2
per_device_train_batch_size = 3
per_device_eval_batch_size = 2


# In[ ]:


class Task(Enum):
    TOKEN_CLASSIFICATION="token_classification"
    SEQUENCE_CLASSIFICATION = "sequence_classification"


# In[ ]:


df = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')


# In[ ]:


# setting the fold. 
df['kfold'] = -1

kf = KFold(n_splits=5, random_state=SEED, shuffle=True)

for f, (t_,v_) in enumerate(kf.split(X=np.arange(len(df)))):
    df.loc[v_,'kfold'] = f


# In[ ]:


df.kfold.value_counts()


# In[ ]:


# A utility to convert text to token ids and stack them 

def generate_data(model_name='roberta-base'):
    x_input_ids = []
    x_masks = []
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # save the tokenizer to a file 
    with open(f'{OUTPUT_FOLDER}/{model_name}-tokenizer.pkl','wb') as f:
        pickle.dump(tokenizer, f)

    for excerpt in df.excerpt:
        input_data = tokenizer(excerpt, add_special_tokens=True, 
                               return_tensors='pt', padding='max_length',
                               max_length=MAX_LENGTH, truncation=True)
                              
        x_input_ids.append(input_data['input_ids'])
        x_masks.append(input_data['attention_mask'])

    x_input_ids = torch.cat(x_input_ids)
    x_masks = torch.cat(x_masks)
    y_target = torch.tensor(df.target.values, dtype=torch.float32)

    return x_input_ids, x_masks, y_target


# In[ ]:


# define the dataset class 

class CommonReadabilityDataset(Dataset):

    def __init__(self, input_ids, masks, targets):
        self.input_ids = input_ids
        self.masks = masks
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        input_id = self.input_ids[idx]
        mask = self.masks[idx]
        target = self.targets[idx]

        return (input_id, mask, target)


# In[ ]:


# Train the model (includes head modification)

def get_model(model_name:str, num_targets, task:Task=Task.TOKEN_CLASSIFICATION):

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # define the model 
    if 'token' in task.value:
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    if 'sequence' in task.value:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, num_targets)

    return model, tokenizer, config


# In[ ]:


# define the attention block
class AttentionBlock(nn.Module):

    def __init__(self, in_features, middle_features, out_features):
        super(AttentionBlock, self).__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)
        
    def forward(self, features):
        att = torch.tanh(self.W(features)) #(bs, seq, middle_features)
        scores = self.V(att) #(bs, seq, 1)

        # convert the scores into probabilities
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = attention_weights * features #(bs, seq, in_features)
        context_vector = torch.sum(context_vector, dim=1) #(bs , in_features)

        return context_vector


# In[ ]:


class MultHeadAttentionBlock(nn.Module):

    def __init__(self,in_features, num_heads=8, dropout = 0.1):
        super(MultHeadAttentionBlock,self).__init__()

        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.norm = nn.LayerNorm(in_features)
        
        self.q_proj = nn.Linear(in_features, in_features)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)

        self.proj = nn.Linear(in_features, in_features)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        residual = x
        x = self.norm(x)
        
        batch_size, seq_len, _ = x.shape

        # reshaping 
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(2,1)
        k = k.transpose(2,1)
        v = v.transpose(2,1)
        
        scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.head_dim)

        if mask is None:
            scores = scores.masked_fill_(mask == 0, float('-inf'))
        
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        # (b_s, num_heads,seq_len, head_dim)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(batch_size,seq_len,self.num_heads * self.head_dim)

        return self.out_drop(self.proj(out)) + residual


# In[ ]:


# CounterFactual Reasoning Prediction
class CRPTokenModel(nn.Module):

    def __init__(self, model_name: str, num_targets: int):
        super(CRPTokenModel, self).__init__()

        model, tokenizer, config = get_model(model_name= model_name, 
                                             num_targets = num_targets, 
                                             task = Task.TOKEN_CLASSIFICATION)

        self.model = model
        
        in_features = model.classifier.in_features

        self.model.classifier = nn.Identity()
        self.att_blk = MultHeadAttentionBlock(in_features)
        self.fc = nn.Linear(in_features, num_targets)

    def forward(self, input_ids, attention_mask):

        
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        context_features = self.att_blk(output['logits'], mask=attention_mask)
        att_blk = torch.sum(context_features, dim=1)
        return self.fc(att_blk)


# In[ ]:


model, tokenizer, config = get_model("roberta-base",num_targets=1,task=Task.TOKEN_CLASSIFICATION)


# In[ ]:


x_input_ids, x_masks, y_target = generate_data()
dataset = CommonReadabilityDataset(input_ids = x_input_ids, masks=x_masks, targets = y_target)


# In[ ]:


get_ipython().system('mkdir models')


# In[ ]:


def one_step(input_id, attention_mask,y, model, loss_fn, optimizer, scheduler=None, device='cuda'):

    input_id = input_id.to(device=device)
    attention_mask = attention_mask.to(device=device)
    y = y.to(device=device)
    model.to(device=device)
    optimizer.zero_grad()
    
    o = model(input_ids = input_id, attention_mask = attention_mask)
    loss = loss_fn(o,y)
    loss.backward()

    with torch.no_grad():
        l = loss.item()
        r2 = r2_score(y.cpu().numpy(), o.cpu().numpy())
        rmse = torch.sqrt(torch.mean(torch.square(o - y))).item()
        mad = torch.mean(torch.abs(o - y)).item()
    
    optimizer.step()
    return l, r2, rmse, mad


# In[ ]:


@torch.no_grad()
def evaluate(model, valid_dl, criterion, device):
    model.eval()
    progress_bar = tqdm(valid_dl, leave=False)
    predicted_output, actual_target = [],[]
    for (input_id, mask, target) in progress_bar:

        input_id = input_id.to(device=device)
        mask = mask.to(device=device)
        target = target.to(device=device)
        model.to(device=device)
        
        o = model(input_ids = input_id, attention_mask = mask)
        predicted_output.append(o)
        actual_target.append(target)

    predicted_output = torch.cat(predicted_output)
    actual_target = torch.cat(actual_target)

    rmse = torch.sqrt(torch.mean(torch.square(predicted_output - actual_target))).item()
    mean_absolute_deviation = torch.mean(predicted_output - actual_target).item()
    val_loss = criterion(predicted_output, actual_target).item()
    r2 = r2_score(actual_target.cpu().numpy(), predicted_output.cpu().numpy())

    return val_loss, r2_score, rmse, mean_absolute_deviation


# In[ ]:


def one_epoch(model, criterion, optimizer, scheduler, train_dl, valid_dl, device='cuda'):

    model.train()
    loss,r2,root_mean_square, mean_absolute_deviation, iteration = 0,0,0,0,0
    progress_bar = tqdm(train_dl, leave=False)
    
    
    for (input_id, mask, target) in progress_bar:
        _loss,_r2,_root_mean_square, _mean_absolute_deviation = one_step(input_id,mask,target,model, criterion,optimizer, device=device)
        
        loss += _loss
        r2 += _r2
        root_mean_square += _root_mean_square
        mean_absolute_deviation += _mean_absolute_deviation 
        
        iteration += 1

        progress_bar.set_postfix(
            loss='{:.3f}'.format(loss/iteration),
            r2='{:.3f}'.format(r2/iteration),
            root_mean_square='{:.3f}'.format(root_mean_square/iteration),
            mean_absolute_deviation='{:.3f}'.format(mean_absolute_deviation/iteration)
        )

        scheduler.step()

    loss /= iteration
    r2 /= iteration
    root_mean_square /= iteration
    mean_absolute_deviation /= iteration
    val_loss, r2_val_score, val_rmse, val_mean_absolute_deviation = evaluate(model, valid_dl, criterion, device)

    return (loss, val_loss),(r2, r2_val_score),(mean_absolute_deviation, val_mean_absolute_deviation),(root_mean_square,val_rmse)


# In[ ]:


class SaveModelConfiguration:

    def __init__(self,name, metric, mode='min', top_k=2):
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.top_metrics = []
        self.top_k = top_k
        self.name = name
        self.model_root_path = Path(MODEL_ROOT)
        self.top_models = []
        self.logs = []

    def log(self, model, metrics):
        metric = metrics[self.metric]
        rank = self.get_rank(metric)

        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop()
        self.top_metrics.insert(rank + 1, metric)

        self.logs.append(metrics)
        self.save(model, metric, metrics['epoch'], rank)
    
    def save(self, model, metric, epoch, rank):
        timestamp = time.strftime('%Y%m%d%H%M%S')
        
        # create filename format: name_epoch_01_loss_0.45_timestamp.pth
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(
            self.name,
            epoch,
            self.metric,
            metric,
            timestamp
        ) 

        # remove unwanted characters 
        name = re.sub(r'[^\w_\-\.]','',name) + '.pth'
        self.top_models.insert(rank+1, name)

        if len(self.top_models) > self.top_k:
            last_model_name = self.top_models[-1]
            old_model = self.model_root_path.joinpath(last_model_name)

            if old_model:
                old_model.unlink()
        
        model_path = self.model_root_path.joinpath(name)
        torch.save(model.state_dict(), model_path)

        
    def get_rank(self, val):

        if self.mode == 'min':
            for index, metric_val in enumerate(self.top_metrics):
                if val <= metric_val:
                    return index - 1

        if self.mode == 'max':
            self.top_metrics = sorted(self.top_metrics, reverse=True)
            for index, metric_val in enumerate(self.top_metrics):
                if val >= metric_val:
                    return index - 1
                    
        return len(self.top_metrics)  - 1    

    def write_to_json(self):
        name = "{}_logs".format(self.name)
        name = re.sub(r"[^\w_\-\.]", "", name) + ".json"
        path = self.model_root_path.joinpath(name)
        
        with path.open('w') as f:
            json.dump(self.logs, indent=2)


# In[ ]:


def one_fold(model_name, train_set, valid_set, num_targets=1, save=True, device='cuda'):
    # collect information from epoch
    # (loss, val_loss),(r2, r2_val_score),(mean_absolute_deviation, val_mad),(root_mean_square,val_rmse)
    model_saver = SaveModelConfiguration(name='roberta-base', metric='loss')

    model = CRPTokenModel(model_name, num_targets = num_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=n_epochs)

    train_input_ids, train_masks, train_target = x_input_ids[train_set], x_masks[train_set], y_target[train_set]
    train_dataset = CommonReadabilityDataset(input_ids = train_input_ids, masks=train_masks, targets = train_target)

    train_dl = DataLoader(train_dataset, pin_memory=True, shuffle=True, batch_size=per_device_train_batch_size, num_workers=2)

    valid_input_ids, valid_masks, valid_target = x_input_ids[train_set], x_masks[train_set], y_target[train_set]
    valid_dataset = CommonReadabilityDataset(input_ids = train_input_ids, masks=train_masks, targets = train_target)

    valid_dl = DataLoader(valid_dataset, pin_memory=True, shuffle=False, batch_size=per_device_eval_batch_size, num_workers=2)

    progress_bar = tqdm(range(n_epochs), leave=False)
    criterion = nn.MSELoss()
    
    for epoch in progress_bar:
        progress_bar.set_description(f'Epoch {epoch:02d}')

        model.train()
        (loss, val_loss),(r2, r2_val_score),(mean_absolute_deviation, val_mean_absolute_deviation),(root_mean_square,val_rmse) = one_epoch(
            model, criterion, optimizer, scheduler, train_dl, valid_dl)
        
        progress_bar.set_postfix(
            loss='({:.3f},{:.3f})'.format(loss, val_loss),
            rmse='({:.3f},{:.3f})'.format(root_mean_square, val_rmse),
            mad='({:.3f},{:.3f})'.format(mean_absolute_deviation, val_mean_absolute_deviation)
        )

        if save:
            progress_metrics = {
                'epoch':epoch,
                'rmse': -root_mean_square,'mad': mean_absolute_deviation, 'r2':r2, 'loss':loss,
                'val_rmse':val_rmse, 'val_mad':val_mean_absolute_deviation, 'r2_val_score':r2_val_score,'val_loss':val_loss
            }
            model_saver.log(model=model, metrics=progress_metrics)


# In[ ]:


def train(model_name, num_targets=1, seed=42):
    gc.collect()
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fold_bar = tqdm(df.reset_index(drop=True).reset_index().groupby('kfold').index.apply(list).items(),total=df.kfold.max() + 1)
    
    # seed everything 
    seed = seed_everything(seed=132)
    
    for fold, valid_set in fold_bar:
        fold_bar.set_description(f'FOLD {fold} SEED {seed}')
    
        train_set = np.setdiff1d(df.index, valid_set)
        one_fold(model_name, train_set, valid_set, num_targets=num_targets, save=True, device=device)
    
        gc.collect()
        torch.cuda.empty_cache()


# In[ ]:


train('roberta-base', num_targets=1, seed=42)


# In[ ]:




