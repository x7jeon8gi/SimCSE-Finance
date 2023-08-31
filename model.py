import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import pytorch_lightning as pl
import wandb
import math

def compute_avg_outs(outs: dict):
    mean_outs = {}
    for k in outs[0].keys():
        mean_outs.setdefault(k, 0.)
        for i in range(len(outs)):
            mean_outs[k] += outs[i][k]
        mean_outs[k] /= len(outs)
    return mean_outs


class Fin_SimCSE(pl.LightningModule):
    def __init__(self, config, length_of_dataset):
        # ! caution: need length_of_dataset for calculating total_steps
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config['transformer']["from_pretrained"])
        self.tokenizer = AutoTokenizer.from_pretrained(config['transformer']["from_pretrained"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(config['transformer']['hidden_size'], config['transformer']['hidden_size'])
        self.activation = nn.Tanh()
        self.lr = config['train']['learning_rate']
        
        # todo: for scheduler
        self.total_steps =  math.ceil(length_of_dataset / (config['train']['batch_size'] * config['train']['gpu_counts'] )) * config['train']['epochs']
        print(self.total_steps)
        self.warmup_steps = config['train']['warmup_steps']
        
        self.supervision:bool = config['train']['supervision']
        
        self._cuda = config['train']['device']

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        return [optimizer], [sch_config]
    
    def get_emb(self, **inputs):
        # inputs: input_ids, attention_mask, token_type_ids
        # Use only BaseModelOutputWithPoolingAndCrossAttentions (batch_size, sequence_length, hidden_size)
        outputs = self.model(**inputs)[0] 
        
        # take only the first token ([CLS]) of the last layer
        emb = outputs[:, 0, :]
        
        # MLP
        emb= self.linear(emb)
        emb = self.activation(emb)
        
        # Automatically dropout
        return emb
    
    def forward(self, **inputs):
        
        if self.supervision == False:
            #* Unsupervised SimCSE
            emb1 = self.get_emb(**inputs) # (batch_size, hidden_size)
            emb2 = self.get_emb(**inputs) # (batch_size, hidden_size)
            
            emb1 = emb1.unsqueeze(1) # (batch_size, 1, hidden_size)
            emb2 = emb2.unsqueeze(0) # (1, batch_size, hidden_size)
            
            sim_matrix = F.cosine_similarity(emb1, emb2, dim=-1) # (batch_size, batch_size)
            sim_matrix = sim_matrix / self.config['train']['temperature']
            
            labels = torch.arange(sim_matrix.size(0)).long().to(self._cuda)
            loss = F.cross_entropy(sim_matrix, labels)
            
            return loss
        
        else:
            #* Supervised SimCSE
            anchor , positive, negative = inputs['anchor'], inputs['positive'], inputs['negative']
            anchor_emb = self.get_emb(**anchor)
            positive_emb = self.get_emb(**positive)
            negative_emb = self.get_emb(**negative)
            
            anchor_emb = anchor_emb.unsqueeze(1)
            positive_emb = positive_emb.unsqueeze(0)
            negative_emb = negative_emb.unsqueeze(0)
            
            positive_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1) / self.config['train']['temperature'] # (batch_size, batch_size)
            negative_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=-1) / self.config['train']['temperature'] # (batch_size, batch_size)
            cos_sim = torch.cat([positive_sim, negative_sim], dim=1) # (batch_size, 2*batch_size)
            
            # in batch negative & supervised
            labels = torch.arange(cos_sim.size(0)).long().to(self._cuda) # (batch_size)
            loss = F.cross_entropy(cos_sim, labels)
            
            return loss
            
    
    # ! Caution: NAMES
    def training_step(self, batch, batch_idx):
        train_step_loss = self(**batch)
        self.log('train_step_loss', train_step_loss, sync_dist=True)
        
        return {"loss" :train_step_loss}
    
    def validation_step(self, batch, batch_idx):
        val_step_loss = self(**batch)
        self.log('val_step_loss', val_step_loss, sync_dist=True)
        
        return {"loss": val_step_loss}
    
    def test_step(self, batch, batch_idx):
        test_step_loss = self(**batch)
        self.log('test_step_loss', test_step_loss)
        
        return {"loss": test_step_loss}
    
    def training_epoch_end(self, outs):
        # print(outs)
        # self.check = outs
        # mean_loss = torch.stack([x['loss'] for x in outs]).mean()
        mean_loss = compute_avg_outs(outs)
        self.log('train_epoch_loss', mean_loss['loss'], sync_dist=True)

    def validation_epoch_end(self, outs):
        # mean_loss = torch.stack([x['loss'] for x in outs]).mean()
        mean_loss = compute_avg_outs(outs)
        self.log('val_epoch_loss', mean_loss['loss'], sync_dist=True)
        
    def test_epoch_end(self, outs):
        # mean_loss = torch.stack([x['loss'] for x in outs]).mean()
        mean_loss = compute_avg_outs(outs)
        self.log('test_epoch_loss', mean_loss['loss'])
    