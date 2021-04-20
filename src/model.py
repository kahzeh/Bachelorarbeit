import json
import os
import re
import sys

import ipdb
import jsonlines
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.metrics import (Accuracy, MetricCollection, Precision,
                                       Recall)
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dictionaries import charwise_dict, eco_dict, names_dict, wordpiece_dict3, result_dict


class DataModule(pl.LightningDataModule):

    def __init__(self):

        super().__init__()

        with open('./test.jsonl') as f:
            lines = f.read().splitlines()
        
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']

        self.df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))

        self.df_final['tokenized_pgn'] = self.df_final['tokenized_pgn'].apply(lambda x: np.array(x))
        self.df_final['tokenized_pgn'] = self.df_final['tokenized_pgn']
        self.x = torch.Tensor(list(self.df_final['tokenized_pgn'].values))
        self.x = self.x.int()

        self.df_final['ECO']= self.df_final['ECO'].map(eco_dict)
        #self.df_final['White']= self.df_final['White'].map(names_dict)
        self.df_final['Result'] = self.df_final['Result'].map(result_dict)
        self.y_task1 = torch.tensor(list(self.df_final['ECO'].values))
        self.y_task2 = torch.tensor(list(self.df_final['Result'].values))
        self.y_task1 = self.y_task1.long()
        self.y_task2 = self.y_task2.long()
        #self.y_task2 = torch.tensor(list(self.df_final['White'].values))
        #self.y_task2 = self.y_task2.float()
        #self.y_task3 = torch.tensor(list(df_final['Black'].values))
        #self.y_task4 = torch.tensor(list(df_final['WhiteElo'].values))

    def __getitem__(self, index):
        return self.x[index], self.y_task1[index], self.y_task2[index]
    
    def __len__(self):
        return len(list(self.df_final['tokenized_pgn'].values))


    


class Transformer(pl.LightningModule):

    def __init__(
        self, 
        *,
        d_model,
        hidden_dim=2048,
        num_heads,
        dropout,
        num_layers,
        activation='relu',
        kernel_size,
        num_emb,
        emb_dim
        ):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation
                
            )
            self.dataset = DataModule()

            train_size = 185452
            test_size = 39740
            val_size = 39740

            self.train_set, self.test_set, self.val_set = random_split(self.dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))
            self.train_loader = DataLoader(self.train_set, batch_size=4, num_workers=4)
            self.val_loader = DataLoader(self.val_set, batch_size=4, num_workers=4)
            self.emb = nn.Embedding(num_emb, emb_dim, padding_idx=0)
            self.loss_eco = nn.CrossEntropyLoss()
            self.loss_result = nn.CrossEntropyLoss()
            self.task_list = ['eco', 'result']
            task_labels= {'eco': eco_dict, 'result': result_dict}

            self.train_metrics = nn.ModuleDict()
            self.val_metrics = nn.ModuleDict()
            for task in self.task_list:
                labels = task_labels[task]

                xmetrics = pl.metrics.MetricCollection([
                    Accuracy(),
                    Precision(num_classes=len(labels), average='macro'),
                    Recall(num_classes=len(labels), average='macro')
                ])

                self.train_metrics[task] = xmetrics.clone()
                self.val_metrics[task] = xmetrics.clone()

            
            

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

            self.classifier_eco = nn.Sequential(
                nn.Linear(emb_dim, len(eco_dict)) #emb_size -> label_size
            )

            # self.classifier_names = nn.Sequential(
            #     nn.Linear(emb_dim, len(names_dict))
            # )

            self.classifier_result = nn.Sequential(
                nn.Linear(emb_dim, len(result_dict))
            )

            # self.classifier_elo = nn.Sequential(
            #     #TODO
            # )

            # self.classifier_year = nn.Sequential(
            #     #TODO
            # )

            self.smax = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        x, y_task1, y_task2 = batch
        # z \in B x H_o
        x = self.emb(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        pred_task1 = self.classifier_eco(x)
        # pred_task1 B x N_c
        pred_task2 = self.classifier_result(x)
        # pred_task2
        #pred_task3 = self.forward(x)
        loss_task1 = self.loss_eco(pred_task1, y_task1)
        loss_task2 = self.loss_result(pred_task2, y_task2)
        #loss_task3 = F.mse_loss(pred_task3, y_task3)
        loss = sum([loss_task1, loss_task2])
    
        self.train_metrics['eco'](self.smax(pred_task1), y_task1)
        self.train_metrics['result'](self.smax(pred_task2), y_task2)
        self.log_dict(self.train_metrics['eco'], on_step=True, on_epoch=False)
        self.log_dict(self.train_metrics['result'], on_step=True, on_epoch=False)
        return loss


    def training_step(self, train_batch, batch_idx):

        loss = self.forward(train_batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, val_batch, batch_idx):
        loss = self.forward(val_batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.classifier_eco.parameters())




neptune_logger = NeptuneLogger(
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5Mjc4YmIyYi0yMjI4LTRlNDQtOTEzZS01NmM2NGU0OWRjODIifQ==',
    project_name='kahzeh/bachelorarbeit',
    #experiment_name='default',  # Optional,
    #params={'max_epochs': 10},  # Optional,
    #tags=['pytorch-lightning', 'mlp']  # Optional,
)

transformer = Transformer(d_model=512, num_heads=4, dropout=0.1, num_layers=8, kernel_size=512, num_emb=1155, emb_dim=512)

trainer = pl.Trainer(gpus=-1, overfit_batches=1, logger=neptune_logger, profiler='simple', max_epochs=20)
trainer.fit(transformer, transformer.train_loader, transformer.val_loader)
