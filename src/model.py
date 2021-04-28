import argparse
import json
import os
import pprint
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

from dictionaries import (charwise_dict, eco_dict, elo_dict, names_dict,
                          names_list, result_dict, wordpiece_dict3, year_dict)


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
        self.x = (torch.tensor(list(self.df_final['tokenized_pgn'].values))).int()

        self.df_final['ECO']= self.df_final['ECO'].map(eco_dict)
        self.df_final['White']= self.df_final['White'].map(names_dict)
        self.df_final['Black']= self.df_final['Black'].map(names_dict)
        self.df_final['Result'] = self.df_final['Result'].map(result_dict)
        self.df_final['Decade'] = self.df_final['Decade'].map(year_dict)
        self.df_final['WhiteElo'] = self.df_final['WhiteElo'].map(elo_dict)
        self.df_final['BlackElo'] = self.df_final['BlackElo'].map(elo_dict)

        self.y_task1 = torch.tensor(list(self.df_final['ECO'].values))
        self.y_task2 = torch.tensor(list(self.df_final['Result'].values))
        self.y_task3 = torch.tensor(list(self.df_final['White'].values))
        self.y_task4 = torch.tensor(list(self.df_final['Black'].values))
        
        self.y_task5 = torch.tensor(list(self.df_final['WhiteElo'].values))
        self.y_task6 = torch.tensor(list(self.df_final['BlackElo'].values))

        self.y_task7 = torch.tensor(list(self.df_final['Decade'].values))

        self.y_task1 = self.y_task1.long()
        self.y_task2 = self.y_task2.long()
        self.y_task3 = self.y_task3.long()
        self.y_task4 = self.y_task4.long()
        self.y_task5 = self.y_task5.long()
        self.y_task6 = self.y_task6.long()
        self.y_task7 = self.y_task7.long()


    def __getitem__(self, index):
        return self.x[index], self.y_task1[index], self.y_task2[index], self.y_task3[index], self.y_task4[index], self.y_task5[index], self.y_task6[index], self.y_task7[index]
    
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
        emb_dim,
        learning_rate
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

            self.learning_rate = learning_rate

            train_size = 185452
            test_size = 39740
            val_size = 39740

            self.train_set, self.test_set, self.val_set = random_split(self.dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))
            self.train_loader = DataLoader(self.train_set, batch_size=4, num_workers=4, pin_memory=True)
            self.val_loader = DataLoader(self.val_set, batch_size=4, num_workers=4, pin_memory=True)
            self.emb = nn.Embedding(num_emb, emb_dim, padding_idx=0)

            self.task_list = ['eco', 'result', 'black', 'white', 'whiteelo', 'blackelo', 'decade']
            task_labels= {'eco': eco_dict, 'result': result_dict, 'black': names_dict, 'white': names_dict, 'whiteelo': elo_dict, 'blackelo': elo_dict, 'decade': year_dict}

            self.train_metrics_list = []
            self.train_loss_metrics_list = []
            self.classifiers = nn.ModuleDict()
            self.train_metrics = nn.ModuleDict()
            self.val_metrics = nn.ModuleDict()
            self.loss = nn.ModuleDict()
            for task in self.task_list:
                labels = task_labels[task]

                xmetrics = pl.metrics.MetricCollection([
                    Accuracy(),
                    Precision(num_classes=len(labels), average='macro'),
                    Recall(num_classes=len(labels), average='macro')
                ])

                self.train_metrics[task] = xmetrics.clone()
                self.val_metrics[task] = xmetrics.clone()

                self.classifiers[task] = nn.Linear(emb_dim, len(labels))

                self.loss[task] = nn.CrossEntropyLoss()

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

            self.smax = torch.nn.Softmax(dim=1)


    def forward(self, batch):
        x, y_task1, y_task2, y_task3, y_task4, y_task5, y_task6, y_task7 = batch

        x = self.emb(x)

        x = self.encoder(x)

        x = x.mean(dim=1)

        #pred_task1 = self.classifiers['eco'](x)
        #pred_task2 = self.classifier_result(x)
        #pred_task3 = self.classifier_white(x)
        #pred_task4 = self.classifier_black(x)
        #pred_task5 = self.classifier_whiteelo(x)
        #pred_task6 = self.classifier_blackelo(x)
        pred_task7 = self.classifiers['decade'](x)


        #loss_task1 = self.loss['eco'](pred_task1, y_task1)
        #loss_task2 = self.loss_result(pred_task2, y_task2)
        #loss_task3 = self.loss_white(pred_task3, y_task3)
        #loss_task4 = self.loss_black(pred_task4, y_task4)
        #loss_task5 = self.loss_whiteelo(pred_task5, y_task5)
        #loss_task6 = self.loss_blackelo(pred_task6, y_task6)
        loss_task7 = self.loss['decade'](pred_task7, y_task7)


        #loss = sum([loss_task1, loss_task2, loss_task3, loss_task4, loss_task5, loss_task6])
        loss = loss_task7
        metrics = []

        #metrics.append({'eco': self.train_metrics['eco'](self.smax(pred_task1), y_task1)})
        #metrics.append({'result': self.train_metrics['result'](self.smax(pred_task2), y_task2)})
        #metrics.append({'white': self.train_metrics['white'](self.smax(pred_task3), y_task3)})
        #metrics.append({'black': self.train_metrics['black'](self.smax(pred_task4), y_task4)})
        #metrics.append({'whiteelo': self.train_metrics['whiteelo'](self.smax(pred_task5), y_task5)})
        #metrics.append({'blackelo': self.train_metrics['blackelo'](self.smax(pred_task6), y_task6)})
        metrics.append({'decade': self.train_metrics['decade'](self.smax(pred_task7), y_task7)})
        
        self.train_metrics_list.append(metrics)
        #self.train_loss_metrics_list.append([{'loss_eco': loss_task1.item(), 'loss_result': loss_task2.item(), 'loss_white': loss_task3.item(), 'loss_black': loss_task4.item(), 'loss_whiteelo': loss_task5.item(), 'loss_blackelo': loss_task6.item()}])
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
        return Adam(
            #[
            #{'params': self.classifier_eco.parameters()}, 
            #{'params': self.classifier_white.parameters()}, 
            #{'params': self.classifier_black.parameters()}, 
            #{'params': self.classifier_result.parameters()},
            #{'params': self.classifier_whiteelo.parameters()},
            #{'params': self.classifier_blackelo.parameters()}
            #{'params': self.parameters()},
            #{'params': self.emb.parameters()},
            #],
            self.parameters(),
            lr=self.learning_rate
            )

#neptune_logger = NeptuneLogger(
    #api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5Mjc4YmIyYi0yMjI4LTRlNDQtOTEzZS01NmM2NGU0OWRjODIifQ==',
    #project_name='kahzeh/bachelorarbeit',
    #experiment_name='default',  # Optional,
    #params={'max_epochs': 10},  # Optional,
    #tags=['pytorch-lightning', 'mlp']  # Optional,
#)

def main(hparams):


    transformer = Transformer(
        d_model=hparams.d_model,
        num_heads=hparams.num_heads, 
        dropout=hparams.dropout, 
        num_layers=hparams.num_layers, 
        kernel_size=hparams.kernel_size, 
        num_emb=hparams.num_emb, 
        emb_dim=hparams.emb_dim,
        learning_rate=hparams.learning_rate
        )

    trainer = pl.Trainer(
        gpus=-1, 
        overfit_batches=10,
        fast_dev_run=hparams.fast_dev_run, 
        max_epochs=hparams.epoch_num, 
        weights_summary='full')


    trainer.fit(transformer, transformer.train_loader, transformer.val_loader)



    for item in transformer.train_metrics_list:
        for dicts in item:
            for sub_dicts in dicts.values():
                for key, value in sub_dicts.items():
                    sub_dicts[key] = value.item()

    # with open('metrics.json', 'w') as f:
    #     json.dump(transformer.train_metrics_list, f)

#with open('loss.json', 'w') as f:
    #json.dump(transformer.train_loss_metrics_list, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        default=0.01,
        type=float
    )

    parser.add_argument(
        "--num_heads",
        default=4,
        type=int
    )

    parser.add_argument(
        "--fast_dev_run",
        default=False,
        type=bool
    )

    parser.add_argument(
        "--epoch_num",
        default=10,
        type=int
    )
    
    parser.add_argument(
        "--dropout",
        default=0.2,
        type=float
    )

    parser.add_argument(
        "--kernel_size",
        default=512,
        type=int
    )
    
    parser.add_argument(
        "--num_emb",
        default=1155,
        type=int
    )

    parser.add_argument(
        "--emb_dim",
        default=512,
        type=int
    )

    parser.add_argument(
        "--num_layers",
        default=8,
        type=int
    )

    parser.add_argument(
        "--d_model",
        default=512,
        type=int
    )


    hparams = parser.parse_args()

    main(hparams)
