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
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchnlp.encoders import LabelEncoder

from dictionaries import (charwise_dict, eco_dict, elo_dict, names_dict,
                          names_list, result_dict, wordpiece_dict3, year_dict)

from torchsummary import summary


class PrepareData(pl.LightningDataModule):

    def __init__(self, hparams):

        super().__init__() 

        with open(hparams.data) as f:
            lines = f.read().splitlines()
        
        self.task_list = ['ECO', 'Result', 'Black', 'White', 'WhiteElo2', 'BlackElo2', 'Decade']
        self.task_labels= {'ECO': eco_dict, 'Result': result_dict, 'Black': names_dict, 'White': names_dict, 'WhiteElo2': elo_dict, 'BlackElo2': elo_dict, 'Decade': year_dict}


        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']

        self.df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
        self.df_final['tokenized_pgn'] = self.df_final['tokenized_pgn'].apply(lambda x: np.array(x))
        self.df_final['tokenized_pgn'] = self.df_final['tokenized_pgn']
        self.x = (torch.Tensor(list(self.df_final['tokenized_pgn'].values))).int()

        self.label_encoder = self.get_label_encoder()


        for task in self.task_list:
            self.df_final[task] = self.label_encoder[task].batch_encode(self.df_final[task])

        self.y_task1 = (torch.Tensor(list(self.df_final['ECO'].values))).long()
        self.y_task2 = (torch.Tensor(list(self.df_final['Result'].values))).long()
        self.y_task3 = (torch.Tensor(list(self.df_final['White'].values))).long()
        self.y_task4 = (torch.Tensor(list(self.df_final['Black'].values))).long()          
        self.y_task5 = (torch.Tensor(list(self.df_final['WhiteElo2'].values))).long()
        self.y_task6 = (torch.Tensor(list(self.df_final['BlackElo2'].values))).long()
        self.y_task7 = (torch.Tensor(list(self.df_final['Decade'].values))).long()
   
    def __getitem__(self, index):
        return self.x[index], self.y_task1[index], self.y_task2[index], self.y_task3[index], self.y_task4[index], self.y_task5[index], self.y_task6[index], self.y_task7[index]

    def __len__(self):
        return len(list(self.df_final['tokenized_pgn'].values))

    def get_label_encoder(self):
        label_encoder = {}
        for task in self.task_list:
            label_encoder[task] = LabelEncoder(
                [*self.task_labels[task]],
                reserved_labels=['unknown'],
                unknown_index=0
            )
        return label_encoder



class DataModule(pl.LightningDataModule):

    def __init__(self, hparams, dataset):
        super().__init__()
        self.dataset = dataset
        self.label_encoder = dataset.label_encoder
    
    def setup(self, stage = 'fit'):

        train_size = 185451
        test_size = 39740
        val_size = 39740

        self.train_set, self.test_set, self.val_set = random_split(self.dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):

        return DataLoader(
            self.train_set, 
            batch_size=hparams.batch_size, 
            num_workers=4, 
            pin_memory=True
            )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_set,
            batch_size=hparams.batch_size,
            num_workers=4,
            pin_memory=True
            )

    def test_dataloader(self):

        return DataLoader(self.test_set, 
        batch_size=hparams.batch_size, 
        num_workers=4, 
        pin_memory=True
        )

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
        learning_rate,
        label_encoder,
        hparams
        ):
            super().__init__()

            self.losses_weight = {
                'ECO' : hparams.eco_loss_wt,
                'Result' : hparams.result_loss_wt,
                'Black' : hparams.black_loss_wt,
                'White' : hparams.white_loss_wt,
                'WhiteElo2' : hparams.whiteelo_loss_wt,
                'BlackElo2' : hparams.blackelo_loss_wt,
                'Decade' : hparams.decade_loss_wt
            }

            self.learning_rate = learning_rate

            self.task_list = ['ECO', 'Result', 'Black', 'White', 'WhiteElo2', 'BlackElo2', 'Decade']

            self.train_metrics_list = []
            self.train_loss_metrics_list = []
            self.wt_loss_metrics_list = []
            self.classifiers = nn.ModuleDict()
            self.train_metrics = nn.ModuleDict()
            self.val_metrics = nn.ModuleDict()
            self.loss = nn.ModuleDict()
            for task in self.task_list:

                xmetrics = pl.metrics.MetricCollection([
                    Accuracy(),
                    Precision(num_classes=label_encoder[task].vocab_size, average='macro'),
                    Recall(num_classes=label_encoder[task].vocab_size, average='macro')
                ])

                self.train_metrics[task] = xmetrics.clone()
                self.val_metrics[task] = xmetrics.clone()

                self.classifiers[task] = nn.Linear(emb_dim, label_encoder[task].vocab_size)

                self.loss[task] = nn.CrossEntropyLoss()

            self.emb = nn.Embedding(num_emb, emb_dim, padding_idx=0)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation
                
            )

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

            self.smax = torch.nn.Softmax(dim=1)



    def forward(self, batch):
        x = batch
        embedding = self.emb(x)

        x = self.encoder(embedding)

        x = x.mean(dim=1)

        return x


    def training_step(self, train_batch, batch_idx):
        x, y_task1, y_task2, y_task3, y_task4, y_task5, y_task6, y_task7 = train_batch

        x = self.forward(x)

        pred_task1 = self.classifiers['ECO'](x)

        
        pred_task2 = self.classifiers['Result'](x)
        pred_task3 = self.classifiers['White'](x)
        pred_task4 = self.classifiers['Black'](x)
        pred_task5 = self.classifiers['WhiteElo2'](x)
        pred_task6 = self.classifiers['BlackElo2'](x)
        pred_task7 = self.classifiers['Decade'](x)
        
        loss_task1 = self.loss['ECO'](pred_task1, y_task1)
        wt_loss_task1 = loss_task1 * self.losses_weight['ECO']
        loss_task2 = self.loss['Result'](pred_task2, y_task2)
        wt_loss_task2 = loss_task2 * self.losses_weight['Result']
        loss_task3 = self.loss['White'](pred_task3, y_task3)
        wt_loss_task3 = loss_task3 * self.losses_weight['White']
        loss_task4 = self.loss['Black'](pred_task4, y_task4)
        wt_loss_task4 = loss_task4 * self.losses_weight['Black']
        loss_task5 = self.loss['WhiteElo2'](pred_task5, y_task5)
        wt_loss_task5 = loss_task5 * self.losses_weight['WhiteElo2']
        loss_task6 = self.loss['BlackElo2'](pred_task6, y_task6)
        wt_loss_task6 = loss_task6 * self.losses_weight['BlackElo2']
        loss_task7 = self.loss['Decade'](pred_task7, y_task7)
        wt_loss_task7 = loss_task7 * self.losses_weight['Decade']
        
        loss = sum([wt_loss_task1, wt_loss_task2, wt_loss_task3, wt_loss_task4, wt_loss_task5, wt_loss_task6, wt_loss_task7])

        #loss =loss_task1

        metrics = []

        metrics.append({'ECO': self.train_metrics['ECO'](self.smax(pred_task1), y_task1)})
        metrics.append({'Result': self.train_metrics['Result'](self.smax(pred_task2), y_task2)})
        metrics.append({'White': self.train_metrics['White'](self.smax(pred_task3), y_task3)})
        metrics.append({'Black': self.train_metrics['Black'](self.smax(pred_task4), y_task4)})
        metrics.append({'WhiteElo': self.train_metrics['WhiteElo2'](self.smax(pred_task5), y_task5)})
        metrics.append({'BlackElo': self.train_metrics['BlackElo2'](self.smax(pred_task6), y_task6)})
        metrics.append({'Decade': self.train_metrics['Decade'](self.smax(pred_task7), y_task7)})

        self.train_metrics_list.append(metrics)

        self.train_loss_metrics_list.append([{'loss_eco': loss_task1.item(), 'loss_result': loss_task2.item(), 'loss_white': loss_task3.item(), 'loss_black': loss_task4.item(), 'loss_whiteelo': loss_task5.item(), 'loss_blackelo': loss_task6.item(), 'loss_decade': loss_task7.item()}])
        self.wt_loss_metrics_list.append([{'loss_eco': wt_loss_task1.item(), 'loss_result': wt_loss_task2.item(), 'loss_white': wt_loss_task3.item(), 'loss_black': wt_loss_task4.item(), 'loss_whiteelo': wt_loss_task5.item(), 'loss_blackelo': wt_loss_task6.item(), 'loss_decade': wt_loss_task7.item()}])        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    # def validation_step(self, val_batch, batch_idx):
    #     #loss = self.forward(val_batch)
    #     #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     pass

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.learning_rate
            )

def main(hparams):

    prep_data = PrepareData(hparams)
    dataset = DataModule(hparams, prep_data)

    transformer = Transformer(
        d_model=hparams.d_model,
        num_heads=hparams.num_heads, 
        dropout=hparams.dropout, 
        num_layers=hparams.num_layers, 
        kernel_size=hparams.kernel_size, 
        num_emb=hparams.num_emb, 
        emb_dim=hparams.emb_dim,
        learning_rate=hparams.learning_rate,
        label_encoder=dataset.label_encoder,
        hparams=hparams
        )

    trainer = pl.Trainer(
        gpus=-1, 
        fast_dev_run=hparams.fast_dev_run, 
        max_epochs=hparams.epoch_num, 
        weights_summary='full',
        overfit_batches=10
        )
    
    if hparams.find_lr == True:

        lr_finder = trainer.tuner.lr_find(transformer, datamodule=dataset)

        lr_finder.results

        fig = lr_finder.plot(suggest=True)
        fig.savefig('./metrics/lr.png')

        new_lr = lr_finder.suggestion()

        transformer.hparams.learning_rate = new_lr


    trainer.fit(transformer, datamodule=dataset)


    for item in transformer.train_metrics_list:
        for dicts in item:
            for sub_dicts in dicts.values():
                 for key, value in sub_dicts.items():
                    sub_dicts[key] = value.item()

    with open('./metrics/metrics.json', 'w') as f:
       json.dump(transformer.train_metrics_list, f)

    with open('./metrics/loss.json', 'w') as f:
        json.dump(transformer.train_loss_metrics_list, f)

    with open('./metrics/wt_loss.json', 'w') as f:    
        json.dump(transformer.wt_loss_metrics_list, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        default=0.001,
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
        default=0.,
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
        default=2,
        type=int
    )

    parser.add_argument(
        "--d_model",
        default=512,
        type=int
    )

    parser.add_argument(
        "--find_lr",
        default=False,
        type=bool
    )

    parser.add_argument(
        "--data",
        default='./data.jsonl',
        type=str
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int
    )

    parser.add_argument(
        '--eco_loss_wt',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--result_loss_wt',
        default=2.0,
        type=float
    )

    parser.add_argument(
        '--white_loss_wt',
        default=1.0,
        type=float
    )    
    
    parser.add_argument(
        '--black_loss_wt',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--whiteelo_loss_wt',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--blackelo_loss_wt',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--decade_loss_wt',
        default=1.0,
        type=float
    )


    hparams = parser.parse_args()

    main(hparams)
