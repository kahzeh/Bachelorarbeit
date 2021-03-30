import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class Transformer(pl.LightningModule):

    def __init__(
        self, 
        *,
        input_dim,
        hidden_dim,
        num_heads,
        dropout,
        num_layers,
        activation='relu'
        ):
        super().__init__()
        trm_layer = nn.TransformerEncoderLayer(
            input_dim,
            num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation
        )
        self.trm = nn.TransformerEncoder(trm_layer, num_layers)


    def forward(self, x):
        embedding = self.trm(x)
        # embedding B x T x H
        pooled_output = self.pooling_layer(embedding)
         #take the first token, mean pooling
         #pooled_output B x H_o
        return(pooled_output)


    def training_step(self, train_batch, batch_idx):
        x, y1, y2, y3 = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        # z \in B x H_o
        pred_task1 = self.prediction_head_task1(pooled_output)
        # pred_task1 B x N_c
        pred_task2 = self.prediction_head_task2(pooled_output)
        # pred_task2
        loss_task1 = F.mse_loss(pred_task1, y_task1)
        loss_task2 = F.mse_loss(pred_task2, y_task2)
        loss_task3 = F.mse_loss(pred_task3, y_task3)
        loss = loss_task1 + loss_task2 + loss_task3
        self.log('train_loss', loss)
        return(loss)
