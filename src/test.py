import pandas as pd
import json
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dictionaries import eco_dict

class DataModule(pl.LightningDataModule):   

    def __init__(self):

        super().__init__()

        with open('./test.jsonl') as f:
            lines = f.read().splitlines()
        
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']

        self.df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
        x = (self.df_final['White'].value_counts()).to_dict()
        liste = []
        for z in list(x)[0:99]:
            print(z, x[z])
            liste.append(z)
        print(liste)
        with open('uniquenames.txt', 'w') as f:
            f.write(json.dumps(x))
        self.df_final['tokenized_pgn'] = self.df_final['tokenized_pgn'].apply(lambda x: np.array(x))
        self.x = torch.Tensor(list(self.df_final['tokenized_pgn'].values))

        self.df_final['ECO']= self.df_final['ECO'].map(eco_dict)
        self.y_task1 = torch.tensor(list(self.df_final['ECO'].values))
        #self.y_task2 = torch.tensor(list(df_final['White'].values))
        #self.y_task3 = torch.tensor(list(df_final['Black'].values))
        #self.y_task4 = torch.tensor(list(df_final['WhiteElo'].values))


    def __getitem__(self, index):
        return self.x[index], self.y_task1[index]
        
    def __len__(self):
        return len(list(self.df_final['tokenized_pgn'].values))

dataset = DataModule()


train_size = 185452
test_size = 39740
val_size = 39740

train_set, test_set, val_set = random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset = train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset = test_set, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset = val_set, batch_size=4, shuffle=True)
