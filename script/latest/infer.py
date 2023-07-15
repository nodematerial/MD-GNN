import os
import yaml
import random

import numpy as np
import pandas as pd

from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torch_geometric.nn import SAGEConv

os.environ['WANDB_SILENT'] = 'true'

device = 'cuda'


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20000, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.squeeze(x, 1)
        prediction = self.fc2(x)
        return prediction


class MDGraphDataset(torchdata.Dataset):
    def __init__(self, dfs, cnadicts, datadirs):
        # 0sec の値が異常値なので取り除く
        self.dfs = dfs
        self.cnadicts = cnadicts
        self.datadirs = datadirs
        self.datalen = 0
        self.datalens = []
        for df in self.dfs:
            self.datalens.append(self.datalen)
            self.datalen += len(df)
        self.datalens.append(np.inf)

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx: int):
        new_idx, df, cnadict, datadir = self.fetch_object_by_idx(idx)
        step = df.iloc[new_idx, 0] # Step
        cna = cnadict[step]
        x, edge_index = load_data(step, datadir)
        target = df.iloc[new_idx, 3] # PotEng
        x = np.concatenate([x, cna[:, np.newaxis]], 1)
        x = to_tensor(x).float()
        edge_index = to_tensor(edge_index)
        target = to_tensor(target).float().float()

        return x, edge_index, target

    # idx に対して適切なオブジェクトを選択する
    def fetch_object_by_idx(self, idx: int):
        for i, length in enumerate(self.datalens):
            if idx < self.datalens[i + 1]:
                new_idx = idx - length
                return new_idx, self.dfs[i], self.cnadicts[i], self.datadirs[i]

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_cna(cna_path):
    cna_dict = dict()
    for file in cna_path.iterdir():
        with open(file, 'r') as f:
            li = np.array([int(s.strip()) for s in f.readlines()])
            li = np.where(li >= 1, 1.0, 0.0)
        a = int(file.stem)
        cna_dict[a] = li
    return cna_dict


def infer(dataset, model):
    predictions = []
    with torch.no_grad():
        for x, edge, label in dataset:
            prediction = model(x, edge).cpu().numpy()[0]
            predictions.append(prediction)
    return predictions


def load_data(step, datadir):
    feature_path = datadir / 'x' / f'{step}.npy'
    edges_path = datadir / 'edges' / f'{step}.npy'
    x = np.load(feature_path)
    label_index = np.load(edges_path)
    edge_index = np.concatenate([label_index, label_index[[1, 0], :]], 1)
    return x, edge_index


def to_tensor(input_array):
    return torch.tensor(input_array).to(device)


def create_csv(dfs, prediction):
    df = pd.concat(dfs)
    df['Prediction'] = prediction
    df = df[['PotEng', 'Prediction']]
    df.to_csv('prediction.csv', index=False)


def load_dataset(type, CFG):
    dfs = []
    cnadicts = []
    cutoffdirs = []
    for dir in CFG[f'{type}_dirs']:
        DATADIR = Path('../../dataset') / str(dir)
        CUTOFFDIR = DATADIR / str(CFG['cutoff'])
        cna_path = DATADIR / 'cna'
        csv_path = DATADIR / 'thermo.csv'
        # 0sec の値が異常値なので取り除く
        df = pd.read_csv(csv_path)
        ground_energy = df.iloc[0, 3]  # 最初のステップの位置エネルギー
        df = df.iloc[1:, :]
        df['PotEng'] = df['PotEng'] - ground_energy
        cnadict = load_cna(cna_path)
        dfs.append(df)
        cnadicts.append(cnadict)
        cutoffdirs.append(CUTOFFDIR)
    dataset = MDGraphDataset(dfs, cnadicts, cutoffdirs)
    return dfs, dataset


def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    seed_everything(CFG['seed'])
    dfs, dataset = load_dataset('all', CFG)
    in_channels = dataset[0][0].size()[1]
    hidden_channels = CFG['hidden_channels']
    model = GCNModel(in_channels, hidden_channels).to(device)
    weight = torch.load('best.pth')
    model.load_state_dict(weight)
    prediction = infer(dataset, model)
    create_csv(dfs, prediction)


if __name__ == '__main__':
    main()
