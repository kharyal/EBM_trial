'''
creates dataset for training and testing
and calls the model trainer
'''

import os
import numpy as np
import torch
from trainer.concept_ebm_trainer import ConceptEBMTrainer
import pickle # to read the data file
import argparse
from torch.utils.data import DataLoader, Dataset

argparse = argparse.ArgumentParser(description="Concept EBM Trainer")
argparse.add_argument('--data_directory', type=str, default='data', help='Directory to save the data')
argparse.add_argument('--data_file', type=str, default='random_points_data.pkl', help='File name to save the data')
argparse.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
argparse.add_argument('--num_epochs', type=int, default=400, help='Number of epochs for training')
argparse.add_argument('--buffer_size', type=int, default=500000, help='Buffer size for training')
argparse.add_argument('--num_langevin_steps', type=int, default=30, help='Number of Langevin steps for training')
argparse.add_argument('--step_lr', type=float, default=0.01, help='Step size for Langevin steps')
argparse.add_argument('--kl_weight', type=float, default=0.1, help='Weight for KL divergence loss')
argparse.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')


args = argparse.parse_args()
datafile = os.path.join(args.data_directory, args.data_file)
if not os.path.exists(datafile):
    raise FileNotFoundError(f"Data file {datafile} not found. Please generate the data first.")

with open(datafile, 'rb') as f:
    data = pickle.load(f)

class RandomPointsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points, observer, labels = self.data[idx]
        points = torch.tensor(points, dtype=torch.float32)
        observer = torch.tensor(observer, dtype=torch.float32)
        # points[..., :2] = points[..., :2] - observer
        labels = torch.tensor(labels, dtype=torch.float32)
        objects = labels[0].long()
        subjects = labels[1].long()
        obj_onehot = torch.nn.functional.one_hot(objects, num_classes=6)
        sub_onehot = torch.nn.functional.one_hot(subjects, num_classes=4)
        labels = torch.cat((obj_onehot, sub_onehot), dim=-1)

        return points, observer, labels
    
# train and test split
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
train_dataset = RandomPointsDataset(train_data)
test_dataset = RandomPointsDataset(test_data)
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
    'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
}

trainer = ConceptEBMTrainer(dataloaders, args)

trainer.train()
