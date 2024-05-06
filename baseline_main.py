import argparse
import torch
import torch.nn as nn

from torchmetrics import metric
from torch.utils.data import DataLoader
from baseline_model import BaseCNN
from sklearn.model_selection import KFold
from baseline_dataloader import load_dataset, MNISTDataLoader



parser = argparse.ArgumentParser('MNIST with noise classification')
parser.add_argument('--batch_size',     type=int,   help='train batch size', default=16)
parser.add_argument('--num_workers',    type=int,   help='The number of CPU cores', default=5)
parser.add_argument('--learning_rate',  type=float, help='Initial learning rate', default=1e-4)
parser.add_argument('--num_epochs',     type=int, help='Train Epoch', default=80)

args = parser.parse_args()




def main():
    num_gpus = torch.cuda.device_count()
    gpu_indices = list(range(num_gpus))
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_indices[0]))
    
    x_train, y_train, x_test = load_dataset()
    
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_indices, val_indices) in enumerate(kf.split(x_train)):
        if fold == 1: break
        train_images, train_labels = x_train[train_indices], y_train[train_indices]
        val_images, val_labels = x_train[val_indices], y_train[val_indices]

        train_dataset = MNISTDataLoader(image=train_images, labels=train_labels, mode='train')
        val_dataset = MNISTDataLoader(image=val_images, labels=val_labels, mode='train')
        
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                                      num_workers=args.num_workers, 
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, 
                                    pin_memory=True)
        
        baseline = BaseCNN().to(device)
        optimizer = torch.optim.Adam(baseline.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(args.num_epochs):
            baseline.train()
            
            for step, (image_sample, label_sample) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                image_sample = torch.tensor(image_sample, dtype=torch.float32, device=device)
                label_sample = torch.tensor(label_sample, dtype=torch.float32, device=device)
                
                prediction = baseline(image_sample)
                loss = criterion(prediction, label_sample)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
            
            
            
if __name__ == "__main__":
    main()