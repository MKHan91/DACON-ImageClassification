import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from baseline_model import BaseCNN
from sklearn.model_selection import KFold
from baseline_dataloader import load_dataset, MNISTDataLoader



parser = argparse.ArgumentParser('MNIST with noise classification')
parser.add_argument('--batch_size',     type=int,   help='train batch size',        default=16)
parser.add_argument('--num_workers',    type=int,   help='The number of CPU cores', default=5)
parser.add_argument('--learning_rate',  type=float, help='Initial learning rate',   default=1e-4)
parser.add_argument('--num_epochs',     type=int,   help='Train Epoch',             default=80)

args = parser.parse_args()




def main():
    num_gpus = torch.cuda.device_count()
    gpu_indices = list(range(num_gpus))
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_indices[0]))
    
    x_train, y_train, x_test = load_dataset()
    
    kf = KFold(n_splits=5, shuffle=True)
    global_step = 0
    for fold, (train_indices, val_indices) in enumerate(kf.split(x_train)):
        if fold == 1: break
        # ---------------------- 데이터 로더 정의 ----------------------------------
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
        # ---------------------------------------------------------------------------
        
        # ------------------------------- 뉴럴 네트워크 정의 -------------------------------
        baseline = BaseCNN().to(device)
        # ----------------------------------------------------------------------------------
        
        # -------------------------------- 학습을 위한 최적화 정의 ----------------------
        optimizer = torch.optim.Adam(baseline.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        # --------------------------------------------------------------------------------
        
        
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(args.num_epochs):
            baseline.train()
            
            avg_loss = 0
            pred_ep = torch.tensor([], device=device)
            label_ep = torch.tensor([], device=device)
            for step, (image_sample, label_sample) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                image_sample = torch.tensor(image_sample, dtype=torch.float32, device=device)
                label_sample = torch.tensor(label_sample, dtype=torch.long, device=device)
                
                with torch.cuda.amp.autocast():
                    prediction = baseline(image_sample)
                loss = criterion(prediction, label_sample)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                avg_loss += loss
                
                pred_ep = torch.cat([pred_ep, torch.argmax(prediction, axis=1)])
                label_ep = torch.cat([label_ep, label_sample])
                
                if step % 10 == 0:
                    print(f"[Epoch] [steps/total_step]: [{epoch+1:>4d}][{step+1:>4d}/{len(train_dataloader)}] | learning rate: {args.learning_rate:10.4f} | loss: {loss:10.4f}")

            
            acc_ep = (pred_ep == label_ep).float().mean().item()
            print(f"[Epoch] [steps/total_step]: [{epoch+1:>4d}][{step+1:>4d}/{len(train_dataloader)}] | learning rate: {args.learning_rate:10.4f} | loss: {loss:10.4f} | accuracy: {acc_ep:10.4f}")
            
            global_step += 1
            
            
            
            # Validation 코드 작성 
            # 모델 저장 코드 작성
            # learning rate schedular 코드 작성
            
if __name__ == "__main__":
    main()
    
