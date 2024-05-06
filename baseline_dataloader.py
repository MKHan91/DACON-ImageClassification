import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms


def load_dataset():
    train = pd.read_csv('./DATASET/train.csv')
    test = pd.read_csv('./DATASET/test.csv')

    x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
    x_train = x_train.reshape(-1, 28, 28, 1)
    y = train['digit']
    # one-hot encoding: 레이블을 모델이 학습할 수 있게 형태를 변환해주는 과정.
    y_train = np.zeros((len(y), len(y.unique())))
    for i, digit in enumerate(y):
        y_train[i, digit] = 1
    
    x_test = test.drop(['id', 'letter'], axis=1).values
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # 0~1사이로 Normalization
    x_test = x_test / 255
    # 0 ~ 1사이로 정규화 해야 함.
    x_train = x_train/255

    return x_train, y_train, x_test



class MNISTDataLoader(Dataset):
    def __init__(self, image, labels, mode='train'):
        self.image = image
        self.labels = labels
        self.mode = mode
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return self.image.shape[0]


    def __getitem__(self, idx):
        img = self.image[idx]
        if self.mode == 'train':
            img = self.transforms(img)

        label = self.labels[idx]
        return img, label


    def get_labels(self):
        return self.labels
    
    
