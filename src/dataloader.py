import os
os.chdir("/home/jhkim/2022_tomcube_timepoint_refactoring/")

import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from config import * 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms 
import cv2


# set dataloader 
# %%
def get_xy(x_path, y_path):
    x = torch.Tensor(np.load(x_path)).unsqueeze(1)
    y = torch.LongTensor(np.load(y_path))
    return x, y

def get_dataset(x,y):
    return TensorDataset(x, y)

def get_loader(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path, batch_size):
    loader = {}
    for mode in ['train', 'valid', 'test']:
        if mode == 'train':
            x_path, y_path = x_train_path, y_train_path
            shuffle = True
        elif mode == 'valid':
            x_path, y_path = x_valid_path, y_valid_path
            shuffle = False
        elif mode == 'test':
            x_path, y_path = x_test_path, y_test_path
            shuffle = True

        x, y = get_xy(x_path, y_path)
        dataset = get_dataset(x, y)
        #loader[mode] = DataLoader(dataset= dataset, batch_size = batch_size, shuffle = shuffle, pin_memory = True, num_workers = 8)
        loader[mode] = DataLoader(dataset= dataset, batch_size = batch_size, shuffle = shuffle, pin_memory = True)
    
    #return loader['train'], loader['valid'], loader['test']
    return loader

def get_test_loader(x_test_path, y_test_path, batch_size):
    loader = {}
    for mode in ['test']:
        if mode == 'test':
            x_path, y_path = x_test_path, y_test_path
            shuffle = False

        x, y = get_xy(x_path, y_path)
        dataset = get_dataset(x, y)
        loader[mode] = DataLoader(dataset= dataset, batch_size = batch_size, shuffle = shuffle, pin_memory = True)
    
    return loader







from config import * 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms 
import cv2
class CustomDataset(Dataset):
    def __init__(self, images : np.array, 
                        label_list : np.array, 
                        train_mode=True, 
                        transforms=None): #필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        # self.img_path_list = img_path_list
        self.images = images
        self.label_list = label_list

    def __getitem__(self, index): #index번째 data를 return
        image = self.images[index]
        # Get image data
        # image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image
    
    def __len__(self): #길이 return
        return len(self.images)


train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
])

test_transform = transforms.Compose([
                    ])





def get_augmentation_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    train_df = torch.Tensor(np.load(x_train_path)).unsqueeze(1)
    train_dataset = CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)
   
    for _ in range(5):
        train_dataset += CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)

    valid_df = torch.Tensor(np.load(x_valid_path)).unsqueeze(1)
    valid_dataset = CustomDataset(valid_df ,torch.LongTensor(np.load(y_valid_path)), train_mode=True, transforms=test_transform)

    test_df = torch.Tensor(np.load(x_test_path)).unsqueeze(1)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=False, transforms=test_transform)

    return train_dataset, valid_dataset, test_dataset



def get_augmentation_loader(train, valid, test, batch_size):
    loader = {}

    loader['train'] = DataLoader(train, batch_size = batch_size, shuffle = True,  pin_memory = True)
    loader['valid'] = DataLoader(valid, batch_size = batch_size, shuffle = False,  pin_memory = True)
    loader['test'] = DataLoader(test, batch_size = batch_size,  shuffle = False, pin_memory = True)
    return loader


