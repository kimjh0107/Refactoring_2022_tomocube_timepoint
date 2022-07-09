import numpy as np 
from config import * 
from src.dataloader import CustomDataset, get_augmentation_loader, get_test_augmentation_loader
import torchvision.transforms as transforms 
from src.device import get_device
import torch 
from src.accuracy import *
from src.seed import seed_everything
from src.loss import get_loss
from src.model import * 

seed_everything(42)

def get_path(remove_patient:int, cell_type:str, target:str, type:str):
    return Path(f'npy/blind_test_{remove_patient}/{cell_type}_{target}_{type}.npy')

def get_pathes(remove_patient:int, celltype:str):
    cd8_x_train_path = get_path(remove_patient, celltype, 'x', 'train')
    cd8_y_train_path = get_path(remove_patient, celltype, 'y', 'train')
    cd8_x_valid_path = get_path(remove_patient, celltype, 'x', 'valid')
    cd8_y_valid_path = get_path(remove_patient, celltype, 'y', 'valid')
    cd8_x_test_path = get_path(remove_patient, celltype, 'x', 'test')
    cd8_y_test_path = get_path(remove_patient, celltype, 'y', 'test')
    return cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path

def get_test_path(remove_patient:int, cell_type:str, target:str, type:str):
    return Path(f'npy/blind_test_{remove_patient}/test/test_{cell_type}_{target}_{type}.npy')

def get_test_pathes(remove_patient:int, celltype:str):
    test_cd8_x_test_path = get_test_path(remove_patient, celltype, 'x', 'test')
    test_cd8_y_test_path = get_test_path(remove_patient, celltype, 'y', 'test')
    return test_cd8_x_test_path, test_cd8_y_test_path



# get path 
cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path = get_pathes(2, 'cd8')
test_cd8_x_test_path, test_cd8_y_test_path = get_test_pathes(2, 'cd8')  
#img = np.load(cd8_x_train_path)
img = np.load(test_cd8_x_test_path)

normalize_mean = np.mean(img)
normalize_std = np.std(img)

train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.Normalize(mean = [normalize_mean], std= [normalize_std])
])

test_transform = transforms.Compose([
    #transforms.Normalize(mean = [normalize_mean], std = [normalize_std])
                ])


def get_augmentation_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    
    train_df = torch.Tensor(np.load(x_train_path)).unsqueeze(1)
    train_df = (train_df) / (10000)
    train_dataset = CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)

    valid_df = torch.Tensor(np.load(x_valid_path)).unsqueeze(1)
    valid_df = (valid_df) / (10000)
    valid_dataset = CustomDataset(valid_df ,torch.LongTensor(np.load(y_valid_path)), train_mode=True, transforms=test_transform)

    test_df = torch.Tensor(np.load(x_test_path)).unsqueeze(1)
    test_df = (test_df) / (10000)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)

    return train_dataset, valid_dataset, test_dataset

def get_test_augmentation_dataset(x_test_path, y_test_path):
    test_df = torch.Tensor(np.load(x_test_path).astype('float64')).unsqueeze(1)
    test_df = (test_df) / (10000)
    test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)
    return test_dataset

def main():
      device = get_device()
      criterion = get_loss()

    #   train_dataset, valid_dataset, test_dataset = get_augmentation_dataset(cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path)
    #   dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, BATCH_SIZE)


      blind_test_dataset = get_test_augmentation_dataset(test_cd8_x_test_path, test_cd8_y_test_path)
      dataloaders = get_test_augmentation_loader(blind_test_dataset, BATCH_SIZE)
      best_model = torch.load('model/blind_test_model/tt_blindtest_2_cd8_model_1.pt')

      # get each scores 
      loss_sum, acc, auroc, aupr, conf_matrix, labels , outputs = test_model(dataloaders, best_model, criterion,device)
      roc_scores, answers, preds = calculate_roc(dataloaders['test'], best_model, device)
    #  f1_score = f1_score(answers,preds,average='macro')

      # print scores 
      print({f"AUROC on test set: {auroc*100:.2f}"})
      print({f"AUPR on test set: {aupr*100:.2f}"})
      print({f"ACC on test set: {acc*100:.2f}"})
    #  print({f"F1 Score on test set: {f1_score*100:.2f}"})

      print({f"loss on test set: {loss_sum}"})
      print(conf_matrix)



if __name__ == '__main__':
    main()


