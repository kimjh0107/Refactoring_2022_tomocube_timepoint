import numpy as np 
from config import * 
from src.dataloader import *
from src.device import get_device
import torch 
from src.accuracy import *
from src.seed import seed_everything
from src.loss import get_loss
from src.model import * 

seed_everything(42)

def main():
      device = get_device()
      criterion = get_loss()
      dataloaders = get_test_loader(X_TEST_CD8_PATH, Y_TEST_CD8_PATH,BATCH_SIZE)
      # dataloaders = get_loader(CD8_X_TRAIN_PATH, CD8_Y_TRAIN_PATH, 
      #                          CD8_X_VALID_PATH, CD8_Y_VALID_PATH, 
      #                          CD8_X_TEST_PATH, CD8_Y_TEST_PATH, BATCH_SIZE)
      best_model = torch.load('model/ANN_lr_00001_dp_01__82_79.pt')

      # get each scores 
      loss_sum, acc, auroc, aupr, conf_matrix, labels , outputs = test_model(dataloaders, best_model, criterion,device)
      roc_scores, answers, preds = calculate_roc(dataloaders['test'], best_model, device)
      f1_score = f1_score(answers,preds,average='macro')

      # print scores 
      print({f"AUROC on test set: {auroc*100:.2f}"})
      print({f"AUPR on test set: {aupr*100:.2f}"})
      print({f"ACC on test set: {acc*100:.2f}"})
      print({f"F1 Score on test set: {f1_score*100:.2f}"})

      print({f"loss on test set: {loss_sum}"})
      print(conf_matrix)



if __name__ == '__main__':
    main()


"""https://learn-pytorch.oneoffcoder.com/rnn.html""" roc curve 참고 