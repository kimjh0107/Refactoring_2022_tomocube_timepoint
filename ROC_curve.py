"""ROC AUPRC curve plot"""

import numpy as np 
import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms 
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, f1_score
import torch.nn.functional as F
from src.dataloader import *
from src.loss import get_loss
from src.device import get_device
from src.optimizer import *
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

# %% 

# check accuracy 
def test_model(dataloaders, model, criterion, device) :
    dataloader = dataloaders['test']
    outputs = []
    labels = []
    model.eval()
    epoch_loss = 0
    for X,y in dataloader : 
        X = X.type(torch.FloatTensor).to(device)
        output = model(X)
        loss = criterion(output, y.to(device))
        outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader), labels, outputs

# calculate roc 
def calculate_roc(loader , model, device):
    num_correct = 0
    num_samples = 0
    answers = []
    preds = []
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0) 

            answers.extend(y.detach().cpu().numpy())
            preds.extend(predictions.detach().cpu().numpy())

       
    
    return roc_auc_score(answers, preds) , answers, preds

# %%
X_TEST_CD8_PATH = 'test_cd8_x_test.npy'
Y_TEST_CD8_PATH = 'test_cd8_y_test.npy'
BATCH_SIZE = 1

device = get_device()
criterion = get_loss()
dataloaders = get_test_loader(X_TEST_CD8_PATH, Y_TEST_CD8_PATH,BATCH_SIZE)

MODEL_PATH = 'model/ANN_lr_00001_dp_01__82_79.pt'
best_model = torch.load(MODEL_PATH)
loss_sum, labels, outputs = test_model(dataloaders, best_model, criterion,device)
roc_scores, answers, preds = calculate_roc(dataloaders['test'], best_model, device)

# %% 
# ROC curve 
output_probs = [x[1] for x in outputs]
fpr, tpr, _ = roc_curve(labels, output_probs)
roc_auc = roc_auc_score(labels, output_probs) 

plt.plot(
    fpr,
    tpr,
    color = 'darkorange',
    lw = 2, 
    label="CD8 patients 4 (area = %0.2f)" % roc_auc,
        )


plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# %% 
# AUPRC curve 
fpr2, tpr2, _ = precision_recall_curve(labels, output_probs)
aps_score = average_precision_score(labels, output_probs) 


display = PrecisionRecallDisplay(
    recall=tpr2,
    precision=fpr2,
    average_precision=aps_score,
)
display.plot()
_ = display.ax_.set_title("Micro-averaged over all classes")