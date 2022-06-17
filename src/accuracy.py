import torch 
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, f1_score
import numpy as np 


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

def calculate_correct_num(loader , model, device):
    num_correct = 0
    num_samples = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)  
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0) 
    
    return num_correct, num_samples


def calculate_test_roc_score(loader , model, device, criterion):
    num_correct = 0
    num_samples = 0
    model.eval() 
    with torch.no_grad():

        running_loss = 0.0

        for data, targets in loader['test']:
            data = data.to(device)
            targets = targets.to(device)
            
            # for roc scores 
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions ==targets).sum()
            num_samples += predictions.size(0)
            # for test loss 
            test_loss = criterion(scores, data)
            running_loss += test_loss.item() * data.size(0)
        
        test_loss = running_loss / len(loader['test'].dataset)
    
    return roc_auc_score(targets.cpu().numpy(), predictions.cpu().numpy()), test_loss


def calculate_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            predictions = scores.argmax(1)  
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0)  

    model.train()
    return num_correct/num_samples


def print_accuracy_scores(train_loader, valid_loader, test_loader, model, device):
    print(f"Accuracy on train set: {calculate_accuracy(train_loader, model,device)*100:.2f}",
        f"Accuracy on valid set: {calculate_accuracy(valid_loader, model,device)*100:.2f}",
        f"Accuracy on test set: {calculate_accuracy(test_loader, model,device)*100:.2f}", sep = '\n')




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
        outputs.extend(output.detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
        epoch_loss += loss.item()
    
    outputs = np.argmax(outputs, axis = 1)
    acc = (outputs == labels).sum() / outputs.shape[0]
    try : 
        auc = roc_auc_score(labels, outputs)
    except :
        auc = 0.5
    conf = confusion_matrix(labels, outputs)
    aupr = average_precision_score(labels,outputs)

    return epoch_loss / len(dataloader), acc, auc, aupr, conf, labels, outputs 