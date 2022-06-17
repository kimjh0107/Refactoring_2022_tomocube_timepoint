# %%
import torch.nn as nn
import torch.nn.functional as F 

def get_loss():
    return nn.CrossEntropyLoss(reduction='mean')
  #  return nn.BCELoss()
# %%
