import torch.optim as optim 

def get_optim(model, learning_rate, weight_decay, adam_epsilon):
    return optim.Adam(model.parameters(), 
                      lr = learning_rate, 
                      weight_decay = weight_decay,
                      eps = adam_epsilon)



def create_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)




