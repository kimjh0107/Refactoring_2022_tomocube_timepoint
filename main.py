import os 
from config import * 
from src.seed import seed_everything
from src.logger import set_logger
from src.device import get_device
from src.model import *
from src.optimizer import *
from src.loss import get_loss
from src.dataloader import *
from src.earlystopping import EarlyStopping
from src.train import *

seed_everything(42)

def main():

    logger = set_logger()
    device = get_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
   # model = create_model(device)
    model = test(device)
    model = nn.DataParallel(model)

    optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY, ADAM_EPSILON)
    scheduler = create_scheduler(optimizer)
    criterion = get_loss()

    # dataloaders = get_loader(CD8_X_TRAIN_PATH, CD8_Y_TRAIN_PATH, 
    #                         CD8_X_VALID_PATH, CD8_Y_VALID_PATH, 
    #                         CD8_X_TEST_PATH, CD8_Y_TEST_PATH, BATCH_SIZE)


    train_dataset, valid_dataset, test_dataset = get_augmentation_dataset(CD8_X_TRAIN_PATH, CD8_Y_TRAIN_PATH, 
                                                                          CD8_X_VALID_PATH, CD8_Y_VALID_PATH, 
                                                                          CD8_X_TEST_PATH, CD8_Y_TEST_PATH)

    dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, BATCH_SIZE)


    logger.info("Start Training") 
    logger.info(f"learning_rate: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY},  epoch: {NUM_EPOCH}, batch_size: {BATCH_SIZE},dropout: {DROPOUT}")
    
    early_stopping = EarlyStopping(
        metric=EARLYSTOPPING_METRIC, # 'val_loss'
        mode=EARLYSTOPPING_MODE,     # 'min'
        patience=PATIENCE,           # 10
        path=MODEL_PATH,             
        verbose=False,
        )



    best_model, train_loss_history, val_loss_history =  train_model_v2(model, NUM_EPOCH, dataloaders, criterion, optimizer, device, scheduler, early_stopping)
  #  best_model, train_loss_history, val_loss_history =  train_model_v3(model, NUM_EPOCH, dataloaders, criterion, optimizer, device, scheduler)

    torch.save(best_model, 'model/model_1.pt') 
    save_loss_plot(train_loss_history, val_loss_history, LOSS_PATH)


if __name__ == '__main__':
    main()
