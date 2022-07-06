import os 
from config import * 
from src.seed import seed_everything
from src.logger import set_logger
from src.device import get_device
from src.model import *
from src.model_densenet import * 
from src.optimizer import *
from src.loss import get_loss
from src.dataloader import *
from src.earlystopping import EarlyStopping
from src.train import *
from src.model_densenet import * 

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



def main(remove_patient:int, celltype:str):

    logger = set_logger()
    device = get_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = test(device)
    model = create_densenet_model(device)
    model = nn.DataParallel(model)

    optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY, ADAM_EPSILON)
    scheduler = create_scheduler(optimizer)
    criterion = get_loss()


    cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path = get_pathes(remove_patient, celltype)

    train_dataset, valid_dataset, test_dataset = get_augmentation_dataset(cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path)

    dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, BATCH_SIZE)


    logger.info("Start Training") 
    logger.info(f"learning_rate: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY},  epoch: {NUM_EPOCH}, batch_size: {BATCH_SIZE},dropout: {DROPOUT}")
    
    early_stopping = EarlyStopping(
        metric=EARLYSTOPPING_METRIC,
        mode=EARLYSTOPPING_MODE,     
        patience=PATIENCE,           
        path=MODEL_PATH,             
        verbose=False,
        )



    best_model, train_loss_history, val_loss_history =  train_model_v2(model, NUM_EPOCH, dataloaders, criterion, optimizer, device, scheduler, early_stopping)

    torch.save(best_model, f'model/blind_test_model/blindtest_{remove_patient}_{celltype}_model_1.pt') 
    LOSS_PATH = f'model/plot/blindtest_{remove_patient}_{celltype}'
    save_loss_plot(train_loss_history, val_loss_history, LOSS_PATH)


if __name__ == '__main__':
    main(2, 'cd8')
    main(2, 'cd4')
