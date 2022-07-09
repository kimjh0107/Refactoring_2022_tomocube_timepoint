import os 
from config import * 
from src.seed import seed_everything
from src.logger import set_logger
from src.device import get_device
from src.model import *
from src.model_densenet import * 
from src.optimizer import *
from src.loss import get_loss
from src.dataloader import CustomDataset, get_augmentation_loader
import torchvision.transforms as transforms 
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

def normalize_individual_image(img_list):
    result = []

    for i in range(len(img_list)):
        min_value = np.min(img_list[i])
        max_value = np.max(img_list[i])
        output = (img_list[i] - min_value) / (max_value - min_value)
        result.append(output)
    return result

def main(remove_patient:int, celltype:str):

    logger = set_logger()
    device = get_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
   # model = test(device)
    model = create_densenet_model(device)
    model = nn.DataParallel(model)

    optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY, ADAM_EPSILON)
    scheduler = create_scheduler(optimizer)
    criterion = get_loss()


    cd8_x_train_path, cd8_y_train_path, cd8_x_valid_path, cd8_y_valid_path, cd8_x_test_path, cd8_y_test_path = get_pathes(remove_patient, celltype)

    # img = np.load(cd8_x_train_path)
    # normalize_mean = np.mean(img)
    # normalize_std = np.std(img)

    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Normalize(mean = [normalize_mean], std= [normalize_std])
    ])

    test_transform = transforms.Compose([
       # transforms.Normalize(mean = [normalize_mean], std = [normalize_std])
                    ])



    def get_augmentation_dataset(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
        
        train_df = torch.Tensor(normalize_individual_image(np.load(x_train_path))).unsqueeze(1)
        train_dataset = CustomDataset(train_df, torch.LongTensor(np.load(y_train_path)), train_mode=True, transforms=train_transform)
    
        valid_df = torch.Tensor(normalize_individual_image(np.load(x_valid_path))).unsqueeze(1)
        valid_dataset = CustomDataset(valid_df ,torch.LongTensor(np.load(y_valid_path)), train_mode=True, transforms=test_transform)

        test_df = torch.Tensor(normalize_individual_image(np.load(x_test_path))).unsqueeze(1)
        test_dataset = CustomDataset(test_df , torch.LongTensor(np.load(y_test_path)), train_mode=True, transforms=test_transform)

        return train_dataset, valid_dataset, test_dataset

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

    torch.save(best_model, f'model/blind_test_model_individual/blindtest_{remove_patient}_{celltype}_model_1.pt') 
    LOSS_PATH = f'model/plot/ind_blindtest_{remove_patient}_{celltype}'
    save_loss_plot(train_loss_history, val_loss_history, LOSS_PATH)
    plt.clf()


if __name__ == '__main__':
    main(2, 'cd8')
    main(2, 'cd4')
    main(4, 'cd8')
    main(4, 'cd4')
