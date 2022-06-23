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
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

seed_everything(42)

def main():

    logger = set_logger()
    device = get_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    model = test(device)
    #model = nn.DataParallel(model)

    train_dataset, valid_dataset, test_dataset = get_augmentation_dataset(CD8_X_TRAIN_PATH, CD8_Y_TRAIN_PATH, 
                                                                          CD8_X_VALID_PATH, CD8_Y_VALID_PATH, 
                                                                          CD8_X_TEST_PATH, CD8_Y_TEST_PATH)
    dataloaders = get_augmentation_loader(train_dataset, valid_dataset, test_dataset, BATCH_SIZE)

    # gradcam target layer add 
    target_layers = [model.layer4[-1]]
    test_dataloaders = dataloaders['test']
    input_tensor = next(iter(test_dataloaders))

    # construct cam 
    cam = GradCAMPlusPlus(model=model,
                          target_layers=target_layers,
                          use_cuda=True)


    targets = ClassifierOutputTarget(281)               
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image("""rgb_img""", grayscale_cam, use_rgb=True)

    visualization


if __name__ == '__main__':
    main()
