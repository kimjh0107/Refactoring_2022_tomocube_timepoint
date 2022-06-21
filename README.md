# [REFACTORING] Tomocube Timepoint Classification 
- Author: JongHyun Kim
- Date: 2022-06-16

## Objective 
Cell classification by each timepoint using every data 

### Data 
- timepoint 1: 20220405, 20220421, 20220422, 20220519, 20220520, 20220526 ,20220526_2, 20220602_3, 20220603_3, 20220607_2
    - patient ID: 1-1, 2-1, 3-1, 5-1, 6-1, 7-1, 8-1, 9-1, 10-1, 11-1
    - CD4: 1106, CD8: 1134
    
- timepoint 2: 20220422_2, 20220425, 20220523, 20220523_2, 20220527, 20220530, 20220603_2, 20220607_3, 20220608
    - patient ID: 2-2, 3-2, 5-2, 6-2, 7-2, 8-2, 9-2, 10-2, 11-2 
    - CD4: 1082, CD8: 1020


- patient4(test set): 20220502, 20220510 
    - timepoint 1: CD4: 34, CD8: 24
    - timepoint 2: CD4: 71, CD8: 59


### Result 
(dropout은 0.001로 우선은 동일하게 진행)
- CD8 lr 000001 seed 2022 : 86 , 59 
- CD8 lr 000001 seed 42 : 86 , 62
- CD8 lr 00001 seed 42 : 
- CD8 lr 0000001 seed 42 epoch100: 84, 51 epoch 100 -> 더 돌려야될듯 
- CD8 lr 0000001 seed 42 epoch1000: 85, 56 

[augmentation 적용해본 후 결과 확인] - 기존 x 4 
- CD8 lr 00001 seed 42 epoch 50: 84, 63    loss - 0.4 

기존 x 6 
- CD8 lr 00001 seed 42 epoch 100: 78,  77,   loss: 0.32 
- CD8 lr 0001, wd 0.1 seed 42 epoch 100:  90, 59 ,loss: 0.62
- CD8 lr 000001 seed 42 epoch 100: 

* 그냥 aug만하고 dataset은 늘리지 않았을 경우 -> 이렇게 진행했을 때 확실히 기존 test의경우는 
- CD8 lr 00001 seed 42 epoch 100:   87, 66,loss: 0.33 : aug_model_lr_0001_87_test_66.pt

* augmentation 만 적용 Resnet 모델 layer1,2,3,4 를 32,64,64,128로변경 + 그리고 Resnet50이 아니라 Renet152사용중 
- 1) CD8 lr 00001 , dp = 0.1, seed 42 epoch 200: 
        test: AUROC: 82.43, ACC:82.87, F1 Score: 82.60, loss: 0.43
        patient4: AUROC: 79.41, ACC:79.52, F1 Score: 76.89, loss: 0.37

- 2) CD8 lr 0001 , dp = 0.1, seed 42 epoch 200: 


우선은 추가적으로 optimizer도 변경했음 