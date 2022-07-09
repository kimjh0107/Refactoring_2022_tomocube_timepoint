from textwrap import indent
import numpy as np 
import matplotlib.pyplot as plt
from src.seed import seed_everything
from pathlib import Path 
from tqdm import tqdm

seed_everything(42)

def get_test_path(remove_patient:int, cell_type:str, target:str, type:str):
    return Path(f'npy/blind_test_{remove_patient}/test/test_{cell_type}_{target}_{type}.npy')

def get_test_pathes(remove_patient:int, celltype:str):
    test_cd8_x_test_path = get_test_path(remove_patient, celltype, 'x', 'test')
    test_cd8_y_test_path = get_test_path(remove_patient, celltype, 'y', 'test')
    return test_cd8_x_test_path, test_cd8_y_test_path


patient_2_x_cd8, patient_2_y_cd8 = get_test_pathes(1, 'cd8')  # patient 2
patient_3_x_cd8, patient_3_y_cd8 = get_test_pathes(2, 'cd8')  # patient 3
patient_4_x_cd8, patient_4_y_cd8 = get_test_pathes(3, 'cd8')  # patient 4
patient_5_x_cd8, patient_5_y_cd8 = get_test_pathes(4, 'cd8')  # patient 5 
patient_6_x_cd8, patient_6_y_cd8 = get_test_pathes(5, 'cd8')  # patient 6
patient_7_x_cd8, patient_7_y_cd8 = get_test_pathes(6, 'cd8')  # patient 7
patient_8_x_cd8, patient_8_y_cd8 = get_test_pathes(7, 'cd8')  # patient 8
patient_9_x_cd8, patient_9_y_cd8 = get_test_pathes(8, 'cd8')  # patient 9 
patient_10_x_cd8, patient_10_y_cd8 = get_test_pathes(9, 'cd8')  # patient 10
patient_11_x_cd8, patient_11_y_cd8 = get_test_pathes(10, 'cd8')  # patient 11


#patient_path = Path(f('patient_{}_x_{celltype}'))
def get_path(patient_ID:int, cell_type:str):
    return (f'patient_{patient_ID}_x_{cell_type}')

def make_plot(patient_ID:int, celltype:str):
    test_x_test_path, test_y_test_path = get_test_pathes(patient_ID, celltype)
    patient_id = np.load(test_x_test_path)
    patient_id = patient_id.flatten()
    plt.hist(patient_id)
    plt.gca().set(title=f'Patient_{patient_ID}_{celltype}')
    plt.savefig(f'plot/Patient_{patient_ID}_{celltype}.png')

def make_normalized_plot(patient_ID:int, celltype:str):
    test_x_test_path, test_y_test_path = get_test_pathes(patient_ID, celltype)
    patient_id = np.load(test_x_test_path)
    mean_value = np.mean(patient_id)
    std_value = np.std(patient_id)

    normalized_patient_id = (patient_id - mean_value) / (std_value)
    normalized_patient_id = normalized_patient_id.flatten()
    plt.hist(normalized_patient_id)
    # plt.gca().set(title=f'normalized_Patient_{patient_ID}_{celltype}')
    # plt.savefig(f'plot/Normalized_Patient_{patient_ID}_{celltype}.png')


def normalize_individual_image(img_list):

    result = []

    for i in range(len(img_list)):
         min_value = np.min(img_list[i])
         max_value = np.max(img_list[i])
         output = (img_list[i] - min_value) / (max_value - min_value)
         result.append(output)
    return result

def make_individual_normalized_plot(patient_ID:int, celltype:str):
    test_x_test_path, test_y_test_path = get_test_pathes(patient_ID, celltype)
    patient_id = np.load(test_x_test_path)
    normalized_img = normalize_individual_image(patient_id)
    normalized_img = np.array(normalized_img)
    normalized_patient_id = normalized_img.flatten()
    plt.hist(normalized_patient_id)
    # plt.gca().set(title=f'Individual normalized_Patient_{patient_ID}_{celltype}')
    # plt.savefig(f'plot/Individual_Normalized_Patient_{patient_ID}_{celltype}.png')


for i in range(10):
    make_plot(i+1, 'cd8')
    plt.clf()

for i in range(10):
    make_normalized_plot(i+1, 'cd8')
    plt.savefig('plot/all_patients_normalized.png')

for i in range(10):
    tqdm(make_individual_normalized_plot(i+1, 'cd8'))
    plt.savefig('plot/all_patients_individual_normalized.png')

#    plt.clf()







# patient_2_x_cd4, patient_2_y_cd4 = get_test_pathes(1, 'cd4')  # patient 2
# patient_3_x_cd4, patient_3_y_cd4 = get_test_pathes(2, 'cd4')  # patient 3
# patient_4_x_cd4, patient_4_y_cd4 = get_test_pathes(3, 'cd4')  # patient 4
# patient_5_x_cd4, patient_5_y_cd4 = get_test_pathes(4, 'cd4')  # patient 5 
# patient_6_x_cd4, patient_6_y_cd4 = get_test_pathes(5, 'cd4')  # patient 6
# patient_7_x_cd4, patient_7_y_cd4 = get_test_pathes(6, 'cd4')  # patient 7
# patient_8_x_cd4, patient_8_y_cd4 = get_test_pathes(7, 'cd4')  # patient 8
# patient_9_x_cd4, patient_9_y_cd4 = get_test_pathes(8, 'cd4')  # patient 9 
# patient_10_x_cd4, patient_10_y_cd4 = get_test_pathes(9, 'cd4')  # patient 10
# patient_11_x_cd4, patient_11_y_cd4 = get_test_pathes(10, 'cd4')  # patient 11


"""
< 추가로 진행해야 될 사항 > 
1. 각각의 환자에서 세포들 하나하나에 대한 RI 분포에 대해서 확인해보기 
2. 우리들 standarazation default 값이 1.34인데 값을 한번 확인해보도록 하기 
3. CD4, CD8이랑 분포가 좀 다른지에 대해서 확인해보기 
"""


mean_value = np.mean(patient_id)
std_value = np.std(patient_id)

normalized_patient_id = (patient_id - mean_value) / (std_value)
normalized_patient_id = normalized_patient_id.flatten()



test_x_test_path, test_y_test_path = get_test_pathes(4, 'cd8')
patient4_cd8 = np.load(test_x_test_path)

# standardization 방법 적용 
mean_value = np.mean(patient4_cd8)
std_value = np.std(patient4_cd8)
stand_patient4_cd8 = (patient4_cd8 - mean_value) / (std_value)

# individual min-max 
individual_p4_cd8 = normalize_individual_image(patient4_cd8)
individual_p4_cd8 = np.array(individual_p4_cd8)


# cell 하나하나 분리 
for i in range(50):
    plt.hist(patient4_cd8[i,:,:,:].flatten())


test_x_test_path, test_y_test_path = get_test_pathes(4, 'cd4')
patient4_cd4 = np.load(test_x_test_path)


for i in range(10):
    sns.displot(individual_p4_cd8[i,:,:,:].flatten(), alpha=0.9, kind='kde')
