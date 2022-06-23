
from tqdm import tqdm 
import numpy as np 
from pathlib import Path

cd4_timepoint_list1_list = []
cd8_timepoint_list1_list = []

def get_timepoint1_new_list(cd4_list, cd4_path, cd8_list, cd8_path) -> list:

    for p in cd4_list['file_path']:
        cd4_timepoint_list1_list.append(f'{cd4_path}/{p}')
    cd4_list['file_path'] = cd4_timepoint_list1_list

    for p in cd8_list['file_path']:
        cd8_timepoint_list1_list.append(f'{cd8_path}/{p}')
    cd8_list['file_path'] = cd8_timepoint_list1_list

    return cd4_list, cd8_list

cd4_timepoint_list2_list = []
cd8_timepoint_list2_list = []

def get_timepoint2_new_list(cd4_list, cd4_path, cd8_list, cd8_path) -> list:
    
    for p in cd4_list['file_path']:
        cd4_timepoint_list2_list.append(f'{cd4_path}/{p}')
    cd4_list['file_path'] = cd4_timepoint_list2_list

    for p in cd8_list['file_path']:
        cd8_timepoint_list2_list.append(f'{cd8_path}/{p}')
    cd8_list['file_path'] = cd8_timepoint_list2_list

    return cd4_list, cd8_list

def read_npy(path:Path):
    return np.load(path)

def get_append_img_list(list):
    
    img_list = []

    for f in tqdm(list['file_path']):
        img = read_npy(f)
        img = img.transpose(1,2,0)
        img_list.append(img)
    return img_list 

def get_xyz_list(list):
    
    x_list = []
    y_list = []
    z_list = []   
    
    xyz = list[['x','y','z']].values

    for c in tqdm(range(len(xyz))):
        x,y,z = xyz[c]
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list 


def new_normalize(time_list):
    result = []

    for i in range(len(time_list)):
         min_value = np.min(time_list[i])
         max_value = np.max(time_list[i])
         output = (time_list[i] - min_value) / (max_value - min_value)
         result.append(output)
    return result


def split_dataset(timepoint1_list, timepoint2_list):
    lungT_lengths = len(timepoint1_list)
    sepsis_lengths = len(timepoint2_list)

    lungT_list = np.array(timepoint1_list)
    sepsis_list = np.array(timepoint2_list)

    lungT_labels = np.array([1 for _ in range(lungT_lengths)])
    sepsis_labels = np.array([0 for _ in range(sepsis_lengths)])

    x_train = np.concatenate((lungT_list[: int(lungT_lengths * 0.8)], 
                              sepsis_list[: int(sepsis_lengths * 0.8)]), axis=0)
    y_train = np.concatenate((lungT_labels[: int(lungT_lengths * 0.8)], 
                              sepsis_labels[: int(sepsis_lengths * 0.8)]), axis=0)

    x_valid = np.concatenate((lungT_list[int(lungT_lengths * 0.8) : int(lungT_lengths * 0.9)], 
                             sepsis_list[int(sepsis_lengths * 0.8) : int(sepsis_lengths * 0.9)]), axis=0)
    y_valid = np.concatenate((lungT_labels[int(lungT_lengths * 0.8) : int(lungT_lengths * 0.9)], 
                             sepsis_labels[int(sepsis_lengths * 0.8) : int(sepsis_lengths * 0.9)]), axis=0)

    x_test = np.concatenate((lungT_list[int(lungT_lengths * 0.9) : ], 
                             sepsis_list[int(sepsis_lengths * 0.9) : ]), axis=0)
    y_test = np.concatenate((lungT_labels[int(lungT_lengths * 0.9) : ], 
                             sepsis_labels[int(sepsis_lengths * 0.9) : ]), axis=0)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def save_cd4_preprocessed_data(x_train, y_train,x_valid,y_valid,x_test,y_test) : 
    np.save('data/npy/cd4_x_train.npy', x_train)
    np.save('data/npy/cd4_y_train.npy', y_train)
    np.save('data/npy/cd4_x_valid.npy', x_valid)
    np.save('data/npy/cd4_y_valid.npy', y_valid)
    np.save('data/npy/cd4_x_test.npy', x_test)
    np.save('data/npy/cd4_y_test.npy', y_test)

def save_cd8_preprocessed_data(x_train, y_train,x_valid,y_valid,x_test,y_test) : 
    np.save('data/npy/cd8_x_train.npy', x_train)
    np.save('data/npy/cd8_y_train.npy', y_train)
    np.save('data/npy/cd8_x_valid.npy', x_valid)
    np.save('data/npy/cd8_y_valid.npy', y_valid)
    np.save('data/npy/cd8_x_test.npy', x_test)
    np.save('data/npy/cd8_y_test.npy', y_test)

def save_tcell_preprocessed_data(x_train, y_train,x_valid,y_valid,x_test,y_test) : 
    np.save('data/npy/tcell_x_train.npy', x_train)
    np.save('data/npy/tcell_y_train.npy', y_train)
    np.save('data/npy/tcell_x_valid.npy', x_valid)
    np.save('data/npy/tcell_y_valid.npy', y_valid)
    np.save('data/npy/tcell_x_test.npy', x_test)
    np.save('data/npy/tcell_y_test.npy', y_test)




def split_only_test_dataset(timepoint1_list, timepoint2_list):
    lungT_lengths = len(timepoint1_list)
    sepsis_lengths = len(timepoint2_list)

    lungT_list = np.array(timepoint1_list)
    sepsis_list = np.array(timepoint2_list)

    lungT_labels = np.array([1 for _ in range(lungT_lengths)])
    sepsis_labels = np.array([0 for _ in range(sepsis_lengths)])

    x_train = np.concatenate((lungT_list[: int(lungT_lengths * 1)], 
                              sepsis_list[: int(sepsis_lengths * 1)]), axis=0)
    y_train = np.concatenate((lungT_labels[: int(lungT_lengths * 1)], 
                              sepsis_labels[: int(sepsis_lengths * 1)]), axis=0)

    return x_train, y_train


def save_cd4_test_preprocessed_data(x_test,y_test) : 
    np.save('data/npy/test_cd4_x_test.npy', x_test)
    np.save('data/npy/test_cd4_y_test.npy', y_test)

def save_cd8_test_preprocessed_data(x_test,y_test) : 
    np.save('data/npy/test_cd8_x_test.npy', x_test)
    np.save('data/npy/test_cd8_y_test.npy', y_test)








tcell_timepoint_list1_list = []
def get_tcell_timepoint1_new_list(tcell_list, tcell_path) -> list:
    for p in tcell_list['file_path']:
        tcell_timepoint_list1_list.append(f'{tcell_path}/{p}')
    tcell_list['file_path'] = tcell_timepoint_list1_list
    return tcell_list

tcell_timepoint_list2_list = []
def get_timepoint2_new_list(tcell_list, tcell_path) -> list:
    for p in tcell_list['file_path']:
        tcell_timepoint_list2_list.append(f'{tcell_path}/{p}')
    tcell_list['file_path'] = tcell_timepoint_list2_list
    return tcell_list