""" 모든 tcell에 대해서 전부 cd4, cd8 합쳐서 진행해보도록 하기"""
import os 
os.chdir("/home/data/tomocube/")

from config import * 
from src.label_table import * 
from src.preprocess_numpy import * 
from src.crop import * 

def main():
    
    tcell_timepoint_list1, tcell_timepoint_list2 = get_tcell_file_list(TIMEPOINT_1_PATH, TIMEPOINT_2_PATH)
    
    tcell_timepoint_list1 = get_tcell_timepoint1_new_list(tcell_timepoint_list1, TIMEPOINT_1_PATH)
    tcell_timepoint_list2 = get_timepoint2_new_list(tcell_timepoint_list2, TIMEPOINT_2_PATH)


    # get x,y,z 
    tcell_timepoint1_img_list = get_append_img_list(tcell_timepoint_list1)
    tcell_timepoint1_x_list, tcell_timepoint1_y_list, tcell_timepoint1_z_list = get_xyz_list(tcell_timepoint_list1)


    tcell_timepoint2_img_list = get_append_img_list(tcell_timepoint_list2)
    tcell_timepoint2_x_list, tcell_timepoint2_y_list, tcell_timepoint2_z_list = get_xyz_list(tcell_timepoint_list2)


    # crop image 
    tcell_timepoint1_bbox_list = get_cropped_image(tcell_timepoint1_img_list, tcell_timepoint1_x_list, tcell_timepoint1_y_list, tcell_timepoint1_z_list)
    tcell_timepoint2_bbox_list = get_cropped_image(tcell_timepoint2_img_list, tcell_timepoint2_x_list, tcell_timepoint2_y_list, tcell_timepoint2_z_list)


    tcell_timepoint1_bbox_list = new_normalize(tcell_timepoint1_bbox_list)
    tcell_timepoint2_bbox_list = new_normalize(tcell_timepoint2_bbox_list)

    x_train, y_train, x_valid,y_valid,x_test,y_test = split_dataset(tcell_timepoint1_bbox_list, tcell_timepoint2_bbox_list)

    save_tcell_preprocessed_data(x_train, y_train, x_valid,y_valid,x_test,y_test)


"""
train:1749
valid:219
test:220
"""

if __name__ == "__main__":
    main()
