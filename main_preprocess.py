import os 
os.chdir("/home/data/tomocube/")

from config import * 
from src.label_table import get_file_list 
from src.preprocess_numpy import * 
from src.crop import * 

def main():
    
    cd4_timepoint_list1, cd4_timepoint_list2, cd8_timepoint_list1, cd8_timepoint_list2 = get_file_list(CD4_TIMEPOINT_1_PATH,CD4_TIMEPOINT_2_PATH,
                                                                                                    CD8_TIMEPOINT_1_PATH,CD8_TIMEPOINT_2_PATH)
    
    cd4_timepoint_list1, cd8_timepoint_list1 = get_timepoint1_new_list(cd4_timepoint_list1, CD4_TIMEPOINT_1_PATH, 
                                                                    cd8_timepoint_list1, CD8_TIMEPOINT_1_PATH)
    cd4_timepoint_list2, cd8_timepoint_list2 = get_timepoint2_new_list(cd4_timepoint_list2, CD4_TIMEPOINT_2_PATH, 
                                                                    cd8_timepoint_list2, CD8_TIMEPOINT_2_PATH)

    # get x,y,z 
    cd4_timepoint1_img_list = get_append_img_list(cd4_timepoint_list1)
    cd4_timepoint1_x_list, cd4_timepoint1_y_list, cd4_timepoint1_z_list = get_xyz_list(cd4_timepoint_list1)

    cd8_timepoint1_img_list = get_append_img_list(cd8_timepoint_list1)
    cd8_timepoint1_x_list, cd8_timepoint1_y_list, cd8_timepoint1_z_list = get_xyz_list(cd8_timepoint_list1)

    cd4_timepoint2_img_list = get_append_img_list(cd4_timepoint_list2)
    cd4_timepoint2_x_list, cd4_timepoint2_y_list, cd4_timepoint2_z_list = get_xyz_list(cd4_timepoint_list2)

    cd8_timepoint2_img_list = get_append_img_list(cd8_timepoint_list2)
    cd8_timepoint2_x_list, cd8_timepoint2_y_list, cd8_timepoint2_z_list = get_xyz_list(cd8_timepoint_list2)

    # crop image 
    cd4_timepoint1_bbox_list = get_cropped_image(cd4_timepoint1_img_list, cd4_timepoint1_x_list, cd4_timepoint1_y_list, cd4_timepoint1_z_list)
    cd8_timepoint1_bbox_list = get_cropped_image(cd8_timepoint1_img_list, cd8_timepoint1_x_list, cd8_timepoint1_y_list, cd8_timepoint1_z_list)

    cd4_timepoint2_bbox_list = get_cropped_image(cd4_timepoint2_img_list, cd4_timepoint2_x_list, cd4_timepoint2_y_list, cd4_timepoint2_z_list)
    cd8_timepoint2_bbox_list = get_cropped_image(cd8_timepoint2_img_list, cd8_timepoint2_x_list, cd8_timepoint2_y_list, cd8_timepoint2_z_list)

    cd4_timepoint1_bbox_list = new_normalize(cd4_timepoint1_bbox_list)
    cd8_timepoint1_bbox_list = new_normalize(cd8_timepoint1_bbox_list)
    cd4_timepoint2_bbox_list = new_normalize(cd4_timepoint2_bbox_list)
    cd8_timepoint2_bbox_list = new_normalize(cd8_timepoint2_bbox_list)

    cd4_x_train, cd4_y_train, cd4_x_valid,cd4_y_valid,cd4_x_test,cd4_y_test = split_dataset(cd4_timepoint1_bbox_list, cd4_timepoint2_bbox_list)
    cd8_x_train, cd8_y_train, cd8_x_valid,cd8_y_valid,cd8_x_test,cd8_y_test = split_dataset(cd8_timepoint1_bbox_list, cd8_timepoint2_bbox_list)

    save_cd4_preprocessed_data(cd4_x_train, cd4_y_train, cd4_x_valid,cd4_y_valid,cd4_x_test,cd4_y_test)
    save_cd8_preprocessed_data(cd8_x_train, cd8_y_train, cd8_x_valid,cd8_y_valid,cd8_x_test,cd8_y_test)


"""
train:1749
valid:219
test:220
"""

if __name__ == "__main__":
    main()
