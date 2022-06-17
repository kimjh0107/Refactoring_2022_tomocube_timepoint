import os 
os.chdir("/home/data/tomocube/")

from config import * 
from src.label_table import get_file_list 
from src.preprocess_numpy import * 
from src.crop import * 

def main():
    
    cd4_timepoint_list1, cd4_timepoint_list2, cd8_timepoint_list1, cd8_timepoint_list2 = get_file_list(TEST_CD4_TIMEPOINT_1_PATH,TEST_CD4_TIMEPOINT_2_PATH,
                                                                                                    TEST_CD8_TIMEPOINT_1_PATH,TEST_CD8_TIMEPOINT_2_PATH)

                                                                                                    
    cd4_timepoint_list1, cd8_timepoint_list1 = get_timepoint1_new_list(cd4_timepoint_list1, TEST_CD4_TIMEPOINT_1_PATH, 
                                                                    cd8_timepoint_list1, TEST_CD8_TIMEPOINT_1_PATH)

    cd4_timepoint_list2, cd8_timepoint_list2 = get_timepoint2_new_list(cd4_timepoint_list2, TEST_CD4_TIMEPOINT_2_PATH, 
                                                                    cd8_timepoint_list2, TEST_CD8_TIMEPOINT_2_PATH)

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

    cd4_x_train, cd4_y_train = split_only_test_dataset(cd4_timepoint1_bbox_list, cd4_timepoint2_bbox_list)
    cd8_x_train, cd8_y_train = split_only_test_dataset(cd8_timepoint1_bbox_list, cd8_timepoint2_bbox_list)

    save_cd4_test_preprocessed_data(cd4_x_train, cd4_y_train)
    save_cd8_test_preprocessed_data(cd8_x_train, cd8_y_train)




if __name__ == "__main__":
    main()
