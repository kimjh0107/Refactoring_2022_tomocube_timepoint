import os
os.chdir("/home/data/tomocube/")
from src.preprocess_tiff import * 
from tqdm import tqdm 

timepoint1_list = Path('20220502/')
timepoint2_list = Path('20220510')



def main():
    
    # timepoint 1 
    cd4_table, cd8_table = get_qc_tiff(timepoint1_list)
    t1_cd4_table = pd.DataFrame(cd4_table)
    t1_cd8_table = pd.DataFrame(cd8_table)
    # timepoint 2 
    cd4_table, cd8_table = get_qc_tiff(timepoint2_list)
    t2_cd4_table = pd.DataFrame(cd4_table)
    t2_cd8_table = pd.DataFrame(cd8_table)

    
    # 경로 다시 설정 
    for p in tqdm(t1_cd4_table['file_y']):
        process_test_timepoint1_cd4_sepsis_image(p) 

    for p in tqdm(t2_cd4_table['file_y']):
        process_test_timepoint2_cd4_sepsis_image(p) 

    for p in tqdm(t1_cd8_table['file_y']):
        process_test_timepoint1_cd8_sepsis_image(p)     

    for p in tqdm(t2_cd8_table['file_y']):
        process_test_timepoint2_cd8_sepsis_image(p)   


if __name__ == '__main__':
     main()






