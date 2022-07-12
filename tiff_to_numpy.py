import os
os.chdir("/home/data/tomocube/")
from src.preprocess_tiff import * 
from tqdm import tqdm 

timepoint1_list = [Path('20220405/'), Path('20220421/'), Path('20220422/'), Path('20220519/'), 
                   Path('20220520/'), Path('20220526/'), Path('20220526_2/'), Path('20220602_3/'), Path('20220603_3/'), Path('20220607_2/')]
timepoint2_list = [Path('20220422_2'), Path('20220425/'), Path('20220523/'), Path('20220523_2/'), 
                   Path('20220527/'), Path('20220530/'), Path('20220603_2/'), Path('20220607_3/'), Path('20220608/')]

p4_timepoint1_list = Path('20220502/')
p4_timepoint2_list = Path('20220510')



def main():
    t1_cd4_table = []
    t1_cd8_table = []
    for i in tqdm(range(len(timepoint1_list))):
        cd4_table, cd8_table = get_qc_tiff(timepoint1_list[i])
        cd4_table = pd.DataFrame(cd4_table)
        cd8_table = pd.DataFrame(cd8_table)
        t1_cd4_table.append(cd4_table)
        t1_cd8_table.append(cd8_table)

    t2_cd4_table = []
    t2_cd8_table = []
    for i in tqdm(range(len(timepoint2_list))):
        cd4_table, cd8_table = get_qc_tiff(timepoint2_list[i])
        cd4_table = pd.DataFrame(cd4_table)
        cd8_table = pd.DataFrame(cd8_table)
        t2_cd4_table.append(cd4_table)
        t2_cd8_table.append(cd8_table)
    
    # merge dataframe in list 
    df_empty = pd.DataFrame({'num':[], 'quality':[], 'file_y':[]})    
    df_empty2 = pd.DataFrame({'num':[], 'quality':[], 'file_y':[]})    

    t1_cd4, t1_cd8 = get_merged_data(df_empty, df_empty2, t1_cd4_table, t1_cd8_table)
    t2_cd4, t2_cd8 = get_merged_data(df_empty, df_empty2, t2_cd4_table, t2_cd8_table)

    # 경로 다시 설정 
    for p in tqdm(t1_cd4['file_y']):
        process_timepoint1_cd4_sepsis_image(p) 

    for p in tqdm(t2_cd4['file_y']):
        process_timepoint2_cd4_sepsis_image(p) 

    for p in tqdm(t1_cd8['file_y']):
        process_timepoint1_cd8_sepsis_image(p)     

    for p in tqdm(t2_cd8['file_y']):
        process_timepoint2_cd8_sepsis_image(p)   


if __name__ == '__main__':
     main()





