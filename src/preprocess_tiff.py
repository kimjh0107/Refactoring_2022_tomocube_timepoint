import os 
from pathlib import Path
from src.DB_connection import Database
import glob
import pandas as pd 
import numpy as np 
import tifffile

def get_file_path_list(file_path:Path):
    return set(file_path.glob('*Tomogram.tiff'))

def get_filtered_file_list(file_list:list):
    df = pd.DataFrame(file_list)
    df.columns = ['file']
    df['new_file'] = df.file.apply(lambda x : str(x).split('/')[-1])
    cd4_df_filter = df[df['new_file'].str.contains("CD4")]
    cd8_df_filter = df[df['new_file'].str.contains("CD8")]
    return cd4_df_filter, cd8_df_filter

def get_db_qc_list():
    database = Database()
    database.execute_sql('SELECT i.image_id, i.file_name, q.quality FROM 2022_tomocube_sepsis_image i LEFT JOIN 2022_tomocube_sepsis_image_quality q ON q.image_id = i.image_id AND q.quality = 0;')
    quality_data = pd.DataFrame(database.execute_sql('SELECT i.image_id, i.file_name, q.quality FROM 2022_tomocube_sepsis_image i LEFT JOIN 2022_tomocube_sepsis_image_quality q ON q.image_id = i.image_id AND q.quality = 0;'))
    quality_data.columns = ['num','file','quality']
    quality_data = quality_data.dropna()
    return quality_data[quality_data['file'].str.contains('Tomogram')] 

def get_merge_table(sql_file_list, cell_file_list):
    return pd.merge(sql_file_list, cell_file_list, left_on='file', right_on='new_file')[['num', 'quality', 'file_y']]

def get_qc_tiff(file_path:Path):
    file_list = get_file_path_list(file_path)
    cd4_list, cd8_list = get_filtered_file_list(file_list)

    # get sql db file list 
    sql_file_list = get_db_qc_list()
    cd4_label_table = get_merge_table(sql_file_list, cd4_list)
    cd8_label_table = get_merge_table(sql_file_list, cd8_list)
    return cd4_label_table, cd8_label_table

def get_merged_data(df, df2, cd4_list, cd8_list) -> pd.DataFrame:
    for i in range(len(cd4_list)):
        df = pd.merge(df, cd4_list[i], on = ['num','quality','file_y'], how='outer' )
    for i in range(len(cd8_list)):
        df2 = pd.merge(df2, cd8_list[i], on = ['num','quality','file_y'], how='outer' )
    return df, df2


def read_image(Path):
    return tifffile.imread(Path)

def save_to_numpy(img_arr, file):
    np.save(file, img_arr)

# path to save img 
def get_cd4_sepsis_timepoint1_output_filename(p:Path):
    return Path(f'data/processed/cd4_timepoint_1/{p.stem}.npy')

def get_cd8_sepsis_timepoint1_output_filename(p:Path):
    return Path(f'data/processed/cd8_timepoint_1/{p.stem}.npy')

def get_cd4_sepsis_timepoint2_output_filename(p:Path):
    return Path(f'data/processed/cd4_timepoint_2/{p.stem}.npy')

def get_cd8_sepsis_timepoint2_output_filename(p:Path):
    return Path(f'data/processed/cd8_timepoint_2/{p.stem}.npy')

#### Process workflow ####
def process_timepoint1_cd4_sepsis_image(path):
    img_arr = read_image(path)
    save_to_numpy(img_arr, get_cd4_sepsis_timepoint1_output_filename(path))
    return img_arr

def process_timepoint1_cd8_sepsis_image(path):
    img_arr = read_image(path)
    save_to_numpy(img_arr, get_cd8_sepsis_timepoint1_output_filename(path))
    return img_arr

def process_timepoint2_cd4_sepsis_image(path):
    img_arr = read_image(path)
    save_to_numpy(img_arr, get_cd4_sepsis_timepoint2_output_filename(path))
    return img_arr

def process_timepoint2_cd8_sepsis_image(path):
    img_arr = read_image(path)
    save_to_numpy(img_arr, get_cd8_sepsis_timepoint2_output_filename(path))
    return img_arr