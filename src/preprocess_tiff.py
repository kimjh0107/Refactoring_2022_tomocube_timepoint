import os 
from pathlib import Path
import glob
import pandas as pd 
from src.DB_connection import Database


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