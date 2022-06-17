
from src.DB_meta_info import load_meta_data
from config import * 
import pandas as pd
from pathlib import Path

def get_meta_table():
    data = load_meta_data()
    data = data.dropna()
    df = data[data['file'].str.contains('Tomogram')]
    df['file'] = df.file.apply(lambda x : str(x)[:-5])
    return df

def get_numpy_file_path_list(file_path:Path):
    return set(file_path.glob('*.npy'))

def create_sepsis_label_table(sepsis_list):
    table = pd.DataFrame({"file":list(sepsis_list), "label":1})
    table.to_csv('data/sepsis_label.csv')
    return table

def get_sepsis_table(path):
    sepsis_file_list = get_numpy_file_path_list(path)
    sepsis_label_table = create_sepsis_label_table(sepsis_file_list)
    sepsis_label_table['file_path'] = sepsis_label_table.file.apply(lambda x : str(x).split('/')[-1])
    sepsis_label_table['new_file'] = sepsis_label_table.file_path.apply(lambda x : str(x)[:-4])
    return sepsis_label_table

def get_merge_table(path):
    df = get_meta_table()
    sepsis_label_table = get_sepsis_table(path)
    return pd.merge(df, sepsis_label_table, left_on='file', right_on='new_file')[['num', 'x', 'y', 'z', 'label', 'file_path']]

def get_file_list(path1, path2, path3, path4)->list:
    table1 = get_merge_table(path1)
    table2 = get_merge_table(path2)
    table3 = get_merge_table(path3)
    table4 = get_merge_table(path4)
    return table1, table2, table3, table4
