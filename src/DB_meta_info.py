
import os 
os.chdir("/home/jhkim/2022_tomocube_timepoint/")

from src.DB_connection import Database
from src.preprocess import *
import pandas as pd
from pathlib import Path

def load_meta_data():
    database = Database()
    sql = """SELECT i.image_id, 
                i.file_name, 
                c.x,
                c.y,
                c.z
        FROM 2022_tomocube_sepsis_image i 
        LEFT JOIN 2022_tomocube_sepsis_image_center c 
        ON i.image_id = c.image_id;
        """

    database.execute_sql(sql)
    result = pd.DataFrame(database.execute_sql(sql))
    result.columns = ['num', 'file', 'x', 'y', 'z']
    return result


def get_meta_table():
    data = load_meta_data()
    data = data.dropna()
    df = data[data['file'].str.contains('Tomogram')]
    df['file'] = df.file.apply(lambda x : str(x)[:-5])
    return df

cd8_timepoint_1_path = Path('data/processed/cd8_timepoint_1')
cd8_timepoint_2_path = Path('data/processed/cd8_timepoint_2')


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

    
