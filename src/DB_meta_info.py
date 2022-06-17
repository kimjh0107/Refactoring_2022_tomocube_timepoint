
from src.DB_connection import Database
from src.preprocess_tiff import *
import pandas as pd

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
