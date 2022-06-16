import os 
os.chdir("/home/data/tomocube/")

# from pathlib import Path
# import glob
# import pandas as pd 
# from src.DB_connection import Database
from src.preprocess_tiff import * 

test_path = Path('20220405/')



def main():
    """경로 설정"""
    t1_cd4_table, t1_cd8_table = get_qc_tiff("지정해준 Path")
    t2_cd4_table, t2_cd8_table = get_qc_tiff("지정해준 Path")


if __name__ == '__main__':
     main()