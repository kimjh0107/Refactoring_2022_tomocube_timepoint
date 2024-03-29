
import pymysql 
from dotenv import load_dotenv
import os 

# %% 
load_dotenv(override = True)
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = int(os.getenv('MYSQL_PORT'))
MYSQL_DB = os.getenv('MYSQL_DB')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_CHARSET = os.getenv('MYSQL_CHARSET')

# %%
class Database: 
    def __init__(self,
                 host = MYSQL_HOST,
                 port = MYSQL_PORT,
                 db = MYSQL_DB,
                 user = MYSQL_USER,
                 password = MYSQL_PASSWORD,
                 charset = MYSQL_CHARSET,):
                 self.conn = self.create_connection(host, port, db, user, password, charset)
                 self.cursor = self.conn.cursor()
    
    def create_connection(self, host, port, db, user, password, charset):
        return pymysql.connect(
            host = host, 
            port = port,
            db = db,
            user = user, 
            password = password,
        )


    def execute_sql(self, sql: str):
        self.cursor.execute(sql)
        return self.cursor.fetchall()


# %%
