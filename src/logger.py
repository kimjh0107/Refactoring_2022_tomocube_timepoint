# %% 
import os
os.chdir("/home/jhkim/2022_tomocube_timepoint/")

import logging
from datetime import datetime


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][PID:%(process)d][%(module)s][%(funcName)s]%(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"log/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log")

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
# %%
