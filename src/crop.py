import numpy as np 
from typing import List 
import pandas as pd 
import pathlib 
from pathlib import Path 
from dataclasses import dataclass 
from tqdm import tqdm 

@dataclass 
class Boudingbox:
    x:float
    y:float 
    z:float 

    def start_point(self, point:int, image_crop_size:int) -> int:
        point -= image_crop_size // 2
        point = max(point, 0)
        return int(point)

    def end_point(self, point:int, image_crop_size:int, max_size:int) -> int:
        point += image_crop_size // 2
        return int(min(point, max_size))
    
    def full_image(self, image:np.array, image_crop_size:int, image_crop_size2:int):
        full_array = image[
            self.start_point(self.x, image_crop_size): 
            self.end_point(self.x, image_crop_size, 276), # for indexing last point

            self.start_point(self.y, image_crop_size): 
            self.end_point(self.y, image_crop_size, 276), # for indexing last point

            self.start_point(self.z, image_crop_size2): 
            self.end_point(self.z, image_crop_size2, 210) # for indexing last point
              ]

        if full_array.shape == (image_crop_size, image_crop_size, image_crop_size2):
            return full_array

        median_filled_image = np.full((image_crop_size, image_crop_size, image_crop_size2),np.median(full_array))
        print(full_array.shape)
        
        x_shape = full_array.shape[0]
        y_shape = full_array.shape[1]
        z_shape = full_array.shape[2]

        start_x = (image_crop_size - x_shape)//2
        start_y = (image_crop_size - y_shape)//2
        start_z = (image_crop_size2 - z_shape)//2 

        end_x = image_crop_size - start_x
        end_y = image_crop_size - start_y
        end_z = image_crop_size2 - start_z 

        end_x = end_x if x_shape % 2 == 0 else end_x -1
        end_y = end_y if y_shape % 2 == 0 else end_y -1
        end_z = end_z if z_shape % 2 == 0 else end_z -1

        median_filled_image[start_x:end_x, start_y:end_y, start_z:end_z] = full_array
        return median_filled_image




def load_cropped_image(meta_info:pd.DataFrame, image, image_crop_size:int, image_crop_size2:int) -> tuple[int, int, int, int]:
    try : x, y, z = meta_info[['x','y','z']].values[0]
    except : return
    bbox = Boudingbox(x,y,z)
    return bbox.full_image(image, image_crop_size, image_crop_size2)



def get_cropped_image(list:np.array, x_list, y_list, z_list):
    bbox_list = []
    for i in range(len(list)):
        bbox = Boudingbox(x_list[i], y_list[i], z_list[i])
        bbox_list.append(bbox)
    
    for i in tqdm(range(len(list))):
        bbox_list[i] = bbox_list[i].full_image(list[i], 128, 64)
    
    return bbox_list


