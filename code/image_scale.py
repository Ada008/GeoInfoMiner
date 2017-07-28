#-*-coding:UTF-8-*-
"""
Standardization of datasets.

Transform the data to center it by removing the mean value of each feature, 
then scale it by dividing non-constant features by their standard deviation.

All image must be scaled before subsequent classification progress

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import pandas as pd
import sklearn.preprocessing as sp
import sys

#Set the path of source files
sys.path.append(r'E:\Research\basic\github\GeoInfoMiner\code')
import utils

# If you have modified the "utils" file, then these two lines of code must be executed
from imp import reload
reload(utils)

# Basic setup
files_path = r"E:\Research\Experiment\qianshan\features\paras"
basic_tif_name = "para_"
basic_pos_name = "failedPos_"
basic_nor_name = "nor_para_"
str_list = ["Red","Blue","Green","NIR","SWIR1","SWIR2"]

# The author used 'for'loop in his experiment for convenience, but it is not 
# essential. Users can modify that according to their own need.
for file_index in str_list:
    # Define temporary path names. Also, modification is needed before executing
    tif_path = files_path + "\\" + basic_tif_name + file_index + ".tif"
    poi_path = files_path + "\\" + basic_pos_name + file_index + ".csv"
    tif_save_path = files_path + "\\" + basic_nor_name + file_index + ".tif"
    
    # Read original image
    rows, cols, data_frame, geo_trans_list, proj_str, num_bands = utils.read_image(tif_path)
    
    # Read positions of NaN values. If your image do not contain NaN values, 
    # this line of code should not be excuted.
    del_list=utils.read_NaN(poi_path)
    
    # Drop out NaN values
    rest_frame = data_frame.drop(del_list)
    
    # Data normalization
    scaled_rest_matrix = sp.scale(rest_frame)
    
    # Put the scaled data into DataFrame which can leave out NaN values and 
    # preserve scaled values without changing original data structure.
    scaled_rest_frame = pd.DataFrame(scaled_rest_matrix, 
                                     index = rest_frame.index, 
                                     columns = ['a','b','c','d'])
    scaled_full_frame = pd.DataFrame(index = data_frame.index, 
                                     columns = data_frame.columns)
    scaled_full_frame.loc[rest_frame.index] = scaled_rest_matrix
    
    # Output final result
    utils.output_frame_to_tif(scaled_full_frame, tif_save_path, rows, cols, 
                              geo_trans_list, proj_str, num_bands)
