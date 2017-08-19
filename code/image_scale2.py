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
import os

#Set the path of source files
sys.path.append(r'E:\Research\basic\github\GeoInfoMiner\code')
import utils

# If you have modified the "utils" file, then these two lines of code must be executed
from imp import reload
reload(utils)

# Basic setup
str_input_path = r"E:\Research\LandCoverMapping\Experiment\qianshan\Final_2\Classification\ImageData\out\scale"
str_output_path = r"E:\Research\LandCoverMapping\Experiment\qianshan\Final_2\Classification\ImageData\out\scaled"

for i in os.walk(str_input_path):
    for str_name in i[2]:
        str_file_format = str_name[-3:]
        if str_file_format == 'tif':
            str_tif_path = os.path.join(i[0], str_name)
    
            # Read original image
            rows, cols, data_frame, geo_trans_list, proj_str, num_bands = utils.read_image(str_tif_path)

            # Data normalization
            scaled_matrix = sp.scale(data_frame)
            
            # Put the scaled data into DataFrame which can leave out NaN values and 
            # preserve scaled values without changing original data structure.
            scaled_frame = pd.DataFrame(scaled_matrix)
            
            # Set output file path
            str_out_tif = os.path.join(str_output_path, str_name)
            
            # Output final result
            utils.output_frame_to_tif(scaled_frame, str_out_tif, rows, cols, 
                                      geo_trans_list, proj_str, num_bands)
