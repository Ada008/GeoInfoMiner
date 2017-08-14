#-*-coding:UTF-8-*-
"""
k-nearest neighbors classifier.

This file contains the basic implementation of the k-nearest neighbors 
algorithm on Landsat 8 data.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017


from sklearn.neighbors import KNeighborsClassifier
import sys

# Set the path of source files
sys.path.append(r'E:\Research\basic\github\GeoInfoMiner\code')
import utils

# If you have modified the "utils" file, then these two lines of code must be executed
from imp import reload
reload(utils)

# Set the path of training samples
train_file_path = r'E:\Research\basic\github\GeoInfoMiner\data\training_sample.csv'

# Set the path of image that is going to be classified
imagery_path = r'E:\Research\basic\github\GeoInfoMiner\data\image_sample.tif'

# Set the path where to output the final classified image
class_file_path = r'E:\Research\basic\github\GeoInfoMiner\data\KNeighbors_class.tif'

# set the number the neighbours in k-NN
n_neighbors=15

# set the split points according to csv files for training
split_points = [2, 8]

# Get all data ready for training and predicting 
# data_tuple contains: x_s, y_s, rows, cols, data_frame, geo_trans_list, 
#                      proj_str, num_bands
data_tuple = utils.prepare_data(imagery_path, train_file_path, split_points)
    
# Train the classifier
clf_KNeighbors = KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
clf_KNeighbors.fit(data_tuple[0], [1])

# Predict lables based on image data
z_ps=clf_KNeighbors.predict(data_tuple[4])

# Output the classification result
utils.output_lables_to_tif(z_ps, class_file_path, data_tuple[2], data_tuple[3], 
                           data_tuple[5], data_tuple[6], 1)
