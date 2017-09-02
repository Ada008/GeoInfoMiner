#-*-coding:UTF-8-*-
"""
k-NN classifier on time series features.

This file contains the basic implementation of the k-nearest neighbors algorithm (k-NN)
algorithm on features generated from time series analysis.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017


from sklearn.neighbors import KNeighborsClassifier
import sys
import time

# Set the path of source files
sys.path.append(r'E:\Research\basic\github\GeoInfoMiner\code')
import utils

# If you have modified the "utils" file, then these two lines of code must be executed
from imp import reload
reload(utils)


n_neighbors=15

# Set the path of training samples
train_file_path = r'E:\Research\LandCoverMapping\Experiment\qianshan\Final_2\Classification\roi_and_csv_files\4_fit_tri\Samples_cord.csv'

# Set the path of image that is going to be classified
imagery_path = r'E:\Research\LandCoverMapping\Experiment\qianshan\Final_2\Classification\ImageData\out\com_nor_fit_tri.tif'

# Set the path where to output the final classified image
class_file_path = r'E:\Research\LandCoverMapping\Experiment\qianshan\Final_2\Classification\ImageData\out\class_knn_fit_tri.tif'

# set the split points according to csv files for training
split_points = [2, 32]

# Get all data ready for training and predicting 
# data_tuple contains: x_s, y_s, rows, cols, data_frame, geo_trans_list, 
#                      proj_str, num_bands
data_tuple = utils.prepare_data(imagery_path, train_file_path, split_points)

print("get tif data!")

# Train the classifier
clf_knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')

t1 = time.time()
clf_knn.fit(data_tuple[0], data_tuple[1])
t2 = time.time()
print("svm_fit_tri fitting time: " + str(t2 - t1))

# Predict lables based on image data
t3 = time.time()
z_ps=clf_knn.predict(data_tuple[2][0])
t4 = time.time()
print("svm_fit_tri predicting time: " + str(t4 - t3))

# Output the classification result
utils.output_lables_to_tif(z_ps, class_file_path, data_tuple[2][1], data_tuple[2][2], 
                           data_tuple[2][3], data_tuple[2][4], 1)