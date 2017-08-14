#-*-coding:UTF-8-*-
"""
Unibagging classification framework.

This is the implementation of Unibagging in a serial manner.
If you need to be more efficient, you need to implement it in parallel.
Also, you can contact the author for a parallel version.

The serial version of the algorithm does not require high performance for the 
computer(especially for Memory and cores of CPU), which can be run in personal
computer.

"""
# Authors: Jingge Xiao <jingge.xiao@gmail.com>
#
# Created on Fri Jul 28 10:21:50 2017

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import gc
import sys

# Set the path of source files
sys.path.append(r'E:\Research\basic\github\GeoInfoMiner\code')
import utils

n_neighbors = 15
n_lables = 7
# set the split points according to csv files for training
split_points = [2, 7]

# setup paths for all data files that will be used
files_path = r'E:\Research\LandCoverMapping\Experiment\qianshan\Final_2\Classification'
class_file_path = files_path + "\\ImageData\\class_Unibagging.tif"
vote_save_path = files_path + "\\OriginalVote.csv"

# define a list to store results of all base classifiers
z_ps=[]

# # define a list to store weights of all base classifiers
w_ps=[]

# process each band of data one by one for Memory saving
str_list = ["Red","Blue","Green","NIR","SWIR1","SWIR2"]
for file_index in str_list:
    
    # define temporary paths
    train_file_path = files_path + "\\roi_and_csv_files\\roi_"  + file_index + "\\Samples_cord.csv"
    imagery_path = files_path + "\\ImageData\\nor_para_"  + file_index + ".tif"
    
    # Get all data ready for training and predicting
    # data_tuple contains: x_s, y_s, rows, cols, data_frame, geo_trans_list, 
    #                      proj_str, num_bands
    data_tuple = utils.prepare_data(imagery_path, train_file_path, split_points)
    
    rows = data_tuple[2][0]
    cols = data_tuple[2][1]
    data_frame = data_tuple[2][2]
    geo_trans_list = data_tuple[2][3]
    proj_str = data_tuple[2][4]
    num_bands = data_tuple[2][5]
    
    # Read and generate training samples
    x_s = data_tuple[0]
    y_s = data_tuple[1]
    
    # Train the classifier
    clf_bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance'), 
                                    max_samples=0.25, max_features=1.0)
    clf_bagging.fit(x_s, y_s)
        
    # use each base classifier to predict
    base_clfs=clf_bagging.estimators_
    for base_clf in base_clfs:
        tmp_z_ps=base_clf.predict(data_frame)
        z_ps.append(tmp_z_ps)
        tmp_w_ps=base_clf.score(x_s,y_s.astype(int)-1)
        w_ps.append(tmp_w_ps)
    
    # collect memory
    del(data_tuple)
    gc.collect()
    
    print(file_index)


# integrate results of base classifiers
final_list=[]
lables=[np.arange(1,n_lables+1),np.zeros([n_lables])] 
cnt=0
for pos_index in range(0,len(z_ps[0])):
    lables[1]=np.zeros([n_lables])
    for vote_index in range(0,len(z_ps)):
        # The results of the original classification is the subscript values, 
        # plus 1, convert them to Lables
        lable_n = z_ps[vote_index][pos_index] + 1
        for lables_index in lables[0]:
            if lables_index==lable_n:
                lables[1][lables_index-1]+=w_ps[vote_index]
                continue
                   
    final_list.append(lables[0][np.argmax(lables[1])])

# Output the classification result
utils.output_lables_to_tif(final_list, class_file_path, rows, cols, geo_trans_list, proj_str, 1)
