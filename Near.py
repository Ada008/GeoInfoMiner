import csv
import gdal
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

n_neighbors = 15
files_path = r'E:\Research\Experiment\qianshan\features\train'
vote_save_path = r'E:\Research\Experiment\qianshan\features\train\OriginalVote_last3.csv'
z_ps=[]

#str_list = ["Red","Blue","Green","NIR","SWIR1","SWIR2"]

str_list = ["NIR","SWIR1","SWIR2"]

for file_index in str_list:
    
    #定义临时文件路径名
    train_file_path = files_path + "\\roi_"  + file_index + "\\Samples_cord.csv"
    imagery_path = files_path + "\\nor_para_"  + file_index + ".tif"
    poi_path = files_path + "\\failedPos_"  + file_index + ".csv"
    
    #读取NaN值点信息
    del_list = []
    csv_reader = csv.reader(open(poi_path, encoding='utf-8'))
    for row in csv_reader:
        del_index = (int(row[0])-1) * 1901 + int(row[1]) - 1
        del_list.append(del_index) 

    #读取影像，展开成Frame，剔除空值点
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    rows = dsmatrix.shape[1]
    cols = dsmatrix.shape[2]
    pars = dsmatrix.shape[0]
    data_array = dsmatrix[:,0,:]
    for irow in range(1,rows):
        tempmatirx = dsmatrix[:,irow,:]
        data_array = np.hstack((data_array,tempmatirx))
    data_frame = pd.DataFrame(data_array.T, index = range(0,rows*cols), columns = ['a','b','c','d'])
    rest_frame = data_frame.drop(del_list)
    #读取并生成训练样本
    original_data_list = []
    csv_reader = csv.reader(open(train_file_path, encoding='utf-8'))
    for row in csv_reader:
        original_data_list.append(row)
    original_data_array = np.array(original_data_list)
    x_s = original_data_array[:,2:6] 
    y_s = original_data_array[:,6]
    
    
    #训练分类器
    clf_bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance'), 
                                    max_samples=0.25, max_features=1.0)
    clf_bagging.fit(x_s, y_s)
        
    #分别用基分类器进行预测
    base_clfs=clf_bagging.estimators_
    for base_clf in base_clfs:
        tmp_z_ps=base_clf.predict(rest_frame)
        z_ps.append(tmp_z_ps)
    
    print(file_index)

z_P_matrix=np.array(z_ps)

np.savetxt(vote_save_path, z_P_matrix.T, fmt='%d', delimiter = ',')