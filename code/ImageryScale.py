#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:23:18 2017

@author: li
"""

import gdal
import numpy as np
import csv
import pandas as pd
import sklearn.preprocessing as sp
import osr

files_path = r"E:\Research\Experiment\qianshan\features\paras"
basic_tif_name = "para_"
basic_pos_name = "failedPos_"
basic_nor_name = "nor_para_"

#str_list = ["Red","Blue","Green","NIR","SWIR1","SWIR2"]
str_list = ["l8121"]

for file_index in str_list:
    #定义临时文件路径名
    tif_path = files_path + "\\" + basic_tif_name + file_index + ".tif"
    poi_path = files_path + "\\" + basic_pos_name + file_index + ".csv"
    tif_save_path = files_path + "\\" + basic_nor_name + file_index + ".tif"
    
    #读取影像并展开成Frame
    dataset = gdal.Open(tif_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    rows = dsmatrix.shape[1]
    cols = dsmatrix.shape[2]
    pars = dsmatrix.shape[0]
    data_array = dsmatrix[:,0,:]
    for irow in range(1,rows):
        tempmatirx = dsmatrix[:,irow,:]
        data_array = np.hstack((data_array,tempmatirx))
    data_frame = pd.DataFrame(data_array.T, index = range(0,rows*cols), columns = ['a','b','c','d'])
    
    #读取NaN值点信息
    del_list = []
    csv_reader = csv.reader(open(poi_path, encoding='utf-8'))
    for row in csv_reader:
        del_index = (int(row[0])-1) * 1901 + int(row[1]) - 1
        del_list.append(del_index)
    
    #进行数据归一化
    rest_frame = data_frame.drop(del_list)
    scaled_rest_matrix = sp.scale(rest_frame)
    scaled_rest_frame = pd.DataFrame(scaled_rest_matrix, index = rest_frame.index, columns = ['a','b','c','d'])
    scaled_full_frame = pd.DataFrame(index = data_frame.index, columns = data_frame.columns)
    scaled_full_frame.loc[rest_frame.index] = scaled_rest_matrix
    
    #将结果Frame转成Matrix
    result_matrix = np.empty_like(dsmatrix)
    for i in range(0,rows):
        result_matrix[:,i,:] = scaled_full_frame.loc[i*cols:(i+1)*cols-1,:].values.T  

    #将结果输出为tiff文件            
    driver=gdal.GetDriverByName("GTiff")
    driver.Register()
    outDataset = driver.Create(tif_save_path,cols,rows,4, gdal.GDT_Float64)
    
    #定义空间参考坐标系
    outDataset.SetGeoTransform( [ 422985, 30, 0, 3439515, 0, -30 ] )
    proj = osr.SpatialReference()
    proj.ImportFromProj4("+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
    outDataset.SetProjection( proj.ExportToWkt())
    for i in range(0,4):
        outDataset.GetRasterBand(i+1).WriteArray(result_matrix[i,:,:])
    outDataset = None
    
    print(file_index)
