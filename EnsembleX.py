# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:20:06 2017

@author: Data
"""
import csv
import gdal
import osr
import numpy as np
import pandas as pd

class_save_path = r'E:\Research\Experiment\qianshan\features\paras\EnsembleX_class.tif'
vote_save_path = r'E:\Research\Experiment\qianshan\features\train\OriginalVote.csv'
poi_path = r'E:\Research\Experiment\qianshan\features\train\failedPos_Blue.csv'
rows=2401
cols=1901

    
#读取原始票数信息
csv_reader = csv.reader(open(vote_save_path, encoding='utf-8'))

#定义最终存放最终票数统计结果的List
final_list=[]

lables=[np.arange(1,7),np.zeros([6])]
   
cnt=0
 
for vote_line in csv_reader:
    
    lables[1]=np.zeros([6])
        
    for index_str in vote_line:

        #原分类结果返回的是下标值，要加1，转换为Lable
        vote_index=int(index_str)
        vote_index += 1
        for lables_index in lables[0]:
            if lables_index==vote_index:
                lables[1][lables_index-1]+=1.0
                continue
                   
    final_list.append(lables[0][np.argmax(lables[1])])        


#创建一个零值的Frame用来存储分类结果
z_p_frame = pd.DataFrame(data = np.empty((rows*cols,1)), index = range(0,rows*cols), columns = ['lables'])

#读取NaN值点信息
del_list = []
csv_reader = csv.reader(open(poi_path, encoding='utf-8'))
for row in csv_reader:
    del_index = (int(row[0])-1) * cols + int(row[1]) - 1
    del_list.append(del_index)
        
rest_frame=z_p_frame.drop(del_list)    
#z_p原是1维数组，将其转为2维后赋值
z_p_frame.loc[rest_frame.index] = np.array(final_list).reshape(len(final_list),1)
                     
#将结果Frame转成Matrix
result_matrix = np.empty((rows,cols))
for i in range(0,rows):
    result_matrix[i,:] = z_p_frame.loc[i*cols:(i+1)*cols-1].T 

#将结果输出为tiff文件            
driver=gdal.GetDriverByName("GTiff")
driver.Register()
outDataset = driver.Create(class_save_path,cols,rows,1, gdal.GDT_Float64)

#定义空间参考坐标系
outDataset.SetGeoTransform( [ 422985, 30, 0, 3439515, 0, -30 ] )
proj = osr.SpatialReference()
proj.ImportFromProj4("+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
outDataset.SetProjection( proj.ExportToWkt())
outDataset.GetRasterBand(1).WriteArray(result_matrix)
outDataset = None               