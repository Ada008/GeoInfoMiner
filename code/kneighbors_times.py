import csv
import gdal
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from imp import reload
import sys
sys.path.append(r'E:\Research\Land Cover Mapping\Experiment\qianshan\Final\PythonCode')
import Xutils
reload(Xutils)


reference_path = r'E:\Research\Land Cover Mapping\Experiment\qianshan\Final\Classification\ReferenceData\6'
images_path = r'E:\Research\Land Cover Mapping\Experiment\qianshan\Final\Classification\ImageData\6'
class_save_path = images_path + '\\class_KNeighbors_times.tif'

n_neighbors=15
str_list = ["Red","Blue","Green","NIR","SWIR1","SWIR2"]
list_dsmatrix=[]
list_x_s=[]
cnt=0
    
for file_index in str_list:
    cnt+=1
    #定义临时文件路径名
    train_file_path = reference_path + "\\roi_"  + file_index + "\\Samples_cord.csv"
    imagery_path = images_path + "\\nor_para_"  + file_index + ".tif"

    #读取影像
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    list_dsmatrix.append(dsmatrix)

    #读取样本
    original_data_list = []
    csv_reader = csv.reader(open(train_file_path, encoding='utf-8'))
    for row in csv_reader:
        original_data_list.append(row)
    original_data_array = np.array(original_data_list)
    x_s = original_data_array[:,2:7] 
    list_x_s.append(x_s)
    
    if cnt ==1:
        y_s = original_data_array[:,7]
        pos_s=original_data_array[:,0:2]
        
    print(file_index)

#数据堆叠
dsmatrixs=np.concatenate(list_dsmatrix, axis=0)
x_ss=np.concatenate(list_x_s, axis=1)
samples_array=np.concatenate((x_ss,y_s.reshape(len(y_s),1)), axis=1)


pos_list=[]
for row in pos_s:
    pos_list.append(int(row[1]) * 1901 + int(row[0]))
x_ss_frame=pd.DataFrame(samples_array,index=pos_list)

#影像数据展开
rows = dsmatrixs.shape[1]
cols = dsmatrixs.shape[2]
pars = dsmatrixs.shape[0]
data_array = dsmatrixs[:,0,:]
for irow in range(1,rows):
    tempmatirx = dsmatrixs[:,irow,:]
    data_array = np.hstack((data_array,tempmatirx))
data_frame = pd.DataFrame(data_array.T)

#训练分类器
clf_RF = KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
clf_RF.fit(x_ss_frame[list(range(0,len(x_ss_frame.columns)-1))], 
                           x_ss_frame[len(x_ss_frame.columns)-1].values)
#用分类器进行预测
z_ps=clf_RF.predict(data_frame)

#创建一个零值的Frame用来存储分类结果
z_p_frame = pd.DataFrame(data = np.empty((rows*cols,1)), columns = ['lables'])
  
#z_p原是1维数组，将其转为2维后赋值
z_p_frame.loc[data_frame.index] = np.array(z_ps).reshape(len(z_ps),1)

#输出分类结果
Xutils.output_lables_to_tif(z_p_frame,class_save_path,rows,cols)